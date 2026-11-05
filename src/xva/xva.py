"""Credit (survival bootstrapped on CDS spreads) and CVA, DVA, FCA, FBA adjustments; incremental and allocation.

CVA = (1-R_C) Σ EE*(t̄_i)·[S_C(t_{i-1}) - S_C(t_i)]·S_B(t_i)      DVA: symmetric with NEE*, S_B, R_B
FCA = Σ ℓ_i·EE*(t̄_i)·S_C·S_B·Δt_i   FBA = Σ ℓ_i·NEE*(t̄_i)·S_C·S_B·Δt_i,   ℓ = max(s_F - (1-R_B)·λ_B, 0)
EE*(t̄_i) = mean of the two ends of the interval (trapezoid), on the base grid, with E(0) at t = 0.
Adjusted value = V - CVA + DVA - FCA + FBA.
MVA = Σ ℓ_i·S_C·S_B·Δt_i·P(0,t̄_i)·IM(t̄_i)   KVA = hurdle·Σ K(t̄_i)·P(0,t̄_i)·S_C·S_B·Δt_i, IM and K deterministic.
"""
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import brentq

from .exposure import exposure
from .market import ROOT, Curve
from .products import cube_mtm, index, value


@dataclass
class Survival:
    """Piecewise-constant λ: λ_i on (pillars[i-1], pillars[i]], extended beyond the last pillar."""
    pillars: np.ndarray
    lam: np.ndarray
    R: float = 0.4
    spreads: dict | None = None  # original quotes, to re-bootstrap after a bump

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        bounds = np.r_[0.0, self.pillars[:-1]]
        # ∫_0^t λ: share of each piece, the last one without a right bound
        dur = np.clip(t[..., None] - bounds, 0, np.r_[np.diff(np.r_[0.0, self.pillars])[:-1], np.inf])
        return np.exp(-dur @ self.lam)


def risk_free() -> Survival:
    return Survival(np.array([1.0]), np.array([0.0]))


def cds_spread(survival: Survival, P: Curve, times) -> float:
    """Par spread, MidPointCdsEngine convention: default and payment in the middle of each period,
    accrued coupon paid at default. times = payment dates (years), protection from 0."""
    b = np.asarray(times, dtype=float)
    a = np.r_[0.0, b[:-1]]
    m, d = (a + b) / 2, b - a
    dpd = survival(a) - survival(b)
    protection = (1 - survival.R) * np.sum(dpd * P(m))
    annuity = np.sum(d * survival(b) * P(b) + d / 2 * dpd * P(m))
    return protection / annuity


def cds_schedule(T: float, freq: int = 4) -> np.ndarray:
    # shortcut: regular quarterly schedule from today, no IMM dates (standard CDS) or date adjustment
    return np.arange(1, round(T * freq) + 1) / freq


def bootstrap(spreads: dict, P: Curve, R: float = 0.4, freq: int = 4) -> Survival:
    """λ per pillar, pillar after pillar, so that each par spread reprices exactly."""
    pillars = np.array(sorted(spreads), dtype=float)
    lam = np.zeros(len(pillars))
    for i, T in enumerate(pillars):
        def f(l):
            lam[i] = l
            return cds_spread(Survival(pillars[: i + 1], lam[: i + 1].copy(), R), P, cds_schedule(T, freq)) - spreads[T]
        try:
            lam[i] = brentq(f, 0.0, 5.0, xtol=1e-15)
        except ValueError:
            raise ValueError(f"no default intensity reprices the {T:g}Y spread: curve too inverted, or above 500%") from None
    return Survival(pillars, lam, R, dict(spreads))


@dataclass
class Credit:
    counterparty: Survival
    bank: Survival
    sF: float  # unsecured funding spread of the bank


def load_credit(path: Path = ROOT / "spreads.yaml", P: Curve = None) -> Credit:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    surv = lambda q: bootstrap({float(T): s * 1e-4 for T, s in q["spreads_bp"].items()}, P, q["R"])
    return Credit(surv(d["counterparty"]), surv(d["bank"]), d["bank"]["funding_bp"] * 1e-4)


def weights(t, credit: Credit) -> dict:
    """Weights per interval (0 = t_0, t_1], ..., (t_{m-1}, t_m] of the base grid."""
    u = np.r_[0.0, t]
    SC, SB = credit.counterparty(u), credit.bank(u)
    dt = np.diff(u)
    lam_b = np.log(SB[:-1] / SB[1:]) / dt  # mean intensity of the bank over the interval
    ell = np.maximum(credit.sF - (1 - credit.bank.R) * lam_b, 0)  # FBA does not double-count DVA
    return {"CVA": (1 - credit.counterparty.R) * (SC[:-1] - SC[1:]) * SB[1:],
            "DVA": (1 - credit.bank.R) * (SB[:-1] - SB[1:]) * SC[1:],
            "FCA": ell * SC[1:] * SB[1:] * dt, "FBA": ell * SC[1:] * SB[1:] * dt}


def _midpoints(X, x0):
    """Value in the middle of each interval (trapezoid), X paths × base dates, x0 value at t = 0."""
    X0 = np.broadcast_to(np.asarray(x0, dtype=float), X[:, :1].shape)
    return (np.c_[X0, X[:, :-1]] + X) / 2


def std_error(v) -> float:
    """Standard error of a mean over antithetic paths (i, i + n/2): average the pairs first."""
    h = len(v) // 2
    p = (v[:h] + v[h:2 * h]) / 2
    return float(p.std(ddof=1) / np.sqrt(h))


def contributions(expo: dict, credit: Credit) -> dict:
    """Contribution of each path to CVA, DVA, FCA, FBA (their mean gives the adjustment)."""
    w = weights(expo["t"], credit)
    DE = _midpoints(expo["D"] * expo["E"], expo["E0"])
    DN = _midpoints(expo["D"] * expo["N"], expo["N0"])
    return {"CVA": DE @ w["CVA"], "DVA": DN @ w["DVA"], "FCA": DE @ w["FCA"], "FBA": DN @ w["FBA"]}


def adjustments(expo: dict, credit: Credit, V0: float) -> dict:
    """{name: (value, standard error)} for CVA, DVA, FCA, FBA, and the adjusted value."""
    c = contributions(expo, credit)
    total = V0 - c["CVA"] + c["DVA"] - c["FCA"] + c["FBA"]
    return {k: (float(v.mean()), std_error(v)) for k, v in {**c, "Adjusted value": total}.items()}


def mva(im, im0: float, t, credit: Credit, P) -> float:
    """Cost of funding the initial margin posted, IM(t) on the dates t and im0 today.
    shortcut: the margin is assumed to earn the discount rate, upgrade with the rate paid by the custodian."""
    u = np.r_[0.0, t]
    return float(weights(t, credit)["FCA"] @ (P((u[:-1] + u[1:]) / 2) * (np.r_[im0, im[:-1]] + im) / 2))


def kva(K, t, credit: Credit, P, hurdle: float) -> float:
    """Cost of holding the capital K, given on 0 and on the dates t, at the hurdle rate."""
    u = np.r_[0.0, t]
    return float(hurdle * np.sum((K[:-1] + K[1:]) / 2 * P((u[:-1] + u[1:]) / 2) * credit.counterparty(t) * credit.bank(t) * np.diff(u)))


def pre_trade(V_ns, V0_ns: float, new, model, market, sim, base, credit: Credit, csa=None) -> dict:
    """ΔXVA of a new trade (or a list of them) on the netting set, same paths: only the new MtM is computed.

    V_ns: MtM of the netting set (paths × full grid), precomputed. The new trade must have its dates
    on the grid (otherwise recompute the grid and the netting set cube).
    """
    t0 = time.perf_counter()
    new = new if isinstance(new, list) else [new]
    V_new = cube_mtm(new, model, market, sim).sum(-1)
    V0_new = float(value(new, model, market, 0.0, 0.0, 1.0)[0].sum())
    before = contributions(exposure(V_ns, sim["D"], sim["t"], base, V0_ns, csa), credit)
    after = contributions(exposure(V_ns + V_new, sim["D"], sim["t"], base, V0_ns + V0_new, csa), credit)
    d = {k: (float((after[k] - before[k]).mean()), std_error(after[k] - before[k])) for k in before}
    total = sum((after[k] - before[k]) * s for k, s in {"CVA": 1, "DVA": -1, "FCA": 1, "FBA": -1}.items())
    return {"Δ": d, "total": (float(total.mean()), std_error(total)), "V0": V0_new, "seconds": time.perf_counter() - t0}


def allocation_euler(cube, D, t, base, V0, credit: Credit) -> np.ndarray:
    """Marginal CVA per trade, uncollateralised: (1-R) Σ ΔPD·S_B·E[D·V_j·1{V_NS > 0}]; sum = CVA."""
    k = np.array([index(t, u) for u in base])
    Vb, Db = cube[:, k, :], D[:, k]
    pos = Vb.sum(-1) > 0
    w = weights(base, credit)["CVA"]
    v0 = np.asarray(V0, dtype=float) * (np.sum(V0) > 0)
    return np.array([_midpoints(Db * Vb[:, :, j] * pos, v0[j]).mean(0) @ w for j in range(cube.shape[2])])
