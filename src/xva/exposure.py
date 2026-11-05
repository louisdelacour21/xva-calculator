"""Exposure grid, netting, collateral (CSA) and EE / NEE / PFE / EEPE metrics.

Collateral, classic model: C(t) = max(V(t-M) - H_C, 0) - max(-V(t-M) - H_B, 0).
shortcut: MTA added to the thresholds (H + MTA) instead of a collateral balance tracked path by path;
cash flows paid during the MPoR ignored ("advanced" MPoR in v2).
"""
import numpy as np

from .products import index

MPOR_10D = 14 / 365  # 10 business days ≈ 14 calendar days, in ACT/365F years


def grid(trades, mpors=(MPOR_10D,), step: float = 1 / 12):
    """(base, full): base = monthly step ∪ cash-flow dates; full = base ∪ t - M nodes (> 0)."""
    T = max(tr.maturity for tr in trades)
    flows = np.concatenate([np.r_[tr.start, tr.cashflows()] for tr in trades])
    base = np.unique(np.round(np.r_[np.arange(1, int(np.ceil(T / step)) + 1) * step, flows], 12))
    base = base[(base > 0) & (base <= T + 1e-12)]
    nodes = np.concatenate([base - m for m in mpors if m > 0] or [np.empty(0)])
    full = np.unique(np.round(np.r_[base, nodes[nodes > 0]], 12))
    return base, full


def exposure(V, D, t, base, V0: float, csa: dict | None = None, pfe_level: float = 0.975, im=None) -> dict:
    """Metrics on the base grid from V_NS (paths × full grid) and D(0, t).

    csa: {H_C, H_B, MTA, mpor (years), optional IA} in notional units; None = uncollateralised.
    IA: independent amount held by the bank (< 0 if posted), treated as unsegregated cash.
    im: two-way segregated initial margin on the base dates, held and posted, it reduces both exposures; None = none.
    EE, NEE, PFE undiscounted; EE* = E[D·E], NEE* = E[D·N] with their standard errors.
    """
    k = np.array([index(t, u) for u in base])
    Vb, Db = V[:, k], D[:, k]
    if csa is None:
        C, C0 = np.zeros_like(Vb), 0.0
    else:
        # before the first MPoR, the collateral reflects the initial value
        lag = np.array([np.full(len(V), V0) if u - csa["mpor"] <= 1e-12 else V[:, index(t, u - csa["mpor"])]
                        for u in base]).T
        hc, hb = csa["H_C"] + csa["MTA"], csa["H_B"] + csa["MTA"]
        C = np.maximum(lag - hc, 0) - np.maximum(-lag - hb, 0)
        ia = csa.get("IA", 0.0)
        C, C0 = C + ia, max(V0 - hc, 0) - max(-V0 - hb, 0) + ia
    m = np.zeros(len(base)) if im is None else np.asarray(im)
    E, N = np.maximum(Vb - C - m, 0), np.maximum(C - Vb - m, 0)
    n = len(V)
    ee = E.mean(0)
    dt = np.diff(base, prepend=0.0)
    one_year = base <= min(1.0, base[-1]) + 1e-12
    eee = np.maximum.accumulate(ee)
    return {
        "t": base, "EE": ee, "NEE": N.mean(0), "PFE": np.quantile(E, pfe_level, axis=0),
        "EE*": (Db * E).mean(0), "SE_EE*": (Db * E).std(0) / np.sqrt(n),
        "NEE*": (Db * N).mean(0), "SE_NEE*": (Db * N).std(0) / np.sqrt(n),
        "EEPE": float(np.sum(eee[one_year] * dt[one_year]) / np.sum(dt[one_year])),
        "E": E, "N": N, "D": Db,  # kept for incremental CVA and allocation (step 4)
        "E0": max(V0 - C0 - m[0], 0.0), "N0": max(C0 - V0 - m[0], 0.0),  # the margin of the first date stands in for today's
    }
