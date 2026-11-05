"""XVA sensitivities (bump and reprice, common random numbers) and capital: SA-CCR rates, reduced BA-CVA.

Deltas: 1 bp triangular bumps to the zero rates at the 1, 2, 5, 10, 30-year pillars (sum = parallel bump),
central, same draws, σ recalibrated; vega: swaption vols ±1% (relative); CS01: counterparty spreads
±1 bp per pillar, same exposures (credit does not enter the simulation).
"""
import numpy as np

from .exposure import exposure, grid
from .hw1f import HW1F, calibrate_q
from .market import Market
from .products import aged, cube_mtm, value
from .xva import Credit, bootstrap, contributions, std_error

PILLARS = np.array([1.0, 2.0, 5.0, 10.0, 30.0])


def triangle(k: int, pillars=PILLARS):
    """Weight of pillar k: 1 at the pillar, linear to the neighbours, flat beyond the extremes."""
    return lambda t: np.interp(t, pillars, np.eye(len(pillars))[k])


def rate_shock(market: Market, shape, h: float, only: str | None = None) -> Market:
    """Both curves bumped by h·shape(t) in continuous zero rate (rate bump, basis unchanged), or one with only="estr"
    or only="euribor6m"."""
    def f(P, name):
        if only not in (None, name):
            return P
        return lambda t: P(t) * np.exp(-h * shape(np.asarray(t, dtype=float)) * np.asarray(t, dtype=float))
    return Market(f(market.estr, "estr"), f(market.euribor6m, "euribor6m"))


def _rebootstrap(credit: Credit, P) -> Credit:
    """Spreads unchanged, survivals recomputed on the bumped discount curve."""
    rb = lambda s: s if s.spreads is None else bootstrap(s.spreads, P, s.R)
    return Credit(rb(credit.counterparty), rb(credit.bank), credit.sF)


def revalue(market, quotes, trades, base, t, n, seed, credit, csa, a=0.03, sigma=None) -> dict:
    """Full chain: Q calibration (or σ given), simulation (fixed seed), cube, exposure, contributions per path."""
    model = HW1F(a, sigma, market.estr) if sigma else calibrate_q(market.estr, market.euribor6m, quotes, a)[0]
    sim = model.simulate(t, n, seed=seed)
    cube = cube_mtm(trades, model, market, sim)
    V0 = value(trades, model, market, 0.0, 0.0, 1.0)[0].sum()
    return contributions(exposure(cube.sum(-1), sim["D"], t, base, V0, csa), _rebootstrap(credit, market.estr))


def _central(up: dict, down: dict) -> dict:
    """(up - down)/2 path by path: mean and standard error (common draws cancel most of the noise)."""
    return {k: (float(((up[k] - down[k]) / 2).mean()), std_error((up[k] - down[k]) / 2)) for k in up}


def sensitivities(trades, market, quotes, credit, csa=None, n=10_000, seed=0, h=1e-4, dvol=0.01,
                  a=0.03, sigma=None, step=1 / 12) -> dict:
    """{bump: {adjustment: (sensitivity, SE)}} for +1 bp per pillar and parallel, and +1% of vols (or of σ if given)."""
    base, t = grid(trades, mpors=(csa["mpor"],) if csa else (), step=step)
    rev = lambda m, c, s=sigma: revalue(m, c, trades, base, t, n, seed, credit, csa, a, s)
    shocks = {f"delta {p:g}Y": triangle(k) for k, p in enumerate(PILLARS)}
    shocks["delta parallel"] = np.ones_like
    out = {name: _central(rev(rate_shock(market, f, h), quotes), rev(rate_shock(market, f, -h), quotes))
           for name, f in shocks.items()}
    vol = lambda s: [(T, m, v * (1 + s)) for T, m, v in quotes]
    bumped = (lambda x: sigma * (1 + x)) if sigma else (lambda x: None)
    out["vega +1%"] = _central(rev(market, vol(dvol), bumped(dvol)), rev(market, vol(-dvol), bumped(-dvol)))
    return out


def cs01(expo: dict, credit: Credit, P, h: float = 1e-4) -> dict:
    """CS01 per counterparty spread pillar and parallel, without resimulation."""
    s = credit.counterparty
    rev = lambda d: contributions(expo, Credit(bootstrap(d, P, s.R), credit.bank, credit.sF))
    shock = lambda pillars, x: {T: v + x * (T in pillars) for T, v in s.spreads.items()}
    out = {f"CS01 {T:g}Y": _central(rev(shock([T], h)), rev(shock([T], -h))) for T in s.spreads}
    out["CS01 parallel"] = _central(rev(shock(list(s.spreads), h)), rev(shock(list(s.spreads), -h)))
    return out


# --- SA-CCR, interest rate class (CRE52) ------------------------------------------------------------------------

ALPHA, SF_IR, FLOOR = 1.4, 0.005, 0.05


def duration_factor(S: float, E: float) -> float:
    """SD = (e^{-0.05 S} - e^{-0.05 E}) / 0.05 (CRE52.34), S and E in years."""
    return (np.exp(-0.05 * S) - np.exp(-0.05 * E)) / 0.05


def sa_ccr(trades, V0: float, csa: dict | None = None) -> dict:
    """SA-CCR EAD of a netting set of EUR swaps (a single hedging set). csa: {H_C, H_B, MTA, mpor_days, optional IA}.

    δ = +1 for a fixed payer (gains when rates rise); maturity buckets < 1, 1-5, > 5 years.
    A seasoned trade counts from today (S = 0); IA is the net independent collateral amount (NICA).
    """
    ia = (csa or {}).get("IA", 0.0)

    def addon(mf):
        D = np.zeros(3)
        for tr in trades:
            b = 0 if tr.maturity < 1 else (1 if tr.maturity <= 5 else 2)
            D[b] += tr.side * tr.nominal * duration_factor(max(tr.start, 0.0), tr.maturity) * mf(tr)
        en = np.sqrt(D @ D + 1.4 * D[0] * D[1] + 1.4 * D[1] * D[2] + 0.6 * D[0] * D[2])
        return D, SF_IR * en

    def ead(rc, C, a):
        # a = 0 with a zero margin period of risk: no add-on for the multiplier to scale
        mult = min(1.0, FLOOR + (1 - FLOOR) * np.exp((V0 - C) / (2 * (1 - FLOOR) * a))) if a > 0 else 1.0
        return ALPHA * (rc + mult * a), mult

    D, a = addon(lambda tr: np.sqrt(min(max(tr.maturity, 10 / 250), 1.0)))  # unmargined MF (CRE52.48)
    e_nm, mult = ead(max(V0 - ia, 0.0), ia, a)
    r = {"D": D, "add-on": a, "RC": max(V0 - ia, 0.0), "multiplier": mult, "EAD": e_nm, "unmargined EAD": e_nm}
    if csa is None:
        return r
    hc, hb = csa["H_C"] + csa["MTA"], csa["H_B"] + csa["MTA"]
    C = max(V0 - hc, 0) - max(-V0 - hb, 0) + ia  # variation margin held today plus NICA
    D, a = addon(lambda tr: 1.5 * np.sqrt(csa["mpor_days"] / 250))  # margined MF (CRE52.52)
    rc = max(V0 - C, csa["H_C"] + csa["MTA"] - ia, 0.0)  # CRE52.18
    e_m, mult = ead(rc, C, a)
    return r | {"D": D, "add-on": a, "RC": rc, "multiplier": mult, "EAD": min(e_m, e_nm)}  # cap CRE52.2


# --- Reduced BA-CVA (MAR50) -------------------------------------------------------------------------------------

# MAR50.16, table 1: (sector, IG or HY/NR) -> RW, for the sectors offered in the app.
RW_MAR50 = {("sovereign", "IG"): 0.005, ("sovereign", "HY"): 0.02, ("financial", "IG"): 0.05,
            ("financial", "HY"): 0.12, ("industrial", "IG"): 0.03, ("industrial", "HY"): 0.07,
            ("other", "IG"): 0.05, ("other", "HY"): 0.12}


def effective_maturity(trades) -> float:
    # shortcut: notional-weighted residual maturity, floor 1 year, instead of the cash-flow duration (CRE32.47)
    return max(1.0, sum(tr.nominal * tr.maturity for tr in trades) / sum(tr.nominal for tr in trades))


def scva(rw: float, netting_sets) -> float:
    """SCVA_c = RW_c · Σ M·EAD·DF / α, DF = (1 - e^{-0.05 M}) / (0.05 M); netting_sets = [(M, EAD)]."""
    return rw * sum(M * e * (1 - np.exp(-0.05 * M)) / (0.05 * M) for M, e in netting_sets) / ALPHA


def ba_cva_reduced(scvas, rho: float = 0.5, ds: float = 0.65) -> float:
    """K = DS · √[(ρ Σ SCVA)² + (1 - ρ²) Σ SCVA²] (MAR50.14); regulatory CVA excludes DVA."""
    s = np.asarray(scvas, dtype=float)
    return float(ds * np.sqrt((rho * s.sum()) ** 2 + (1 - rho**2) * np.sum(s**2)))


def capital_profile(trades, mean_value, csa, rw, ccr_rw, dates) -> np.ndarray:
    """Minimum capital (8% of CCR RWA + reduced BA-CVA) of the aged trades, at the mean netting-set value of each date.
    shortcut: capital of the mean value, not the mean of the capital, upgrade by running SA-CCR on the paths."""
    out = []
    for u, v in zip(dates, mean_value):
        alive = aged(trades, u)
        if not alive:
            out.append(0.0)
            continue
        ead = sa_ccr(alive, v, csa)["EAD"]
        out.append(0.08 * ccr_rw * ead + ba_cva_reduced([scva(rw, [(effective_maturity(alive), ead)])]))
    return np.array(out)
