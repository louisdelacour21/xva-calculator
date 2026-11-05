"""Validation: Monte Carlo convergence, P against Q study, point-in-time (PIT) backtest of exposures under P.

Backtest: at each start date d0 of the ECB history, σ(P) calibrated on the year before (no look-ahead),
distribution of V(d0 + h) simulated under HW1F from that day's curve, then PIT = F_model(realised V), the realised V
being the value of the same cash flows on the curve observed at d0 + h. Non-overlapping windows (step = horizon).
"""
import time

import numpy as np
import pandas as pd
from scipy import stats

from .exposure import exposure, grid
from .hw1f import HW1F, calibrate_p, calibrate_q, swap_rate
from .market import ROOT, Market
from .products import _beta, cube_mtm, swap, value
from .xva import Credit, bootstrap, cds_schedule, cds_spread, contributions, std_error

ECB_MATURITIES = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"]
HORIZONS = {"1 month": 21, "3 months": 63, "1 year": 252}  # in business days of the history


def cva_paths(trades, model, market, credit, n, seed=0, csa=None, antithetic=True):
    """Contribution of each path to the CVA (their mean is the CVA)."""
    base, t = grid(trades, mpors=(csa["mpor"],) if csa else ())
    sim = model.simulate(t, n, seed=seed, antithetic=antithetic)
    V0 = value(trades, model, market, 0.0, 0.0, 1.0)[0].sum()
    V = cube_mtm(trades, model, market, sim).sum(-1)
    return contributions(exposure(V, sim["D"], t, base, V0, csa), credit)["CVA"]


def convergence(trades, model, market, credit, sizes=(1_000, 2_000, 5_000, 10_000, 20_000, 50_000)):
    """CVA, standard error with and without antithetics, time; gain = variance ratio (paths saved)."""
    rows = []
    for n in sizes:
        t0 = time.perf_counter()
        c = cva_paths(trades, model, market, credit, n)
        sec = time.perf_counter() - t0
        b = cva_paths(trades, model, market, credit, n, seed=1, antithetic=False)
        se, se_raw = std_error(c), float(b.std(ddof=1) / np.sqrt(n))
        rows.append({"N": n, "CVA": float(c.mean()), "SE": se, "SE without antithetics": se_raw,
                     "antithetic gain": (se_raw / se) ** 2, "seconds": sec})
    df = pd.DataFrame(rows)
    slope = np.polyfit(np.log(df["N"]), np.log(df["SE"]), 1)[0]
    return df, float(slope)


def pq_study(trades, market, quotes, credit, sigma_p, n=10_000, a=0.03):
    """Same CVA (same draws) under implied σ, historical σ, vols ±20%, a = 0.01 / 0.05, spreads ±25%."""
    q = lambda c=quotes, aa=a: calibrate_q(market.estr, market.euribor6m, c, aa)[0]
    vol = lambda s: [(T, m, v * s) for T, m, v in quotes]
    s = credit.counterparty
    spread = lambda f: Credit(bootstrap({T: v * f for T, v in s.spreads.items()}, market.estr, s.R), credit.bank,
                              credit.sF)
    ref = q()
    cases = {"Q, implied σ (reference)": (ref, credit), "P, historical σ": (HW1F(a, sigma_p, market.estr), credit),
             "vols +20%": (q(vol(1.2)), credit), "vols −20%": (q(vol(0.8)), credit),
             "a = 0.01": (q(aa=0.01), credit), "a = 0.05": (q(aa=0.05), credit),
             "spreads +25%": (ref, spread(1.25)), "spreads −25%": (ref, spread(0.75))}
    r = {}
    for name, (m, cr) in cases.items():
        c = cva_paths(trades, m, market, cr, n)
        r[name] = {"σ": m.sigma, "CVA": float(c.mean()), "SE": std_error(c)}
    for v in r.values():
        v["change"] = v["CVA"] / r["Q, implied σ (reference)"]["CVA"] - 1
    return pd.DataFrame(r).T


# --- PIT backtest ----------------------------------------------------------------------------------------------

def full_history() -> pd.DataFrame:
    """ECB spot rates (AAA, Svensson, continuous compounding) in decimal since 2004, columns = maturities in years."""
    cols = {}
    for m in ECB_MATURITIES:
        s = pd.read_csv(ROOT / "ecb_history" / f"SR_{m}.csv", index_col="TIME_PERIOD", parse_dates=True)
        cols[int(m[:-1]) / (12 if m[-1] == "M" else 1)] = s["OBS_VALUE"] / 100
    return pd.DataFrame(cols).dropna()


def ecb_curve(row: pd.Series):
    """t -> P(0, t): zero rates interpolated linearly between maturities, flat at the ends."""
    tau, r = np.array(row.index, dtype=float), row.to_numpy()
    return lambda t: np.exp(-np.interp(t, tau, r) * np.asarray(t, dtype=float))


def sample_portfolios(h: float, P) -> dict:
    """Three netting sets starting at the horizon, strikes at the forward par rate at d0.

    shortcut: single curve (ECB AAA for discounting and projection), the only long and free history;
    start at the horizon so that no fixing falls inside the window.
    """
    atm = lambda n: swap_rate(h, n, P, P)[0]
    return {"payer 5Y": [swap("A", "irs", 1, 10e6, atm(5), h, 5)],
            "receiver 10Y": [swap("B", "irs", -1, 10e6, atm(10), h, 10)],
            "steepener 2Y/10Y": [swap("C1", "irs", 1, 10e6, atm(10), h, 10),
                                 swap("C2", "irs", -1, 45e6, atm(2), h, 2)]}


def pit(P0, P1, h: float, sigma: float, a: float = 0.03, n: int = 2_000, seed: int = 0) -> dict:
    """PIT of each test portfolio: share of simulated V (HW1F, curve P0, horizon h) below the realised V (P1)."""
    m = HW1F(a, sigma, P0)
    x = m.simulate([h], n, seed=seed)["x"][:, 0]
    out = {}
    for name, tr in sample_portfolios(h, P0).items():
        v_sim = value(tr, m, Market(P0, P0), h, x, np.ones(n)).sum(1)
        seen = [swap(q.name, q.type, q.side, q.nominal, q.K, 0.0, q.maturity - q.start) for q in tr]
        v_real = value(seen, HW1F(a, sigma, P1), Market(P1, P1), 0.0, 0.0, 1.0)[0].sum()
        out[name] = float((v_sim < v_real).mean() + 0.5 * (v_sim == v_real).mean())
    return out


def ks_power(n_obs: int, factor: float = 1.25, level: float = 0.05, trials: int = 2_000, seed: int = 0):
    """Probability that KS rejects when the true dispersion is factor × the model's (Gaussian distribution)."""
    rng = np.random.default_rng(seed)
    u = stats.norm.cdf(factor * rng.standard_normal((trials, n_obs)))
    return float(np.mean([stats.kstest(x, "uniform").pvalue < level for x in u]))


def backtest_pit(hist: pd.DataFrame, horizons=HORIZONS, window: int = 252, a: float = 0.03, n: int = 2_000):
    """PIT per horizon and portfolio (non-overlapping windows) and KS table with power."""
    pits, rows = {}, []
    cal = hist[[1.0, 5.0, 10.0]]
    for h_name, k in horizons.items():
        res = []
        for i in range(window, len(hist) - k, k):
            d0, d1 = hist.index[i], hist.index[i + k]
            sigma = calibrate_p(cal.iloc[i - window:i + 1], a)
            res.append(pit(ecb_curve(hist.iloc[i]), ecb_curve(hist.iloc[i + k]), (d1 - d0).days / 365,
                           sigma, a, n, seed=i) | {"date": d0})
        df = pd.DataFrame(res).set_index("date")
        pits[h_name] = df
        power = ks_power(len(df))
        for p in df.columns:
            rows.append({"horizon": h_name, "portfolio": p, "windows": len(df),
                         "KS": stats.kstest(df[p], "uniform").statistic,
                         "p-value": stats.kstest(df[p], "uniform").pvalue,
                         "power (vol ×1.25)": power, "indicative": power < 0.5})
    return pits, pd.DataFrame(rows)


# --- Controls on each run ---------------------------------------------------------------------------------------

def forward_value(trades, market, u: float) -> float:
    """PV today of the netting set's flows paid strictly after u: what E[D(u)·V(u)] must equal under any model."""
    P, total = market.estr, 0.0
    for tr in trades:
        live = tr.fixed > u
        pv = -tr.K * float(np.sum(np.diff(np.r_[tr.start, tr.fixed])[live] * P(tr.fixed[live])))
        for s, e, b in zip(*tr.floating_periods(), _beta(tr, market)):
            if e <= u:
                continue
            if s >= 0:  # a coupon fixing on the path has this value too: the fixing cancels under the expectation
                pv += float(b * P(s) - P(e))
                continue
            curve = market.euribor6m if tr.type == "irs" else market.estr
            L = (1 / curve(e - s) - 1) / (e - s) if tr.last_fixing is None else tr.last_fixing
            pv += float((e - s) * L * P(e) if tr.type == "irs" else (1 - s * L) - P(e))
        total += tr.side * tr.nominal * pv
    return total


def _worst_z(samples, expected) -> float:
    """Largest |mean - expected| in standard errors, over the dates (columns); dates without any noise are skipped."""
    worst = 0.0
    for k in range(samples.shape[1]):
        se = std_error(samples[:, k])
        if se > 1e-9:
            worst = max(worst, abs(samples[:, k].mean() - expected[k]) / se)
    return worst


def live_checks(trades, market, sim, V, credit, bare, euler: float, fit, sigma_given, cva: tuple, rho, im_backtest: dict) -> pd.DataFrame:
    """Pass or fail on this run, not on a stored report. A failure is reported, never tuned away.
    bare: the uncollateralised exposure; euler: sum of the Euler contributions."""
    rows = []

    def row(check, value, limit, ok):
        rows.append({"check": check, "value": value, "limit": limit, "status": "n/a" if ok is None else "pass" if ok else "FAIL"})

    t, D = sim["t"], sim["D"]
    z = _worst_z(D, market.estr(t))
    row("Discount factors: E[D(t)] against the curve", f"{z:.2f} SE", "4 SE", z <= 4)
    z = _worst_z(D * V, [forward_value(trades, market, u) for u in t])
    row("Martingale: E[D(t)·V(t)] against the PV of later flows", f"{z:.2f} SE", "4 SE", z <= 4)

    miss = max(abs(cds_spread(s, market.estr, cds_schedule(T)) - q) for s in (credit.counterparty, credit.bank) for T, q in s.spreads.items()) * 1e4
    row("CDS spreads reprice from the bootstrapped survival", f"{miss:.5f} bp", "0.01 bp", miss <= 0.01)

    if len(fit) and not sigma_given:
        miss = float(np.abs(fit["error (bp)"]).max())
        row("Calibration: largest swaption vol error", f"{miss:.2f} bp", "5 bp", miss <= 5)
    else:
        row("Calibration: largest swaption vol error", "σ typed, no calibration", "5 bp", None)

    miss = abs(euler - float(contributions(bare, credit)["CVA"].mean()))
    row("Euler contributions sum to the uncollateralised CVA", f"{miss:,.4f} EUR", "0.01 EUR", miss <= 0.01)

    value_, se = cva
    row("CVA standard error against the CVA", f"{se / value_:.2%}" if value_ > 0 else "no CVA", "5%", se <= 0.05 * value_ if value_ > 0 else None)

    low = float(np.linalg.eigvalsh(np.asarray(rho, dtype=float)).min())
    row("Initial margin correlations: lowest eigenvalue", f"{low:.4f}", "above -0.001", low >= -1e-3)
    row("Initial margin backtest: exceedances, calibration sample", f"{im_backtest['sample']:.2%}", "2%", im_backtest["sample"] <= 0.02)
    return pd.DataFrame(rows)
