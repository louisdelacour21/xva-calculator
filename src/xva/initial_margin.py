"""Synthetic SIMM-style initial margin of a netting set, with a waterfall for missing data.

Not ISDA SIMM®, no affiliation with ISDA. It keeps the public functional form of a sensitivity-based margin (risk
weights, tenor correlations, sub-curve correlation, concentration factor), delta only, one currency. The parameters are
our own, calibrated on the public ECB history (market/im_params.csv, scripts/calibrate_im.py); a licensed user can paste
their official numbers in the im_params table.

What is known of a position sets how it is margined:
    1. full terms: exact ladder of the trade, by revaluation;
    2. DV01: unsigned ladder on the 6M curve at maturity (no maturity: on the vertex with the highest risk weight);
    3. notional and maturity: same, with DV01 = notional × annuity;
    4. notional only: BCBS-IOSCO schedule, worst bucket, without offset.
A blank direction gives a range, because the margin K is a norm: central = √(k² + Σ K(u_i)²), the root mean square over
independent signs; high = k + K(Σu) and low = max(k - K(Σu), 0), triangle inequalities that need correlations ≥ 0.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from .hw1f import HW1F
from .market import ROOT, Market
from .products import aged, schedule, value
from .risk import rate_shock, triangle

VERTICES = np.array([14 / 365, 1 / 12, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0])
LABELS = ["2w", "1m", "3m", "6m", "1y", "2y", "3y", "5y", "10y", "15y", "20y", "30y"]
CURVES = ("estr", "euribor6m")
SOURCE_OWN, SOURCE_USER = "Own calibration on ECB history", "Parameters supplied by the user"
PARAMS = ROOT / "im_params.csv"
SCHEDULE_WORST = 0.04  # MGN20.16, interest rate, duration of 5 years or more
LEVELS = {1: "Full terms", 2: "DV01 known", 3: "Notional and maturity", 4: "Notional only"}


# --- Parameters -------------------------------------------------------------------------------------------------


def moves(history: pd.DataFrame) -> pd.DataFrame:
    """Overlapping 10-business-day changes in bp at the vertices; history in decimals, columns = maturities in years.
    shortcut: AAA government rates, not swaps, and nothing under 3M (2w and 1m copy 3M), upgrade with money-market data."""
    d = history.diff(10).dropna() * 1e4
    return pd.DataFrame({v: d[max(T, 0.25)] for v, T in zip(LABELS, VERTICES)}, index=d.index)


def _sample(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Recent three years with the worst one-year stress window, and the end of that window."""
    end = d.abs().mean(axis=1).rolling(252).mean().idxmax()
    stress = d.loc[:end].tail(252)
    return d[d.index.isin(d.tail(756).index.union(stress.index))], end


def calibrate(history: pd.DataFrame) -> dict:
    """{rw: 99% quantile of absolute moves in bp, rho: correlation of moves, first and last dates, stress_end}."""
    d = moves(history)
    sample, end = _sample(d)
    return {"rw": sample.abs().quantile(0.99).to_numpy(), "rho": sample.corr().to_numpy(),
            "first": d.index[0], "last": d.index[-1], "stress_end": end}


def params_table(rw, rho) -> pd.DataFrame:
    rho = np.asarray(rho)
    return pd.DataFrame({"vertex": LABELS, "risk_weight_bp": np.round(rw, 2)}
                        | {f"rho_{v}": np.round(rho[:, j], 4) for j, v in enumerate(LABELS)})


def load_params() -> pd.DataFrame:
    return pd.read_csv(PARAMS, comment="#")


# --- Sensitivities ----------------------------------------------------------------------------------------------


def _pv(trades, market: Market) -> np.ndarray:
    # value at t = 0 does not depend on σ
    return value(trades, HW1F(0.03, 0.01, market.estr), market, 0.0, 0.0, 1.0)[0]


def _known_coupons(trades, market: Market) -> list:
    """Running coupons fixed once and for all: left free, a bump would move them although they carry no rate risk."""
    out = []
    for tr in trades:
        s, e = tr.floating_periods()
        k = np.flatnonzero((s < 0) & (e > 0))
        if len(k) and tr.last_fixing is None:
            P = market.euribor6m if tr.type == "irs" else market.estr
            d = e[k[0]] - s[k[0]]
            tr = replace(tr, last_fixing=float((1 / P(d) - 1) / d))
        out.append(tr)
    return out


def ladder(trades, market: Market) -> np.ndarray:
    """2 × 12 one-sided sensitivities in euros per +1 bp: row 0 €STR, row 1 EURIBOR 6M.
    shortcut: zero-rate bumps, consistent with parameters calibrated on zero rates; official SIMM uses market-rate PV01,
    upgrade by bumping par quotes and re-bootstrapping before using licensed parameters."""
    s = np.zeros((len(CURVES), len(VERTICES)))
    if not trades:
        return s
    trades = _known_coupons(trades, market)
    v0 = _pv(trades, market).sum()
    for i, curve in enumerate(CURVES):
        for k in range(len(VERTICES)):
            s[i, k] = _pv(trades, rate_shock(market, triangle(k, VERTICES), 1e-4, only=curve)).sum() - v0
    return s


def _annuity(T: float, market: Market) -> float:
    dates = schedule(0.0, T, 1)
    return float(np.diff(np.r_[0.0, dates]) @ market.estr(dates))


def _mapped(dv01: float, T: float | None, p: dict) -> np.ndarray:
    """Unsigned ladder of a DV01 on the 6M curve: at maturity, where a par swap's par delta sits, linear between the two
    closest vertices; without a maturity, on the vertex with the highest risk weight."""
    s = np.zeros((len(CURVES), len(VERTICES)))
    weights = np.eye(len(VERTICES))[int(np.argmax(p["rw"]))] if T is None else [np.interp(T, VERTICES, e) for e in np.eye(len(VERTICES))]
    s[1] = dv01 * np.asarray(weights)
    return s


def _position(q: dict, market: Market, p: dict) -> tuple:
    """(data level, unsigned DV01, unsigned ladder) of a position; level 4 has neither."""
    if q["dv01"] is not None:
        return 2, q["dv01"], _mapped(q["dv01"], q["maturity"], p)
    if q["maturity"] is None:
        return 4, None, None
    dv01 = q["notional"] * _annuity(q["maturity"], market) * 1e-4
    return 3, dv01, _mapped(dv01, q["maturity"], p)


# --- Aggregation ------------------------------------------------------------------------------------------------


def margin(s: np.ndarray, p: dict, cr: float | None = None) -> float:
    """K = √(Σ_ij φ_ij ρ_kl WS_i WS_j), WS = RW · s · CR, CR = max(1, √(|Σs| / T)) unless given."""
    cr = max(1.0, float(np.sqrt(abs(s.sum()) / p["threshold"]))) if cr is None else cr
    rw, phi = np.asarray(p["rw"], dtype=float), p["phi"]
    ws = np.r_[rw, rw] * s.ravel() * cr
    return float(np.sqrt(max(ws @ np.kron([[1.0, phi], [phi, 1.0]], np.asarray(p["rho"], dtype=float)) @ ws, 0.0)))


def _range(known: np.ndarray, unknown: list, add: float, p: dict) -> tuple:
    """(central, low, high): one concentration factor for the three, so the bounds hold; exact if no direction is unknown."""
    total = np.sum(unknown, axis=0) if unknown else np.zeros_like(known)
    cr = max(1.0, float(np.sqrt((abs(known.sum()) + total.sum()) / p["threshold"])))
    k, spread = margin(known, p, cr), margin(total, p, cr)
    central = float(np.sqrt(k**2 + sum(margin(u, p, cr) ** 2 for u in unknown)))
    return central + add, max(k - spread, 0.0) + add, k + spread + add


def _estimate(trades, positions, market: Market, p: dict) -> tuple:
    known, unknown, add = ladder(trades, market), [], 0.0
    for q in positions:
        level, _, s = _position(q, market, p)
        if level == 4:
            add += SCHEDULE_WORST * q["notional"]
        elif q["side"] is None:
            unknown.append(s)
        else:
            known = known + q["side"] * s
    return _range(known, unknown, add, p)


def synthetic_im(trades, positions, market: Market, p: dict) -> dict:
    """IM of trades in full terms and positions known in part: {central, low, high, ladder, positions, coverage}."""
    central, low, high = _estimate(trades, positions, market, p)
    full = ladder(trades, market)
    groups = {1: [full, [], 0.0, len(trades), 0], 2: [np.zeros_like(full), [], 0.0, 0, 0], 3: [np.zeros_like(full), [], 0.0, 0, 0],
              4: [np.zeros_like(full), [], 0.0, 0, 0]}
    signed, blank, rows = full.copy(), np.zeros(len(VERTICES)), []
    for q in positions:
        level, dv01, s = _position(q, market, p)
        g = groups[level]
        g[3] += 1
        if level == 4:
            alone = SCHEDULE_WORST * q["notional"]
            g[2] += alone
        else:
            alone = margin(s, p)
            if q["side"] is None:
                g[1].append(s)
                g[4] += 1
                blank += s[1]
            else:
                g[0] = g[0] + q["side"] * s
                signed += q["side"] * s
        rows.append({"position": q["name"], "side": {1: "payer", -1: "receiver"}.get(q["side"], "unknown"), "data level": LEVELS[level],
                     "DV01 used": np.nan if dv01 is None else dv01, "maturity": np.nan if q["maturity"] is None else q["maturity"],
                     "standalone margin": alone})
    alone = {lv: _range(g[0], g[1], g[2], p)[0] for lv, g in groups.items()}
    total = sum(alone.values())
    coverage = pd.DataFrame([{"data level": LEVELS[lv], "positions": g[3], "direction unknown": g[4], "standalone margin": alone[lv],
                              "share": alone[lv] / total if total else 0.0} for lv, g in groups.items() if g[3]])
    return {"central": central, "low": low, "high": high,
            "ladder": pd.DataFrame({"vertex": LABELS, "€STR": signed[0], "EURIBOR 6M": signed[1], "direction unknown": blank}),
            "positions": pd.DataFrame(rows), "coverage": coverage}


def _schedule_rate(duration: float) -> float:
    return 0.01 if duration < 2 else (0.02 if duration < 5 else SCHEDULE_WORST)  # MGN20.16, interest rate


def data_levels(trades, market: Market, p: dict, V0s) -> pd.DataFrame:
    """The same netting set margined as if less and less were known: what each missing field costs."""
    frozen = _known_coupons(trades, market)
    dv01 = _pv(frozen, rate_shock(market, np.ones_like, 1e-4)) - _pv(frozen, market)
    annuity = [_annuity(tr.maturity, market) for tr in trades]
    proxy = [tr.nominal * a * 1e-4 for tr, a in zip(trades, annuity)]
    signed = lambda x, sides: sum(sd * _mapped(abs(v), tr.maturity, p) for v, sd, tr in zip(x, sides, trades))
    nominal = sum(tr.nominal for tr in trades)
    # shortcut: "duration" of MGN20.16 taken as the annuity of the remaining life on €STR, the text does not define it further
    gross = sum(tr.nominal * _schedule_rate(a) for tr, a in zip(trades, annuity))
    rc = np.maximum(V0s, 0.0).sum()
    ngr = max(float(np.sum(V0s)), 0.0) / rc if rc > 0 else 1.0
    zeros = np.zeros((len(CURVES), len(VERTICES)))
    rows = [("Full terms", "Exact ladder, by revaluation", _range(ladder(trades, market), [], 0.0, p)),
            ("DV01, maturity, direction", "True DV01 placed at maturity", _range(signed(dv01, np.sign(dv01)), [], 0.0, p)),
            ("Notional, maturity, direction", "DV01 = notional × annuity, at maturity",
             _range(signed(proxy, [tr.side for tr in trades]), [], 0.0, p)),
            ("Notional, maturity", "Same, each direction unknown: range",
             _range(zeros, [_mapped(v, tr.maturity, p) for v, tr in zip(proxy, trades)], 0.0, p)),
            ("Notional", "Regulatory schedule, worst bucket, gross", (SCHEDULE_WORST * nominal,) * 3),
            ("Notional, maturity (schedule)", "Regulatory schedule by duration, gross", (gross,) * 3),
            ("Notional, maturity, MtM (schedule)", "Schedule with netting: 0.4 gross + 0.6 NGR gross", (0.4 * gross + 0.6 * ngr * gross,) * 3)]
    df = pd.DataFrame([{"data available": a, "method": b, "IM": c, "low": lo, "high": hi} for a, b, (c, lo, hi) in rows])
    df["gap to full terms %"] = (df["IM"] / df["IM"].iloc[0] - 1) * 100 if df["IM"].iloc[0] > 0 else np.nan
    return df


# --- Profile and backtest ---------------------------------------------------------------------------------------


def _forward(market: Market, u: float) -> Market:
    return Market(lambda T: market.estr(u + np.asarray(T, dtype=float)) / market.estr(u),
                  lambda T: market.euribor6m(u + np.asarray(T, dtype=float)) / market.euribor6m(u))


def _aged_positions(positions, market: Market, fwd: Market, u: float) -> list:
    """Positions with a maturity run off; a given DV01 follows the forward annuity; without a maturity they stay flat."""
    out = []
    for q in positions:
        if q["maturity"] is None:
            out.append(q)
        elif q["maturity"] > u:
            dv01 = q["dv01"] and q["dv01"] * _annuity(q["maturity"] - u, fwd) / _annuity(q["maturity"], market)
            out.append(q | {"maturity": q["maturity"] - u, "dv01": dv01})
    return out


def im_profile(trades, positions, market: Market, p: dict, base, n: int = 12) -> pd.DataFrame:
    """Central, low and high IM on the dates base, computed on n + 1 dates and interpolated.
    shortcut: along today's forward curves, no dependence on the path, fine for a linear book, upgrade for options."""
    dates = np.linspace(0.0, float(base[-1]), n + 1)
    rows = []
    for u in dates:
        fwd = _forward(market, u) if u > 0 else market
        rows.append(_estimate(aged(trades, u), _aged_positions(positions, market, fwd, u), fwd, p))
    rows = np.array(rows)
    return pd.DataFrame({k: np.interp(base, dates, rows[:, i]) for i, k in enumerate(("central", "low", "high"))},
                        index=pd.Index(base, name="years"))


def backtest(s: np.ndarray, p: dict, history: pd.DataFrame) -> dict:
    """Share of 10-day moves whose P&L on the ladder beats the margin (about 1% expected on the calibration sample)."""
    d = moves(history)
    sample, _ = _sample(d)
    pnl = np.abs(d.to_numpy() @ s.sum(0))
    K = margin(s, p)
    return {"margin": K, "sample": float((pnl[d.index.isin(sample.index)] > K).mean()), "sample days": len(sample),
            "history": float((pnl > K).mean()), "history days": len(d), "worst": float(pnl.max())}
