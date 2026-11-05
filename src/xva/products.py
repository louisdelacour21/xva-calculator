"""EURIBOR 6M IRS and €STR OIS in closed form per path, vectorised paths × trades.

Value of a trade = linear combination of P(t, T_k): at each date, a ZC matrix (paths × maturities)
times a weight matrix (maturities × trades), plus the running floating coupon, fixed on the path.
Convention: at date t, only cash flows paid strictly after t count (ex-coupon).
"""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

from .hw1f import HW1F
from .market import Market


@dataclass
class Trade:
    name: str
    type: str            # "irs" (EURIBOR 6M) or "ois" (compounded €STR)
    side: int            # +1 pays fixed, -1 receives fixed
    nominal: float
    K: float
    start: float         # < 0 for a seasoned trade, already running today
    fixed: np.ndarray    # fixed payment dates (ACT/365F years)
    floating: np.ndarray  # floating payment dates
    last_fixing: float | None = None  # seasoned: EURIBOR fixing (IRS) or annualised €STR accrued so far (OIS)

    @property
    def maturity(self) -> float:
        return float(max(self.fixed[-1], self.floating[-1]))

    def floating_periods(self):
        return np.r_[self.start, self.floating[:-1]], self.floating

    def cashflows(self) -> np.ndarray:
        return np.unique(np.r_[self.fixed, self.floating])


def schedule(start: float, T: float, freq: int) -> np.ndarray:
    """Payment dates rolled backward from maturity; the broken period comes first, merged if under 5% of a period
    (a calendar 10-year trade is 10.008 ACT/365F years: no 3-day coupon).
    shortcut: dates in ACT/365F years, no business-day calendar, take them from project 09 for real confirmations."""
    n = max(int(np.ceil((T - start) * freq - 0.05)), 1)
    return np.round(T - np.arange(n)[::-1] / freq, 12)


def swap(name, type, side, nominal, K, start, tenor, last_fixing=None) -> Trade:
    """Annual fixed leg; semi-annual (IRS) or annual (OIS) floating leg. start < 0: seasoned trade."""
    start, T = round(start, 12), round(start + tenor, 12)
    return Trade(name, type, side, nominal, K, start, schedule(start, T, 1),
                 schedule(start, T, 2 if type == "irs" else 1), last_fixing)


def load_portfolio(path: Path):
    """(trades, csa) from a YAML file; csa in euros and in business days of MPoR."""
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    side = {"payer": 1, "receiver": -1}
    trades = [swap(q["name"], q["type"], side[q["side"]], float(q["nominal"]), float(q["K"]),
                   float(q["start"]), float(q["tenor"])) for q in d["trades"]]
    return trades, d["csa"]


def _beta(trade: Trade, market: Market) -> np.ndarray:
    s, e = trade.floating_periods()
    if trade.type == "ois":
        return np.ones(len(e))
    return (market.euribor6m(s) / market.euribor6m(e)) / (market.estr(s) / market.estr(e))


def value(trades, model: HW1F, market: Market, t: float, x_t, D_t, state=None) -> np.ndarray:
    """value(t, state) interface: MtM (paths × trades) at date t.

    x_t, D_t: x(t) and D(0, t) per path; state(s) -> (x(s), D(0, s)) at past fixing dates
    (needed only for a running coupon, so never at t = 0).
    """
    x_t, D_t = np.atleast_1d(x_t), np.atleast_1d(D_t)
    mats = np.unique(np.concatenate([np.r_[tr.start, tr.cashflows()] for tr in trades]))
    mats = mats[mats >= t]
    W = np.zeros((len(mats), len(trades)))
    col = {u: i for i, u in enumerate(mats)}
    running = []
    for j, tr in enumerate(trades):
        w = tr.side * tr.nominal
        dk = np.diff(np.r_[tr.start, tr.fixed])
        for T, d in zip(tr.fixed, dk):
            if T > t:
                W[col[T], j] -= w * tr.K * d
        beta = _beta(tr, market)
        for s, e, b in zip(*tr.floating_periods(), beta):
            if s >= t:
                W[col[s], j] += w * b
                W[col[e], j] -= w
            elif e > t:
                running.append((j, s, e, b, w))
    PU = model.P(t, mats[None, :], x_t[:, None]) if len(mats) else np.zeros((len(x_t), 0))
    V = PU @ W
    for j, s, e, b, w in running:
        if s < 0:  # fixed before today: the coupon is known, nothing to read on the path
            P = market.euribor6m if trades[j].type == "irs" else market.estr
            L = trades[j].last_fixing
            L = (1 / P(e - s) - 1) / (e - s) if L is None else L  # no fixing given: today's rate for that period
            V[:, j] += w * ((e - s) * L * PU[:, col[e]] if trades[j].type == "irs"
                            else (1 - s * L) / D_t - PU[:, col[e]])
            continue
        x_s, D_s = state(s)
        if trades[j].type == "irs":
            # δ·L paid at e, L fixed at s on the path: 1 + δ·L = β / P(s, e)
            V[:, j] += w * (b / model.P(s, e, x_s) - 1) * PU[:, col[e]]
        else:
            # compounded €STR paid at e: worth the accrued exp(∫_s^t r) = D(0,s)/D(0,t) at t, undiscounted
            # (the project sheet writes e^{∫r}·P(t,T), which is wrong: caught by test_martingale_mtm)
            V[:, j] += w * (D_s / D_t - PU[:, col[e]])
    return V


def par_rate(trade: Trade, model: HW1F, market: Market) -> tuple[float, float]:
    """(par fixed rate, annuity on the remaining fixed dates) seen from 0."""
    unit = replace(trade, side=1, nominal=1.0, K=0.0)
    d = np.diff(np.r_[trade.start, trade.fixed])
    live = trade.fixed > 0
    annuity = float(np.sum(d[live] * market.estr(trade.fixed[live])))
    return float(value([unit], model, market, 0.0, 0.0, 1.0)[0, 0]) / annuity, annuity


def cube_mtm(trades, model: HW1F, market: Market, sim: dict) -> np.ndarray:
    """MtM cube (paths × dates × trades) on the simulation grid, which must contain the fixing dates."""
    t = sim["t"]

    def state(s):
        if s == 0:
            n = sim["x"].shape[0]
            return np.zeros(n), np.ones(n)
        k = index(t, s)
        return sim["x"][:, k], sim["D"][:, k]

    return np.stack([value(trades, model, market, u, sim["x"][:, k], sim["D"][:, k], state)
                     for k, u in enumerate(t)], axis=1)


def index(grid, u) -> int:
    k = int(np.searchsorted(grid, u - 1e-9))
    assert k < len(grid) and abs(grid[k] - u) < 1e-9, f"date {u} missing from the grid"
    return k
