"""Market data: curves, swaption vols, ECB history.

Minimal curve interface: a callable t -> P(0, t), t in ACT/365F years (scalar or numpy array).
The engine builds no curve: in production they come from project 09 through from_curve_09(),
in tests from a FlatCurve.
"""
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2] / "market"
Curve = Callable[[np.ndarray], np.ndarray]


class FlatCurve:
    """Constant continuous zero rate: fallback curve for unit tests."""

    def __init__(self, rate: float):
        self.rate = rate

    def __call__(self, t):
        return np.exp(-self.rate * np.asarray(t, dtype=float))


class ZeroCurve:
    """Continuous zero rates per pillar, linear between pillars, flat beyond: the curve typed or uploaded by the user."""

    def __init__(self, tenors, rates):
        self.tenors, self.rates = np.asarray(tenors, dtype=float), np.asarray(rates, dtype=float)

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        return np.exp(-np.interp(t, self.tenors, self.rates) * t)


def from_curve_09(curve, day=None) -> Curve:
    """Plug-in point: wraps a Curve / CurveHW from the ratecurves package (method P(date), ACT/365F)."""
    day = day or curve.day
    # shortcut: t rounded to the day (error < 0.5 day), enough while the simulation grid is in days
    one = np.vectorize(lambda u: curve.P(day + timedelta(days=round(u * 365))), otypes=[float])
    return lambda t: one(np.asarray(t, dtype=float))


@dataclass
class Market:
    estr: Curve        # discounting (EUR cash CSA)
    euribor6m: Curve   # projection of the 6M fixings


def sample_market(rate: float = 0.025, basis: float = 0.002) -> Market:
    """Two flat curves one basis apart: exercises the multi-curve set-up without project 09."""
    return Market(FlatCurve(rate), FlatCurve(rate + basis))


def swaption_vols(path: Path = ROOT / "swaption_vols.yaml") -> list[tuple[float, float, float]]:
    """(expiry, tenor, normal vol in decimal) of the co-terminal ATM swaptions."""
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [(float(q["expiry"]), float(q["tenor"]), q["vol_bp"] * 1e-4) for q in d["swaptions"]]


def ecb_history(maturities=(1, 5, 10)) -> pd.DataFrame:
    """ECB spot rates (AAA Svensson curve) in decimal, one column per maturity in years. P measure only."""
    cols = {}
    for m in maturities:
        f = ROOT / "ecb" / f"YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{m}Y.csv"
        s = pd.read_csv(f, index_col="TIME_PERIOD")["OBS_VALUE"]
        cols[m] = s / 100
    return pd.DataFrame(cols).dropna()


def ecb_latest() -> tuple[str, pd.DataFrame]:
    """(date, table tenor in years / AAA zero rate in %, continuously compounded) at the latest ECB date."""
    def tenor(f):
        return int(f.stem[3:-1]) * (1 / 12 if f.stem.endswith("M") else 1)
    s = {tenor(f): pd.read_csv(f, index_col="TIME_PERIOD")["OBS_VALUE"] for f in ROOT.glob("ecb_history/SR_*.csv")}
    d = pd.DataFrame(s).dropna().sort_index(axis=1)
    return d.index[-1], d.iloc[-1].rename_axis("tenor").reset_index(name="rate")
