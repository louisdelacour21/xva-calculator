"""Snapshot of the bootstrapped project 09 curves as zero rates, for the app preset (market/curves_09.csv).

    python scripts/snapshot_curves_09.py <path to the Courbes-taux-EUR repository>

Levels come from project 09's reconstructed quotes (ECB data), not from a market feed.
"""
import sys
from datetime import timedelta
from math import log
from pathlib import Path

import pandas as pd

sys.path.insert(0, sys.argv[1])
from ratecurves import quotes as q  # noqa: E402
from ratecurves.curves import CurveHW, estr_curve, euribor6m_curve  # noqa: E402

TENORS = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]

all_quotes = q.read_quotes()
day = sorted(all_quotes.date.unique())[-1]
ois, euribor = q.of_day(all_quotes, day)
estr = estr_curve(day, ois, CurveHW)
six = euribor6m_curve(day, euribor, estr, CurveHW)


def zero(curve, T):
    d = day + timedelta(days=round(T * 365))
    return -log(curve.P(d)) / ((d - day).days / 365) * 100


pd.DataFrame({"date": day, "tenor": TENORS, "estr_pct": [round(zero(estr, T), 6) for T in TENORS],
              "euribor6m_pct": [round(zero(six, T), 6) for T in TENORS]}).to_csv(
    Path(__file__).parents[1] / "market" / "curves_09.csv", index=False)
