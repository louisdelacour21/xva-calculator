import math
from datetime import date

import numpy as np

from xva.market import from_curve_09


class FakeCurve09:
    """Same interface as ratecurves.curves.Curve: day attribute, P(date) method, ACT/365F."""

    day = date(2026, 11, 3)

    def P(self, d):
        return math.exp(-0.02 * (d - self.day).days / 365)


def test_curve_09_adapter():
    P = from_curve_09(FakeCurve09())
    t = np.array([0.0, 0.5, 1.0, 10.0])
    np.testing.assert_allclose(P(t), np.exp(-0.02 * np.round(t * 365) / 365), rtol=1e-15)
    assert P(1.0).shape == ()
