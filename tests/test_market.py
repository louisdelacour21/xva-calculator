import math
from datetime import date

import numpy as np
import pytest

from xva.market import FlatCurve, ZeroCurve, from_curve_09


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


def test_zero_curve_interpolates_rates_and_stays_flat_beyond():
    z = ZeroCurve([1, 10], [0.02, 0.03])
    assert np.allclose(ZeroCurve([1, 10], [0.02, 0.02])(np.array([0.5, 5.0, 40.0])), FlatCurve(0.02)(np.array([0.5, 5.0, 40.0])))
    assert z(5.5) == pytest.approx(math.exp(-0.025 * 5.5))
    assert z(0.5) == pytest.approx(math.exp(-0.02 * 0.5))  # flat before the first pillar
    assert z(30.0) == pytest.approx(math.exp(-0.03 * 30.0))
    assert z(0.0) == 1.0
