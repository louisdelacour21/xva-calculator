import math
from datetime import date

import numpy as np

from xva.market import depuis_courbe_09


class FausseCourbe09:
    """Même interface que courbes.curves.Courbe : attribut jour, méthode P(date), ACT/365F."""

    jour = date(2026, 11, 3)

    def P(self, d):
        return math.exp(-0.02 * (d - self.jour).days / 365)


def test_adaptateur_courbe_09():
    P = depuis_courbe_09(FausseCourbe09())
    t = np.array([0.0, 0.5, 1.0, 10.0])
    np.testing.assert_allclose(P(t), np.exp(-0.02 * np.round(t * 365) / 365), rtol=1e-15)
    assert P(1.0).shape == ()
