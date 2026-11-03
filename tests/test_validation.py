import numpy as np
from scipy import stats

from xva.hw1f import HW1F
from xva.market import marche_test
from xva.products import swap
from xva.validation import convergence, pit, puissance_ks
from xva.xva import charger_credit

MKT = marche_test()


def test_pit_uniforme_si_le_modele_est_le_vrai():
    """Courbes réalisées tirées du modèle lui-même : les PIT doivent être uniformes."""
    h, m = 0.25, HW1F(0.03, 0.008, MKT.estr)
    x = m.simuler([h], 200, graine=7, antithetiques=False)["x"][:, 0]
    pits = [pit(MKT.estr, lambda T, xi=xi: m.P(h, h + np.asarray(T, dtype=float), xi), h, 0.008, n=1000, graine=i)
            for i, xi in enumerate(x)]
    for p in pits[0]:
        assert stats.kstest([q[p] for q in pits], "uniform").pvalue > 0.01


def test_puissance_ks():
    assert abs(puissance_ks(50, facteur=1.0) - 0.05) < 0.02  # sans écart, le KS rejette au niveau nominal
    assert puissance_ks(20) < puissance_ks(250)


def test_convergence_en_racine_de_n():
    tr = [swap("P", "irs", 1, 10e6, 0.025, 0.0, 10)]
    df, pente = convergence(tr, HW1F(0.03, 0.008, MKT.estr), MKT, charger_credit(P=MKT.estr), (1_000, 4_000, 16_000))
    assert abs(pente + 0.5) < 0.1 and (df["gain antithétique"] > 1).all()
