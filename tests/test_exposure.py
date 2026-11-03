from pathlib import Path

import numpy as np
import pytest

from xva.exposure import MPOR_10J, exposition, grille
from xva.hw1f import HW1F
from xva.market import marche_test
from xva.products import cube_mtm, charger_portefeuille, swap, valeur

MKT = marche_test()
HW = HW1F(0.03, 0.009, MKT.estr)
TRADES = [swap("A", "irs", 1, 1.0, 0.024, 0.0, 6), swap("B", "irs", -1, 0.6, 0.03, 1.0, 4),
          swap("C", "ois", 1, 0.8, 0.026, 0.0, 3)]
MPOR_20J = 28 / 365


@pytest.fixture(scope="module")
def etat():
    base, t = grille(TRADES, mpors=(MPOR_10J, MPOR_20J))
    sim = HW.simuler(t, 20_000, graine=5)
    c = cube_mtm(TRADES, HW, MKT, sim)
    V0 = valeur(TRADES, HW, MKT, 0.0, 0.0, 1.0)[0]
    return base, t, sim, c, V0


def expo(etat, csa=None, trades=None):
    base, t, sim, c, V0 = etat
    j = list(range(c.shape[2])) if trades is None else trades
    return exposition(c[:, :, j].sum(-1), sim["D"], t, base, V0[j].sum(), csa)


def test_netting_sous_additif(etat):
    ns = expo(etat)
    somme = sum(expo(etat, trades=[j])["EE"] for j in range(len(TRADES)))
    assert np.all(ns["EE"] <= somme + 1e-15)


def test_swaps_miroirs_exposition_nulle():
    tr = [swap("P", "irs", 1, 1.0, 0.025, 0.0, 5), swap("R", "irs", -1, 1.0, 0.025, 0.0, 5)]
    base, t = grille(tr, mpors=())
    sim = HW.simuler(t, 2_000, graine=1)
    m = exposition(cube_mtm(tr, HW, MKT, sim).sum(-1), sim["D"], t, base, 0.0)
    assert np.max(m["EE"]) < 1e-15 and np.max(m["NEE"]) < 1e-15


def test_seuil_infini_egal_sans_collateral(etat):
    sans = expo(etat)
    avec = expo(etat, {"H_C": 1e18, "H_B": 1e18, "MTA": 0.0, "mpor": MPOR_10J})
    np.testing.assert_array_equal(sans["EE"], avec["EE"])
    np.testing.assert_array_equal(sans["NEE"], avec["NEE"])


def test_collateral_parfait_mpor_nulle(etat):
    m = expo(etat, {"H_C": 0.0, "H_B": 0.0, "MTA": 0.0, "mpor": 0.0})
    assert np.max(m["EE"]) == 0 and np.max(m["NEE"]) == 0


def test_mpor_monotonie_et_racine(etat):
    """EE croissante en MPoR ; loi ≈ √MPoR testée hors dates de flux (tolérance 15 %)."""
    csa = lambda m: {"H_C": 0.0, "H_B": 0.0, "MTA": 0.0, "mpor": m}
    e10, e20 = expo(etat, csa(MPOR_10J)), expo(etat, csa(MPOR_20J))
    base = etat[0]
    flux = np.unique(np.concatenate([tr.flux() for tr in TRADES] + [[tr.debut for tr in TRADES]]))
    loin = np.array([u > MPOR_20J and not np.any((flux > u - MPOR_20J - 1e-9) & (flux <= u + 1e-9)) for u in base])
    loin &= e10["EE"] > 0
    assert loin.sum() > 20
    assert np.mean(e20["EE"][loin] >= e10["EE"][loin]) > 0.95
    ratio = np.mean(e20["EE"][loin] / e10["EE"][loin])
    assert abs(ratio / np.sqrt(2) - 1) < 0.15


def test_eepe_et_pfe_coherents(etat):
    m = expo(etat)
    assert np.all(m["PFE"] >= m["EE"]) and m["EEPE"] >= m["EE"][m["t"] <= 1].mean()


def test_portefeuille_demo():
    trades, csa = charger_portefeuille(Path(__file__).parents[1] / "portfolios" / "demo.yaml")
    mp = csa["mpor_jours"] * 14 / 10 / 365
    base, t = grille(trades, mpors=(mp,))
    sim = HW.simuler(t, 4_000, graine=0)
    c = cube_mtm(trades, HW, MKT, sim)
    V0 = valeur(trades, HW, MKT, 0.0, 0.0, 1.0)[0].sum()
    sans = exposition(c.sum(-1), sim["D"], t, base, V0)
    avec = exposition(c.sum(-1), sim["D"], t, base, V0, {"H_C": csa["H_C"], "H_B": csa["H_B"],
                                                           "MTA": csa["MTA"], "mpor": mp})
    assert c.shape == (4_000, len(t), len(trades))
    assert np.all(avec["EE"] <= sans["EE"] + 1e-6) and avec["EEPE"] < sans["EEPE"]
