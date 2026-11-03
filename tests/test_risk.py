import numpy as np
import pytest

from xva.exposure import exposition, grille
from xva.hw1f import HW1F
from xva.market import marche_test, vols_swaptions
from xva.products import cube_mtm, swap
from xva.risk import PILIERS, ba_cva_reduit, cs01, facteur_duree, sa_ccr, scva, sensibilites, triangle
from xva.xva import charger_credit

MKT = marche_test()
CREDIT = charger_credit(P=MKT.estr)
PAYEUR = swap("P", "irs", 1, 10e6, 0.025, 0.0, 10)


def test_triangles_somment_a_un():
    u = np.linspace(0, 40, 401)
    assert np.allclose(sum(triangle(k)(u) for k in range(len(PILIERS))), 1.0)


@pytest.fixture(scope="module")
def sens():
    return sensibilites([PAYEUR], MKT, vols_swaptions(), CREDIT, n=2000)


def test_deltas_piliers_somment_au_parallele(sens):
    for q in ("CVA", "DVA"):
        total = sum(sens[f"delta {p:g}Y"][q][0] for p in PILIERS)
        assert abs(total / sens["delta parallèle"][q][0] - 1) < 0.05


def test_signes_delta_vega(sens):
    """Payeur : taux en hausse → plus d'exposition positive (CVA ↑) et moins de négative (DVA ↓) ; vol ↑ → les deux ↑."""
    d, v = sens["delta parallèle"], sens["vega +1 %"]
    assert d["CVA"][0] > 5 * d["CVA"][1] and d["DVA"][0] < 0
    assert v["CVA"][0] > 5 * v["CVA"][1] and v["DVA"][0] > 0


def test_cs01():
    hw = HW1F(0.03, 0.009, MKT.estr)
    base, t = grille([PAYEUR], mpors=())
    sim = hw.simuler(t, 2000)
    expo = exposition(cube_mtm([PAYEUR], hw, MKT, sim).sum(-1), sim["D"], t, base, 0.0)
    r = cs01(expo, CREDIT, MKT.estr)
    total = sum(r[f"CS01 {T:g}Y"]["CVA"][0] for T in CREDIT.contrepartie.spreads)
    assert r["CS01 parallèle"]["CVA"][0] > 0 and abs(total / r["CS01 parallèle"]["CVA"][0] - 1) < 0.05
    assert r["CS01 parallèle"]["DVA"][0] < 0  # la DVA ne compte que si la contrepartie survit


# --- capital, calculé à la main ---------------------------------------------------------------------------------

def test_sa_ccr_un_trade_non_marge():
    # payeur 10M, 0-5 ans : SD = (1 - e^-0,25)/0,05 = 4,42398 ; MF = 1 ; add-on = 0,5 %·44,2398M = 221 199
    r = sa_ccr([swap("A", "irs", 1, 10e6, 0.025, 0.0, 5)], V0=0.0)
    assert facteur_duree(0, 5) == pytest.approx(4.423984, abs=1e-6)
    assert r["add-on"] == pytest.approx(221_199.2, abs=0.1)
    assert r["multiplicateur"] == 1.0 and r["EAD"] == pytest.approx(1.4 * 221_199.2, abs=0.2)  # 309 679


def test_sa_ccr_deux_buckets_et_multiplicateur():
    # payeur 10M 5 ans (bucket 1-5) + receveur 10M 10 ans (bucket > 5) : D2 = 44,2398M, D3 = -78,6939M
    tr = [swap("A", "irs", 1, 10e6, 0.025, 0.0, 5), swap("B", "irs", -1, 10e6, 0.025, 0.0, 10)]
    en = np.sqrt(44.23984**2 + 78.69387**2 - 1.4 * 44.23984 * 78.69387) * 1e6  # 57,24M
    V0 = -100_000.0
    a = 0.005 * en
    mult = 0.05 + 0.95 * np.exp(V0 / (2 * 0.95 * a))
    r = sa_ccr(tr, V0)
    assert r["add-on"] == pytest.approx(a, rel=1e-6) and r["RC"] == 0.0
    assert r["EAD"] == pytest.approx(1.4 * mult * a, rel=1e-6)


def test_sa_ccr_marge():
    # CSA sans seuil, MPoR 10 j : MF = 1,5·√(10/250) = 0,3 ; RC = max(V - C, TH + MTA, 0) = 0
    csa = {"H_C": 0.0, "H_B": 0.0, "MTA": 0.0, "mpor_jours": 10}
    r = sa_ccr([swap("A", "irs", 1, 10e6, 0.025, 0.0, 5)], V0=0.0, csa=csa)
    assert r["add-on"] == pytest.approx(0.3 * 221_199.2, abs=0.1)
    assert r["EAD"] == pytest.approx(1.4 * 0.3 * 221_199.2, abs=0.1) and r["EAD"] < r["EAD non margée"]
    # seuil énorme : l'EAD margée serait plus grande, plafonnée à la non margée
    assert sa_ccr([swap("A", "irs", 1, 10e6, 0.025, 0.0, 5)], 0.0, csa | {"H_C": 1e9})["EAD"] == r["EAD non margée"]


def test_ba_cva_reduit():
    # RW 3 %, M = 5, EAD = 309 679 : DF = (1 - e^-0,25)/0,25 = 0,884797 ; SCVA = 0,03·5·309 679·0,884797/1,4 = 29 357
    s = scva(0.03, [(5.0, 309_679.0)])
    assert s == pytest.approx(0.03 * 5 * 309_679.0 * 0.884797 / 1.4, rel=1e-6)
    assert ba_cva_reduit([s]) == pytest.approx(0.65 * s, rel=1e-12)  # une contrepartie : √(0,25 + 0,75) = 1
    assert ba_cva_reduit([s, s]) == pytest.approx(0.65 * s * np.sqrt(2.5), rel=1e-12)  # √(0,25·4 + 0,75·2)
