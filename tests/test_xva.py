import numpy as np
import pytest

from xva.exposure import exposition, grille
from xva.hw1f import HW1F
from xva.market import marche_test
from xva.products import _beta, cube_mtm, indice, swap, valeur
from xva.xva import (Credit, Survie, ajustements, allocation_euler, bootstrap, charger_credit, contributions,
                     echeancier_cds, poids, pre_trade, sans_risque, spread_cds)

A, SIGMA = 0.03, 0.009
MKT = marche_test()
HW = HW1F(A, SIGMA, MKT.estr)
SPREADS = {1.0: 0.0090, 3.0: 0.0110, 5.0: 0.0130, 7.0: 0.0145, 10.0: 0.0160}
CREDIT = charger_credit(P=MKT.estr)


def test_bootstrap_reprix_spreads():
    s = bootstrap(SPREADS, MKT.estr)
    for T, x in SPREADS.items():
        assert abs(spread_cds(s, MKT.estr, echeancier_cds(T)) - x) < 1e-12
    u = np.linspace(0, 12, 200)
    assert s(0.0) == 1.0 and np.all(np.diff(s(u)) < 0) and np.all(s.lam > 0)


def test_cds_quantlib():
    """Même courbe de survie dans QuantLib (HazardRateCurve) : spread au pair MidPointCdsEngine < 0,1 bp."""
    ql = pytest.importorskip("QuantLib")
    jour = ql.Date(3, 11, 2026)
    ql.Settings.instance().evaluationDate = jour
    dc = ql.Actual365Fixed()
    disc = ql.YieldTermStructureHandle(ql.FlatForward(jour, 0.025, dc, ql.Continuous, ql.NoFrequency))
    jours = lambda T: int(round(T * 365))
    s = bootstrap(SPREADS, MKT.estr)
    piliers = [jour] + [jour + jours(T) for T in s.piliers]
    hc = ql.DefaultProbabilityTermStructureHandle(ql.HazardRateCurve(piliers, [s.lam[0], *s.lam], dc))
    for T in (1, 2, 5, 10):
        dates = [jour + jours(u) for u in np.r_[0, echeancier_cds(T)]]
        sched = ql.Schedule(dates, ql.NullCalendar(), ql.Unadjusted)
        cds = ql.CreditDefaultSwap(ql.Protection.Buyer, 1.0, 0.01, sched, ql.Unadjusted, dc, True, True, jour)
        cds.setPricingEngine(ql.MidPointCdsEngine(hc, s.R, disc))
        temps = np.array([dc.yearFraction(jour, d) for d in dates[1:]])
        assert abs(spread_cds(s, MKT.estr, temps) - cds.fairSpread()) < 1e-5, T  # 0,1 bp


def _residuel(tr, t):
    """Swap résiduel en t (date de fixing) écrit nominal - Σ c_j P(t, T_j) pour Jamshidian."""
    c, nominal = {}, 0.0
    for T, d in zip(tr.fixe, np.diff(np.r_[tr.debut, tr.fixe])):
        if T > t:
            c[T] = c.get(T, 0) + tr.K * d
    for s, e, b in zip(*tr.periodes_flot(), _beta(tr, MKT)):
        if s >= t:
            if abs(s - t) < 1e-12:
                nominal += b
            else:
                c[s] = c.get(s, 0) - b
            c[e] = c.get(e, 0) + 1
    return nominal, np.array(list(c)), np.array(list(c.values()))


def test_cva_swap_seul_egal_swaptions():
    """Sans collatéral, CVA d'un payeur = (1-R) Σ ΔPD·S_B·Swaption(t̄) (fermée), à moins de 2 %."""
    tr = swap("P", "irs", 1, 1.0, 0.026, 0.0, 5)
    base, t = grille([tr], mpors=(), pas=0.5)
    sim = HW.simuler(t, 40_000, graine=7)
    V0 = valeur([tr], HW, MKT, 0.0, 0.0, 1.0)[0, 0]
    expo = exposition(cube_mtm([tr], HW, MKT, sim)[:, :, 0], sim["D"], t, base, V0)
    cva, et = ajustements(expo, CREDIT, V0)["CVA"]
    swn = np.array([HW.jamshidian(u, *_residuel(tr, u)[1:], _residuel(tr, u)[0]) for u in base[:-1]] + [0.0])
    ferme = np.sum((np.r_[max(V0, 0), swn[:-1]] + swn) / 2 * poids(base, CREDIT)["CVA"])
    assert abs(cva / ferme - 1) < 0.02 and abs(cva - ferme) < 3 * et


@pytest.fixture(scope="module")
def ns():
    trades = [swap("A", "irs", 1, 1.0, 0.024, 0.0, 6), swap("B", "irs", -1, 0.6, 0.03, 1.0, 4),
              swap("C", "ois", 1, 0.8, 0.026, 0.0, 3)]
    base, t = grille(trades)
    sim = HW.simuler(t, 10_000, graine=11)
    c = cube_mtm(trades, HW, MKT, sim)
    V0 = valeur(trades, HW, MKT, 0.0, 0.0, 1.0)[0]
    return trades, base, t, sim, c, V0


def test_allocation_euler_somme(ns):
    trades, base, t, sim, c, V0 = ns
    cva = ajustements(exposition(c.sum(-1), sim["D"], t, base, V0.sum()), CREDIT, V0.sum())["CVA"][0]
    alloc = allocation_euler(c, sim["D"], t, base, V0, CREDIT)
    assert abs(alloc.sum() - cva) < 1e-12 * abs(cva) + 1e-15


def test_pre_trade_egal_recalcul_complet(ns):
    """ΔXVA en ne calculant que le nouveau trade = différence de deux calculs complets sur les mêmes chemins."""
    trades, base, t, sim, c, V0 = ns
    csa = {"H_C": 0.05, "H_B": 0.05, "MTA": 0.005, "mpor": 14 / 365}
    nouveau = swap("N", "irs", -1, 0.5, 0.025, 0.5, 4)
    r = pre_trade(c.sum(-1), V0.sum(), nouveau, HW, MKT, sim, base, CREDIT, csa)
    tous = trades + [nouveau]
    c2 = cube_mtm(tous, HW, MKT, sim)
    V02 = valeur(tous, HW, MKT, 0.0, 0.0, 1.0)[0].sum()
    apres = ajustements(exposition(c2.sum(-1), sim["D"], t, base, V02, csa), CREDIT, V02)
    avant = ajustements(exposition(c.sum(-1), sim["D"], t, base, V0.sum(), csa), CREDIT, V0.sum())
    for k in ("CVA", "DVA", "FCA", "FBA"):
        assert abs(r["Δ"][k][0] - (apres[k][0] - avant[k][0])) < 1e-12
        assert r["Δ"][k][1] > 0


def test_symetrie_cva_dva(ns):
    """La CVA vue de la banque est la DVA vue de la contrepartie (V → -V, rôles échangés)."""
    trades, base, t, sim, c, V0 = ns
    V = c.sum(-1)
    b = contributions(exposition(V, sim["D"], t, base, V0.sum()), CREDIT)
    inverse = Credit(CREDIT.banque, CREDIT.contrepartie, CREDIT.sF)
    cpt = contributions(exposition(-V, sim["D"], t, base, -V0.sum()), inverse)
    np.testing.assert_allclose(b["CVA"], cpt["DVA"], rtol=1e-13, atol=1e-18)
    np.testing.assert_allclose(b["DVA"], cpt["CVA"], rtol=1e-13, atol=1e-18)


def test_cas_limites(ns):
    trades, base, t, sim, c, V0 = ns
    expo = exposition(c.sum(-1), sim["D"], t, base, V0.sum())
    sur = Credit(sans_risque(), sans_risque(), 0.0)
    r = ajustements(expo, sur, V0.sum())
    assert all(r[k][0] == 0 for k in ("CVA", "DVA", "FCA", "FBA"))
    # spread de financement égal au coût du crédit propre : ℓ = 0, pas de FVA (pas de double comptage DVA)
    plat = Survie(np.array([10.0]), np.array([0.01]))
    r = ajustements(expo, Credit(plat, plat, 0.6 * 0.01), V0.sum())
    assert abs(r["FCA"][0]) < 1e-15 and abs(r["FBA"][0]) < 1e-15 and r["CVA"][0] > 0  # ℓ nul à l'arrondi près
