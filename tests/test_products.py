import numpy as np
import pytest
from scipy.stats import norm

from xva.exposure import grille
from xva.hw1f import HW1F, swap_coupons
from xva.market import CourbePlate, Marche, marche_test
from xva.products import Trade, _beta, cube_mtm, indice, swap, valeur

A, SIGMA = 0.03, 0.009
MKT = marche_test()
HW = HW1F(A, SIGMA, MKT.estr)
IRS = swap("IRS", "irs", 1, 1.0, 0.026, 0.0, 5)
OIS = swap("OIS", "ois", -1, 1.0, 0.024, 0.5, 4)
FWD = swap("FWD", "irs", -1, 1.0, 0.028, 1.0, 3)
TRADES = [IRS, OIS, FWD]


@pytest.fixture(scope="module")
def cube():
    _, t = grille(TRADES, mpors=())
    sim = HW.simuler(t, 40_000, graine=3)
    return sim, cube_mtm(TRADES, HW, MKT, sim)


def v0(trades, marche=MKT, modele=HW):
    return valeur(trades, modele, marche, 0.0, 0.0, 1.0)[0]


def pv_flux_apres(tr: Trade, t: float) -> float:
    """Valeur en 0 des seuls flux payés après t (fermée, flux flottants à leur valeur forward)."""
    w = tr.sens * tr.nominal
    dk = np.diff(np.r_[tr.debut, tr.fixe])
    fixe = -w * tr.K * np.sum((dk * MKT.estr(tr.fixe))[tr.fixe > t])
    s, e = tr.periodes_flot()
    flot = w * np.sum(((_beta(tr, MKT) * MKT.estr(s) - MKT.estr(e)))[e > t])
    return fixe + flot


def _ql_flat(ql, jour, r):
    return ql.YieldTermStructureHandle(ql.FlatForward(jour, r, ql.Actual365Fixed(), ql.Continuous, ql.NoFrequency))


def test_irs_multicourbe_quantlib():
    """IRS payeur 1Y×5Y, projection 6M ≠ actualisation €STR, contre DiscountingSwapEngine (< 1e-10 du nominal)."""
    ql = pytest.importorskip("QuantLib")
    jour = ql.Date(3, 11, 2026)
    ql.Settings.instance().evaluationDate = jour
    dc = ql.Actual365Fixed()
    disc, proj = _ql_flat(ql, jour, 0.025), _ql_flat(ql, jour, 0.027)
    debut, fin = jour + ql.Period(1, ql.Years), jour + ql.Period(6, ql.Years)
    sched = lambda p: ql.Schedule(debut, fin, ql.Period(p), ql.NullCalendar(), ql.Unadjusted, ql.Unadjusted,
                                  ql.DateGeneration.Forward, False)
    idx = ql.IborIndex("E6M", ql.Period(6, ql.Months), 0, ql.EURCurrency(), ql.NullCalendar(), ql.Unadjusted,
                       False, dc, proj)
    K = 0.026
    qs = ql.VanillaSwap(ql.Swap.Payer, 1.0, sched(ql.Annual), K, dc, sched(ql.Semiannual), idx, 0.0, dc)
    qs.setPricingEngine(ql.DiscountingSwapEngine(disc))
    yf = lambda s: np.array([dc.yearFraction(jour, d) for d in list(s)[1:]])
    tr = Trade("ql", "irs", 1, 1.0, K, dc.yearFraction(jour, debut), yf(sched(ql.Annual)), yf(sched(ql.Semiannual)))
    marche = Marche(CourbePlate(0.025), CourbePlate(0.027))
    assert abs(v0([tr], marche, HW1F(A, SIGMA, marche.estr))[0] - qs.NPV()) < 1e-10


def test_ois_quantlib():
    ql = pytest.importorskip("QuantLib")
    jour = ql.Date(3, 11, 2026)
    ql.Settings.instance().evaluationDate = jour
    dc = ql.Actual365Fixed()
    disc = _ql_flat(ql, jour, 0.025)
    sched = ql.Schedule(jour, jour + ql.Period(4, ql.Years), ql.Period(ql.Annual), ql.NullCalendar(),
                        ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False)
    K = 0.024
    qs = ql.OvernightIndexedSwap(ql.Swap.Receiver, 1.0, sched, K, dc, ql.Estr(disc))
    qs.setPricingEngine(ql.DiscountingSwapEngine(disc))
    tr = Trade("ql", "ois", -1, 1.0, K, 0.0, *[np.array([dc.yearFraction(jour, d) for d in list(sched)[1:]])] * 2)
    marche = Marche(CourbePlate(0.025), CourbePlate(0.025))
    # QuantLib capitalise jour ouvré TARGET par jour ouvré, bornes de période non ouvrées comprises : ~1e-8 observé
    assert abs(v0([tr], marche, HW1F(A, SIGMA, marche.estr))[0] - qs.NPV()) < 1e-7  # 0,001 bp, critère fiche 0,1 bp


def test_martingale_mtm(cube):
    """E[D(0,t)·V(t)] = valeur en 0 des flux restants, y compris en milieu de période (coupon fixé sur le chemin)."""
    sim, c = cube
    for u in (3 / 12, 1.0, 21 / 12, 31 / 12, 47 / 12):  # dates de la grille mensuelle
        k = indice(sim["t"], u)
        for j, tr in enumerate(TRADES):
            dv = sim["D"][:, k] * c[:, k, j]
            assert abs(dv.mean() - pv_flux_apres(tr, u)) < 3 * dv.std() / np.sqrt(len(dv)) + 1e-12, (u, tr.nom)


def test_ee_swap_seul_egal_swaption(cube):
    """Sans collatéral, EE*(t) d'un payeur = swaption payeuse d'échéance t sur le swap résiduel."""
    sim, c = cube
    for u in (1.0, 2.0, 3.0, 4.0):
        k = indice(sim["t"], u)
        ee = sim["D"][:, k] * np.maximum(c[:, k, 0], 0)
        nominal, temps, coupons = swap_coupons(u, 5 - u, IRS.K, MKT.estr, MKT.euribor6m)
        assert abs(ee.mean() - HW.jamshidian(u, temps, coupons, nominal)) < 3 * ee.std() / np.sqrt(len(ee))


def test_pfe_quantile_ferme(cube):
    """Swap seul à une date de flux : V monotone en x(t) gaussien, donc PFE = V(t, x_q), dans l'IC binomial."""
    sim, c = cube
    n = len(c)
    for u in (1.0, 3.0):
        k = indice(sim["t"], u)
        xq = norm.ppf(0.975) * SIGMA * np.sqrt((1 - np.exp(-2 * A * u)) / (2 * A))
        pfe = valeur([IRS], HW, MKT, u, xq, 1.0)[0, 0]
        part = np.mean(np.maximum(c[:, k, 0], 0) <= pfe)
        assert abs(part - 0.975) < 3 * np.sqrt(0.975 * 0.025 / n)
