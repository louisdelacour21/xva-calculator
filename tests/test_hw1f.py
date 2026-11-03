import numpy as np
import pytest

from xva.hw1f import HW1F, calibrer_p, calibrer_q, swap_coupons, taux_swap, vol_normale_hw
from xva.market import CourbePlate, historique_bce, marche_test, vols_swaptions

A, SIGMA = 0.03, 0.009
MKT = marche_test()
GRILLE = np.r_[1 / 12, 0.5, 1.0, 2.0, 3.5, 5.0, 7.0, 10.0]  # irrégulière exprès


@pytest.fixture(scope="module")
def sim():
    return HW1F(A, SIGMA, MKT.estr).simuler(GRILLE, 200_000, graine=1)


def test_martingale_actualisation(sim):
    D = sim["D"]
    et = D.std(axis=0) / np.sqrt(len(D))
    assert np.all(np.abs(D.mean(axis=0) - MKT.estr(GRILLE)) < 3 * et + 1e-12)


def test_moments_x_et_integrale(sim):
    m = HW1F(A, SIGMA, MKT.estr)
    t, x, I = sim["t"], sim["x"], sim["I"]
    # 4 ET relatifs d'une variance gaussienne ; les antithétiques ne réduisent pas le bruit des carrés : n/2 effectifs
    tol = 4 * np.sqrt(2 / (len(x) / 2))
    var_x = SIGMA**2 * (1 - np.exp(-2 * A * t)) / (2 * A)
    np.testing.assert_allclose(x.var(axis=0), var_x, rtol=tol)
    np.testing.assert_allclose(I.var(axis=0), m.V(0, t), rtol=tol)
    # Cov(x_t, ∫_0^t x) = σ²/(2a²)·(1 - e^{-at})²
    cov = np.mean(x * I, axis=0)
    np.testing.assert_allclose(cov, SIGMA**2 / (2 * A**2) * (1 - np.exp(-A * t)) ** 2, rtol=tol)
    assert np.all(np.abs(x.mean(axis=0)) < 1e-12)  # antithétiques : moyenne nulle exacte


def test_zc_futur_reprice_courbe(sim):
    """E[D(0,t)·P(t,T)] = P(0,T) : cohérence de la formule de prix ZC avec la simulation."""
    m = HW1F(A, SIGMA, MKT.estr)
    k, T = 4, 12.0  # t = 3,5 ans
    v = sim["D"][:, k] * m.P(GRILLE[k], T, sim["x"][:, k])
    assert abs(v.mean() - MKT.estr(T)) < 3 * v.std() / np.sqrt(len(v))


def test_zbp_quantlib():
    ql = pytest.importorskip("QuantLib")
    jour = ql.Date(3, 11, 2026)
    ql.Settings.instance().evaluationDate = jour
    r = 0.025
    ts = ql.YieldTermStructureHandle(ql.FlatForward(jour, r, ql.Actual365Fixed(), ql.Continuous, ql.NoFrequency))
    qm = ql.HullWhite(ts, A, SIGMA)
    m = HW1F(A, SIGMA, CourbePlate(r))
    for T, S, X in [(1, 5, 0.9), (2, 10, 0.8), (5, 6, 0.97)]:
        assert abs(m.zbp(T, S, X) - qm.discountBondOption(ql.Option.Put, X, T, S)) < 1e-12
        assert abs(m.zbp(T, S, X, call=True) - qm.discountBondOption(ql.Option.Call, X, T, S)) < 1e-12


def test_jamshidian_quantlib():
    """Swaption payeuse 2Y×5Y mono-courbe contre JamshidianSwaptionEngine (< 0,1 bp du nominal)."""
    ql = pytest.importorskip("QuantLib")
    jour = ql.Date(3, 11, 2026)
    ql.Settings.instance().evaluationDate = jour
    r, K = 0.025, 0.027
    dc = ql.Actual365Fixed()
    ts = ql.YieldTermStructureHandle(ql.FlatForward(jour, r, dc, ql.Continuous, ql.NoFrequency))
    debut, fin = jour + ql.Period(2, ql.Years), jour + ql.Period(7, ql.Years)
    sched = lambda p: ql.Schedule(debut, fin, ql.Period(p), ql.NullCalendar(), ql.Unadjusted, ql.Unadjusted,
                                  ql.DateGeneration.Backward, False)
    swap = ql.VanillaSwap(ql.Swap.Payer, 1.0, sched(ql.Annual), K, dc, sched(ql.Semiannual),
                          ql.Euribor6M(ts), 0.0, ql.Actual360())
    opt = ql.Swaption(swap, ql.EuropeanExercise(debut))
    opt.setPricingEngine(ql.JamshidianSwaptionEngine(ql.HullWhite(ts, A, SIGMA)))

    fixe = list(sched(ql.Annual))
    temps = [dc.yearFraction(jour, d) for d in fixe[1:]]
    coupons = [K * dc.yearFraction(a, b) for a, b in zip(fixe, fixe[1:])]
    coupons[-1] += 1.0
    m = HW1F(A, SIGMA, CourbePlate(r))
    assert abs(m.jamshidian(dc.yearFraction(jour, debut), temps, coupons) - opt.NPV()) < 1e-5


def test_jamshidian_contre_monte_carlo_multi_courbe(sim):
    """Swaption payeuse 2Y×8Y avec base 6M/€STR : Jamshidian = E[D(0,T0)·(V(T0))+] simulé."""
    m = HW1F(A, SIGMA, MKT.estr)
    T0, k = 2.0, 3
    K = taux_swap(T0, 8, MKT.estr, MKT.euribor6m)[0]
    nominal, temps, coupons = swap_coupons(T0, 8, K, MKT.estr, MKT.euribor6m)
    V = nominal - m.P(T0, temps[None, :], sim["x"][:, k, None]) @ coupons
    gain = sim["D"][:, k] * np.maximum(V, 0)
    assert abs(m.jamshidian(T0, temps, coupons, nominal) - gain.mean()) < 3 * gain.std() / np.sqrt(len(gain))


def test_parite_call_put():
    """Payeuse - receveuse = valeur forward du swap payeur."""
    m = HW1F(A, SIGMA, MKT.estr)
    nominal, temps, coupons = swap_coupons(3, 7, 0.02, MKT.estr, MKT.euribor6m)
    fwd = nominal * MKT.estr(3) - MKT.estr(temps) @ coupons
    diff = m.jamshidian(3, temps, coupons, nominal) - m.jamshidian(3, temps, coupons, nominal, payeuse=False)
    assert abs(diff - fwd) < 1e-12


def test_calibration_q_retrouve_sigma():
    vrai = HW1F(A, 0.0105, MKT.estr)
    cot = [(T, 10 - T, vol_normale_hw(vrai, T, 10 - T, MKT.euribor6m)) for T in range(1, 10)]
    m, ecarts = calibrer_q(MKT.estr, MKT.euribor6m, cot, a=A)
    assert abs(m.sigma - 0.0105) < 1e-9 and np.max(np.abs(ecarts)) < 1e-6


def test_calibration_q_sur_matrice_synthetique():
    m, ecarts = calibrer_q(MKT.estr, MKT.euribor6m, vols_swaptions(), a=A)
    assert 0.005 < m.sigma < 0.015 and np.max(np.abs(ecarts)) < 5  # bp


def test_calibration_p():
    tau = np.array([1.0, 5.0, 10.0])
    g = (1 - np.exp(-A * tau)) / (A * tau)
    rng = np.random.default_rng(0)
    # incréments gaussiens dont l'écart-type quotidien vaut exactement σ·g/√252
    z = rng.standard_normal((5000, 3))
    z = (z - z.mean(0)) / z.std(0, ddof=1)
    import pandas as pd
    niveaux = np.vstack([np.zeros(3), np.cumsum(z * 0.008 * g / np.sqrt(252), axis=0)])
    hist = pd.DataFrame(niveaux, columns=tau)
    assert abs(calibrer_p(hist, A) - 0.008) < 1e-12
    assert 0.002 < calibrer_p(historique_bce(), A) < 0.02
