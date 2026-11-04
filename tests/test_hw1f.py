import numpy as np
import pytest

from xva.hw1f import HW1F, calibrate_p, calibrate_q, hw_normal_vol, swap_coupons, swap_rate
from xva.market import FlatCurve, ecb_history, sample_market, swaption_vols

A, SIGMA = 0.03, 0.009
MKT = sample_market()
GRID = np.r_[1 / 12, 0.5, 1.0, 2.0, 3.5, 5.0, 7.0, 10.0]  # irregular on purpose


@pytest.fixture(scope="module")
def sim():
    return HW1F(A, SIGMA, MKT.estr).simulate(GRID, 200_000, seed=1)


def test_discount_martingale(sim):
    D = sim["D"]
    se = D.std(axis=0) / np.sqrt(len(D))
    assert np.all(np.abs(D.mean(axis=0) - MKT.estr(GRID)) < 3 * se + 1e-12)


def test_moments_of_x_and_integral(sim):
    m = HW1F(A, SIGMA, MKT.estr)
    t, x, I = sim["t"], sim["x"], sim["I"]
    # 4 relative SE of a Gaussian variance; antithetics do not reduce the noise of squares: n/2 effective
    tol = 4 * np.sqrt(2 / (len(x) / 2))
    var_x = SIGMA**2 * (1 - np.exp(-2 * A * t)) / (2 * A)
    np.testing.assert_allclose(x.var(axis=0), var_x, rtol=tol)
    np.testing.assert_allclose(I.var(axis=0), m.V(0, t), rtol=tol)
    # Cov(x_t, ∫_0^t x) = σ²/(2a²)·(1 - e^{-at})²
    cov = np.mean(x * I, axis=0)
    np.testing.assert_allclose(cov, SIGMA**2 / (2 * A**2) * (1 - np.exp(-A * t)) ** 2, rtol=tol)
    assert np.all(np.abs(x.mean(axis=0)) < 1e-12)  # antithetics: exactly zero mean


def test_future_zc_reprices_curve(sim):
    """E[D(0,t)·P(t,T)] = P(0,T): ZC price formula consistent with the simulation."""
    m = HW1F(A, SIGMA, MKT.estr)
    k, T = 4, 12.0  # t = 3.5 years
    v = sim["D"][:, k] * m.P(GRID[k], T, sim["x"][:, k])
    assert abs(v.mean() - MKT.estr(T)) < 3 * v.std() / np.sqrt(len(v))


def test_zbp_quantlib():
    ql = pytest.importorskip("QuantLib")
    day = ql.Date(3, 11, 2026)
    ql.Settings.instance().evaluationDate = day
    r = 0.025
    ts = ql.YieldTermStructureHandle(ql.FlatForward(day, r, ql.Actual365Fixed(), ql.Continuous, ql.NoFrequency))
    qm = ql.HullWhite(ts, A, SIGMA)
    m = HW1F(A, SIGMA, FlatCurve(r))
    for T, S, X in [(1, 5, 0.9), (2, 10, 0.8), (5, 6, 0.97)]:
        assert abs(m.zbp(T, S, X) - qm.discountBondOption(ql.Option.Put, X, T, S)) < 1e-12
        assert abs(m.zbp(T, S, X, call=True) - qm.discountBondOption(ql.Option.Call, X, T, S)) < 1e-12


def test_jamshidian_quantlib():
    """Single-curve 2Y×5Y payer swaption against JamshidianSwaptionEngine (< 0.1 bp of notional)."""
    ql = pytest.importorskip("QuantLib")
    day = ql.Date(3, 11, 2026)
    ql.Settings.instance().evaluationDate = day
    r, K = 0.025, 0.027
    dc = ql.Actual365Fixed()
    ts = ql.YieldTermStructureHandle(ql.FlatForward(day, r, dc, ql.Continuous, ql.NoFrequency))
    start, end = day + ql.Period(2, ql.Years), day + ql.Period(7, ql.Years)
    sched = lambda p: ql.Schedule(start, end, ql.Period(p), ql.NullCalendar(), ql.Unadjusted, ql.Unadjusted,
                                  ql.DateGeneration.Backward, False)
    swap = ql.VanillaSwap(ql.Swap.Payer, 1.0, sched(ql.Annual), K, dc, sched(ql.Semiannual),
                          ql.Euribor6M(ts), 0.0, ql.Actual360())
    opt = ql.Swaption(swap, ql.EuropeanExercise(start))
    opt.setPricingEngine(ql.JamshidianSwaptionEngine(ql.HullWhite(ts, A, SIGMA)))

    fixed = list(sched(ql.Annual))
    times = [dc.yearFraction(day, d) for d in fixed[1:]]
    coupons = [K * dc.yearFraction(a, b) for a, b in zip(fixed, fixed[1:])]
    coupons[-1] += 1.0
    m = HW1F(A, SIGMA, FlatCurve(r))
    assert abs(m.jamshidian(dc.yearFraction(day, start), times, coupons) - opt.NPV()) < 1e-5


def test_jamshidian_vs_monte_carlo_multi_curve(sim):
    """2Y×8Y payer swaption with 6M/€STR basis: Jamshidian = simulated E[D(0,T0)·(V(T0))+]."""
    m = HW1F(A, SIGMA, MKT.estr)
    T0, k = 2.0, 3
    K = swap_rate(T0, 8, MKT.estr, MKT.euribor6m)[0]
    nominal, times, coupons = swap_coupons(T0, 8, K, MKT.estr, MKT.euribor6m)
    V = nominal - m.P(T0, times[None, :], sim["x"][:, k, None]) @ coupons
    payoff = sim["D"][:, k] * np.maximum(V, 0)
    assert abs(m.jamshidian(T0, times, coupons, nominal) - payoff.mean()) < 3 * payoff.std() / np.sqrt(len(payoff))


def test_put_call_parity():
    """Payer - receiver = forward value of the payer swap."""
    m = HW1F(A, SIGMA, MKT.estr)
    nominal, times, coupons = swap_coupons(3, 7, 0.02, MKT.estr, MKT.euribor6m)
    fwd = nominal * MKT.estr(3) - MKT.estr(times) @ coupons
    diff = m.jamshidian(3, times, coupons, nominal) - m.jamshidian(3, times, coupons, nominal, payer=False)
    assert abs(diff - fwd) < 1e-12


def test_q_calibration_recovers_sigma():
    true = HW1F(A, 0.0105, MKT.estr)
    quotes = [(T, 10 - T, hw_normal_vol(true, T, 10 - T, MKT.euribor6m)) for T in range(1, 10)]
    m, errors = calibrate_q(MKT.estr, MKT.euribor6m, quotes, a=A)
    assert abs(m.sigma - 0.0105) < 1e-9 and np.max(np.abs(errors)) < 1e-6


def test_q_calibration_on_synthetic_matrix():
    m, errors = calibrate_q(MKT.estr, MKT.euribor6m, swaption_vols(), a=A)
    assert 0.005 < m.sigma < 0.015 and np.max(np.abs(errors)) < 5  # bp


def test_p_calibration():
    tau = np.array([1.0, 5.0, 10.0])
    g = (1 - np.exp(-A * tau)) / (A * tau)
    rng = np.random.default_rng(0)
    # Gaussian increments whose daily standard deviation is exactly σ·g/√252
    z = rng.standard_normal((5000, 3))
    z = (z - z.mean(0)) / z.std(0, ddof=1)
    import pandas as pd
    levels = np.vstack([np.zeros(3), np.cumsum(z * 0.008 * g / np.sqrt(252), axis=0)])
    hist = pd.DataFrame(levels, columns=tau)
    assert abs(calibrate_p(hist, A) - 0.008) < 1e-12
    assert 0.002 < calibrate_p(ecb_history(), A) < 0.02
