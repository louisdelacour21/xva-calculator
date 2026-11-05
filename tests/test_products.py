import numpy as np
import pytest
from scipy.stats import norm

from xva.exposure import grid
from xva.hw1f import HW1F, swap_coupons
from xva.market import FlatCurve, Market, sample_market
from xva.products import Trade, _beta, cube_mtm, index, schedule, swap, value

A, SIGMA = 0.03, 0.009
MKT = sample_market()
HW = HW1F(A, SIGMA, MKT.estr)
IRS = swap("IRS", "irs", 1, 1.0, 0.026, 0.0, 5)
OIS = swap("OIS", "ois", -1, 1.0, 0.024, 0.5, 4)
FWD = swap("FWD", "irs", -1, 1.0, 0.028, 1.0, 3)
TRADES = [IRS, OIS, FWD]


@pytest.fixture(scope="module")
def cube():
    _, t = grid(TRADES, mpors=())
    sim = HW.simulate(t, 40_000, seed=3)
    return sim, cube_mtm(TRADES, HW, MKT, sim)


def v0(trades, market=MKT, model=HW):
    return value(trades, model, market, 0.0, 0.0, 1.0)[0]


def pv_flows_after(tr: Trade, t: float) -> float:
    """Time-0 value of the cash flows paid after t only (closed form, floating flows at their forward value)."""
    w = tr.side * tr.nominal
    dk = np.diff(np.r_[tr.start, tr.fixed])
    fixed = -w * tr.K * np.sum((dk * MKT.estr(tr.fixed))[tr.fixed > t])
    s, e = tr.floating_periods()
    floating = w * np.sum(((_beta(tr, MKT) * MKT.estr(s) - MKT.estr(e)))[e > t])
    return fixed + floating


def _ql_flat(ql, day, r):
    return ql.YieldTermStructureHandle(ql.FlatForward(day, r, ql.Actual365Fixed(), ql.Continuous, ql.NoFrequency))


def test_irs_multi_curve_quantlib():
    """1Y×5Y payer IRS, 6M projection ≠ €STR discounting, against DiscountingSwapEngine (< 1e-10 of notional)."""
    ql = pytest.importorskip("QuantLib")
    day = ql.Date(3, 11, 2026)
    ql.Settings.instance().evaluationDate = day
    dc = ql.Actual365Fixed()
    disc, proj = _ql_flat(ql, day, 0.025), _ql_flat(ql, day, 0.027)
    start, end = day + ql.Period(1, ql.Years), day + ql.Period(6, ql.Years)
    sched = lambda p: ql.Schedule(start, end, ql.Period(p), ql.NullCalendar(), ql.Unadjusted, ql.Unadjusted,
                                  ql.DateGeneration.Forward, False)
    idx = ql.IborIndex("E6M", ql.Period(6, ql.Months), 0, ql.EURCurrency(), ql.NullCalendar(), ql.Unadjusted,
                       False, dc, proj)
    K = 0.026
    qs = ql.VanillaSwap(ql.Swap.Payer, 1.0, sched(ql.Annual), K, dc, sched(ql.Semiannual), idx, 0.0, dc)
    qs.setPricingEngine(ql.DiscountingSwapEngine(disc))
    yf = lambda s: np.array([dc.yearFraction(day, d) for d in list(s)[1:]])
    tr = Trade("ql", "irs", 1, 1.0, K, dc.yearFraction(day, start), yf(sched(ql.Annual)), yf(sched(ql.Semiannual)))
    market = Market(FlatCurve(0.025), FlatCurve(0.027))
    assert abs(v0([tr], market, HW1F(A, SIGMA, market.estr))[0] - qs.NPV()) < 1e-10


def test_ois_quantlib():
    ql = pytest.importorskip("QuantLib")
    day = ql.Date(3, 11, 2026)
    ql.Settings.instance().evaluationDate = day
    dc = ql.Actual365Fixed()
    disc = _ql_flat(ql, day, 0.025)
    sched = ql.Schedule(day, day + ql.Period(4, ql.Years), ql.Period(ql.Annual), ql.NullCalendar(),
                        ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False)
    K = 0.024
    qs = ql.OvernightIndexedSwap(ql.Swap.Receiver, 1.0, sched, K, dc, ql.Estr(disc))
    qs.setPricingEngine(ql.DiscountingSwapEngine(disc))
    tr = Trade("ql", "ois", -1, 1.0, K, 0.0, *[np.array([dc.yearFraction(day, d) for d in list(sched)[1:]])] * 2)
    market = Market(FlatCurve(0.025), FlatCurve(0.025))
    # QuantLib compounds TARGET business day by business day, non-business period ends included: ~1e-8 observed
    assert abs(v0([tr], market, HW1F(A, SIGMA, market.estr))[0] - qs.NPV()) < 1e-7  # 0.001 bp, spec says 0.1 bp


def test_mtm_martingale(cube):
    """E[D(0,t)·V(t)] = time-0 value of the remaining flows, mid-period too (coupon fixed on the path)."""
    sim, c = cube
    for u in (3 / 12, 1.0, 21 / 12, 31 / 12, 47 / 12):  # dates of the monthly grid
        k = index(sim["t"], u)
        for j, tr in enumerate(TRADES):
            dv = sim["D"][:, k] * c[:, k, j]
            assert abs(dv.mean() - pv_flows_after(tr, u)) < 3 * dv.std() / np.sqrt(len(dv)) + 1e-12, (u, tr.name)


def test_single_swap_ee_equals_swaption(cube):
    """Without collateral, EE*(t) of a payer = payer swaption expiring at t on the residual swap."""
    sim, c = cube
    for u in (1.0, 2.0, 3.0, 4.0):
        k = index(sim["t"], u)
        ee = sim["D"][:, k] * np.maximum(c[:, k, 0], 0)
        nominal, times, coupons = swap_coupons(u, 5 - u, IRS.K, MKT.estr, MKT.euribor6m)
        assert abs(ee.mean() - HW.jamshidian(u, times, coupons, nominal)) < 3 * ee.std() / np.sqrt(len(ee))


def test_pfe_closed_form_quantile(cube):
    """Single swap at a cash-flow date: V monotone in Gaussian x(t), so PFE = V(t, x_q), within the binomial CI."""
    sim, c = cube
    n = len(c)
    for u in (1.0, 3.0):
        k = index(sim["t"], u)
        xq = norm.ppf(0.975) * SIGMA * np.sqrt((1 - np.exp(-2 * A * u)) / (2 * A))
        pfe = value([IRS], HW, MKT, u, xq, 1.0)[0, 0]
        share = np.mean(np.maximum(c[:, k, 0], 0) <= pfe)
        assert abs(share - 0.975) < 3 * np.sqrt(0.975 * 0.025 / n)


SEASONED_K, SEASONED_L = 0.026, 0.031
S_IRS = swap("S_IRS", "irs", 1, 1.0, SEASONED_K, -0.25, 5.0, last_fixing=SEASONED_L)
S_OIS = swap("S_OIS", "ois", 1, 1.0, SEASONED_K, -0.25, 5.0, last_fixing=SEASONED_L)


def test_schedule_merges_a_very_short_broken_period():
    d = schedule(0.0, 10.008, 1)  # a calendar 10-year trade
    assert len(d) == 10 and d[0] == pytest.approx(1.008)
    assert len(schedule(0.0, 10.5, 1)) == 11  # a real stub stays
    assert schedule(-0.25, 4.75, 2)[0] == pytest.approx(0.25)  # seasoned: next payment in 3 months


def test_seasoned_closed_form_at_zero():
    """Known running coupon, then the regular flows of a forward swap on the same dates."""
    P = MKT.estr
    annuity = np.sum(P(np.arange(5) + 0.75))
    tail = v0([swap("F", "irs", 1, 1.0, 0.0, 0.25, 4.5)])[0]
    assert v0([S_IRS])[0] == pytest.approx(SEASONED_L * 0.5 * P(0.25) + tail - SEASONED_K * annuity, abs=1e-12)
    expected = 1 + 0.25 * SEASONED_L - P(4.75) - SEASONED_K * annuity  # accrued €STR compounds to the next date
    assert v0([S_OIS])[0] == pytest.approx(expected, abs=1e-12)


def test_seasoned_side_and_missing_fixing():
    receiver = swap("R", "irs", -1, 1.0, SEASONED_K, -0.25, 5.0, last_fixing=SEASONED_L)
    assert v0([receiver])[0] == pytest.approx(-v0([S_IRS])[0])
    today = swap("T", "irs", 1, 1.0, SEASONED_K, -0.25, 5.0)
    rate = (1 / MKT.euribor6m(0.5) - 1) / 0.5
    assert v0([today])[0] - v0([S_IRS])[0] == pytest.approx((rate - SEASONED_L) * 0.5 * MKT.estr(0.25), abs=1e-12)


def test_seasoned_martingale_before_and_after_the_first_flow():
    trades = [S_IRS, S_OIS]
    _, t = grid(trades, mpors=())
    sim = HW.simulate(t, 40_000, seed=7)
    c = cube_mtm(trades, HW, MKT, sim)
    V0 = v0(trades)
    k = 0  # before the first flow, at 0.25 and 0.75: nothing paid yet
    assert t[k] < 0.25
    for j in range(2):
        dv = sim["D"][:, k] * c[:, k, j]
        assert abs(dv.mean() - V0[j]) < 3 * dv.std() / np.sqrt(len(dv)) + 1e-12
    k = int(np.searchsorted(t, 0.25 + 1e-9))  # the IRS has paid its known coupon
    dv = sim["D"][:, k] * c[:, k, 0]
    expected = V0[0] - SEASONED_L * 0.5 * MKT.estr(0.25)
    assert abs(dv.mean() - expected) < 3 * dv.std() / np.sqrt(len(dv)) + 1e-12
