import numpy as np
import pytest

from xva.exposure import exposure, grid
from xva.hw1f import HW1F
from xva.market import sample_market
from xva.products import _beta, cube_mtm, index, swap, value
from xva.xva import (Credit, Survival, adjustments, allocation_euler, bootstrap, cds_schedule, cds_spread,
                     contributions, kva, load_credit, mva, pre_trade, risk_free, weights)

A, SIGMA = 0.03, 0.009
MKT = sample_market()
HW = HW1F(A, SIGMA, MKT.estr)
SPREADS = {1.0: 0.0090, 3.0: 0.0110, 5.0: 0.0130, 7.0: 0.0145, 10.0: 0.0160}
CREDIT = load_credit(P=MKT.estr)


def test_bootstrap_reprices_spreads():
    s = bootstrap(SPREADS, MKT.estr)
    for T, x in SPREADS.items():
        assert abs(cds_spread(s, MKT.estr, cds_schedule(T)) - x) < 1e-12
    u = np.linspace(0, 12, 200)
    assert s(0.0) == 1.0 and np.all(np.diff(s(u)) < 0) and np.all(s.lam > 0)


def test_cds_quantlib():
    """Same survival curve in QuantLib (HazardRateCurve): MidPointCdsEngine par spread < 0.1 bp."""
    ql = pytest.importorskip("QuantLib")
    day = ql.Date(3, 11, 2026)
    ql.Settings.instance().evaluationDate = day
    dc = ql.Actual365Fixed()
    disc = ql.YieldTermStructureHandle(ql.FlatForward(day, 0.025, dc, ql.Continuous, ql.NoFrequency))
    days = lambda T: int(round(T * 365))
    s = bootstrap(SPREADS, MKT.estr)
    pillars = [day] + [day + days(T) for T in s.pillars]
    hc = ql.DefaultProbabilityTermStructureHandle(ql.HazardRateCurve(pillars, [s.lam[0], *s.lam], dc))
    for T in (1, 2, 5, 10):
        dates = [day + days(u) for u in np.r_[0, cds_schedule(T)]]
        sched = ql.Schedule(dates, ql.NullCalendar(), ql.Unadjusted)
        cds = ql.CreditDefaultSwap(ql.Protection.Buyer, 1.0, 0.01, sched, ql.Unadjusted, dc, True, True, day)
        cds.setPricingEngine(ql.MidPointCdsEngine(hc, s.R, disc))
        times = np.array([dc.yearFraction(day, d) for d in dates[1:]])
        assert abs(cds_spread(s, MKT.estr, times) - cds.fairSpread()) < 1e-5, T  # 0.1 bp


def _residual(tr, t):
    """Residual swap at t (a fixing date) written nominal - Σ c_j P(t, T_j) for Jamshidian."""
    c, nominal = {}, 0.0
    for T, d in zip(tr.fixed, np.diff(np.r_[tr.start, tr.fixed])):
        if T > t:
            c[T] = c.get(T, 0) + tr.K * d
    for s, e, b in zip(*tr.floating_periods(), _beta(tr, MKT)):
        if s >= t:
            if abs(s - t) < 1e-12:
                nominal += b
            else:
                c[s] = c.get(s, 0) - b
            c[e] = c.get(e, 0) + 1
    return nominal, np.array(list(c)), np.array(list(c.values()))


def test_single_swap_cva_equals_swaptions():
    """Without collateral, CVA of a payer = (1-R) Σ ΔPD·S_B·Swaption(t̄) (closed form), within 2%."""
    tr = swap("P", "irs", 1, 1.0, 0.026, 0.0, 5)
    base, t = grid([tr], mpors=(), step=0.5)
    sim = HW.simulate(t, 40_000, seed=7)
    V0 = value([tr], HW, MKT, 0.0, 0.0, 1.0)[0, 0]
    expo = exposure(cube_mtm([tr], HW, MKT, sim)[:, :, 0], sim["D"], t, base, V0)
    cva, se = adjustments(expo, CREDIT, V0)["CVA"]
    swn = np.array([HW.jamshidian(u, *_residual(tr, u)[1:], _residual(tr, u)[0]) for u in base[:-1]] + [0.0])
    closed = np.sum((np.r_[max(V0, 0), swn[:-1]] + swn) / 2 * weights(base, CREDIT)["CVA"])
    assert abs(cva / closed - 1) < 0.02 and abs(cva - closed) < 3 * se


@pytest.fixture(scope="module")
def ns():
    trades = [swap("A", "irs", 1, 1.0, 0.024, 0.0, 6), swap("B", "irs", -1, 0.6, 0.03, 1.0, 4),
              swap("C", "ois", 1, 0.8, 0.026, 0.0, 3)]
    base, t = grid(trades)
    sim = HW.simulate(t, 10_000, seed=11)
    c = cube_mtm(trades, HW, MKT, sim)
    V0 = value(trades, HW, MKT, 0.0, 0.0, 1.0)[0]
    return trades, base, t, sim, c, V0


def test_euler_allocation_sums(ns):
    trades, base, t, sim, c, V0 = ns
    cva = adjustments(exposure(c.sum(-1), sim["D"], t, base, V0.sum()), CREDIT, V0.sum())["CVA"][0]
    alloc = allocation_euler(c, sim["D"], t, base, V0, CREDIT)
    assert abs(alloc.sum() - cva) < 1e-12 * abs(cva) + 1e-15


def test_pre_trade_equals_full_recompute(ns):
    """ΔXVA computing only the new trade = difference of two full computations on the same paths."""
    trades, base, t, sim, c, V0 = ns
    csa = {"H_C": 0.05, "H_B": 0.05, "MTA": 0.005, "mpor": 14 / 365}
    new = swap("N", "irs", -1, 0.5, 0.025, 0.5, 4)
    r = pre_trade(c.sum(-1), V0.sum(), new, HW, MKT, sim, base, CREDIT, csa)
    all_ = trades + [new]
    c2 = cube_mtm(all_, HW, MKT, sim)
    V02 = value(all_, HW, MKT, 0.0, 0.0, 1.0)[0].sum()
    after = adjustments(exposure(c2.sum(-1), sim["D"], t, base, V02, csa), CREDIT, V02)
    before = adjustments(exposure(c.sum(-1), sim["D"], t, base, V0.sum(), csa), CREDIT, V0.sum())
    for k in ("CVA", "DVA", "FCA", "FBA"):
        assert abs(r["Δ"][k][0] - (after[k][0] - before[k][0])) < 1e-12
        assert r["Δ"][k][1] > 0


def test_cva_dva_symmetry(ns):
    """The bank's CVA is the counterparty's DVA (V → -V, roles swapped)."""
    trades, base, t, sim, c, V0 = ns
    V = c.sum(-1)
    b = contributions(exposure(V, sim["D"], t, base, V0.sum()), CREDIT)
    reverse = Credit(CREDIT.bank, CREDIT.counterparty, CREDIT.sF)
    cpt = contributions(exposure(-V, sim["D"], t, base, -V0.sum()), reverse)
    np.testing.assert_allclose(b["CVA"], cpt["DVA"], rtol=1e-13, atol=1e-18)
    np.testing.assert_allclose(b["DVA"], cpt["CVA"], rtol=1e-13, atol=1e-18)


def test_limit_cases(ns):
    trades, base, t, sim, c, V0 = ns
    expo = exposure(c.sum(-1), sim["D"], t, base, V0.sum())
    safe = Credit(risk_free(), risk_free(), 0.0)
    r = adjustments(expo, safe, V0.sum())
    assert all(r[k][0] == 0 for k in ("CVA", "DVA", "FCA", "FBA"))
    # funding spread equal to the cost of own credit: ℓ = 0, no FVA (no DVA double counting)
    flat = Survival(np.array([10.0]), np.array([0.01]))
    r = adjustments(expo, Credit(flat, flat, 0.6 * 0.01), V0.sum())
    assert abs(r["FCA"][0]) < 1e-15 and abs(r["FBA"][0]) < 1e-15 and r["CVA"][0] > 0  # ℓ zero up to rounding


def test_mva_and_kva_by_hand():
    # ℓ = 0.02 - 0.6·0.02 = 0.008, S_C·S_B = e^{-0.03 t}, two intervals of one year
    flat = lambda lam: Survival(np.array([10.0]), np.array([lam]))
    credit = Credit(flat(0.01), flat(0.02), 0.02)
    t, P = np.array([1.0, 2.0]), MKT.estr
    expected = 0.008 * 1e6 * (np.exp(-0.03) * P(0.5) + np.exp(-0.06) * P(1.5))
    assert mva(np.full(2, 1e6), 1e6, t, credit, P) == pytest.approx(expected, rel=1e-6)
    assert mva(np.full(2, 1e6), 1e6, t, Credit(flat(0.01), flat(0.02), 0.0), P) == 0  # ℓ floored at zero
    expected = 0.1 * (2.5e6 * P(0.5) * np.exp(-0.03) + 1.5e6 * P(1.5) * np.exp(-0.06))
    assert kva(np.array([3e6, 2e6, 1e6]), t, credit, P, 0.1) == pytest.approx(expected, rel=1e-6)
