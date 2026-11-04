from pathlib import Path

import numpy as np
import pytest

from xva.exposure import MPOR_10D, exposure, grid
from xva.hw1f import HW1F
from xva.market import sample_market
from xva.products import cube_mtm, load_portfolio, swap, value

MKT = sample_market()
HW = HW1F(0.03, 0.009, MKT.estr)
TRADES = [swap("A", "irs", 1, 1.0, 0.024, 0.0, 6), swap("B", "irs", -1, 0.6, 0.03, 1.0, 4),
          swap("C", "ois", 1, 0.8, 0.026, 0.0, 3)]
MPOR_20D = 28 / 365


@pytest.fixture(scope="module")
def state():
    base, t = grid(TRADES, mpors=(MPOR_10D, MPOR_20D))
    sim = HW.simulate(t, 20_000, seed=5)
    c = cube_mtm(TRADES, HW, MKT, sim)
    V0 = value(TRADES, HW, MKT, 0.0, 0.0, 1.0)[0]
    return base, t, sim, c, V0


def expo(state, csa=None, trades=None):
    base, t, sim, c, V0 = state
    j = list(range(c.shape[2])) if trades is None else trades
    return exposure(c[:, :, j].sum(-1), sim["D"], t, base, V0[j].sum(), csa)


def test_netting_subadditive(state):
    ns = expo(state)
    total = sum(expo(state, trades=[j])["EE"] for j in range(len(TRADES)))
    assert np.all(ns["EE"] <= total + 1e-15)


def test_mirror_swaps_zero_exposure():
    tr = [swap("P", "irs", 1, 1.0, 0.025, 0.0, 5), swap("R", "irs", -1, 1.0, 0.025, 0.0, 5)]
    base, t = grid(tr, mpors=())
    sim = HW.simulate(t, 2_000, seed=1)
    m = exposure(cube_mtm(tr, HW, MKT, sim).sum(-1), sim["D"], t, base, 0.0)
    assert np.max(m["EE"]) < 1e-15 and np.max(m["NEE"]) < 1e-15


def test_infinite_threshold_equals_no_collateral(state):
    without = expo(state)
    with_ = expo(state, {"H_C": 1e18, "H_B": 1e18, "MTA": 0.0, "mpor": MPOR_10D})
    np.testing.assert_array_equal(without["EE"], with_["EE"])
    np.testing.assert_array_equal(without["NEE"], with_["NEE"])


def test_perfect_collateral_zero_mpor(state):
    m = expo(state, {"H_C": 0.0, "H_B": 0.0, "MTA": 0.0, "mpor": 0.0})
    assert np.max(m["EE"]) == 0 and np.max(m["NEE"]) == 0


def test_mpor_monotonicity_and_square_root(state):
    """EE increasing in MPoR; ≈ √MPoR law tested away from cash-flow dates (15% tolerance)."""
    csa = lambda m: {"H_C": 0.0, "H_B": 0.0, "MTA": 0.0, "mpor": m}
    e10, e20 = expo(state, csa(MPOR_10D)), expo(state, csa(MPOR_20D))
    base = state[0]
    flows = np.unique(np.concatenate([tr.cashflows() for tr in TRADES] + [[tr.start for tr in TRADES]]))
    far = np.array([u > MPOR_20D and not np.any((flows > u - MPOR_20D - 1e-9) & (flows <= u + 1e-9)) for u in base])
    far &= e10["EE"] > 0
    assert far.sum() > 20
    assert np.mean(e20["EE"][far] >= e10["EE"][far]) > 0.95
    ratio = np.mean(e20["EE"][far] / e10["EE"][far])
    assert abs(ratio / np.sqrt(2) - 1) < 0.15


def test_eepe_and_pfe_consistent(state):
    m = expo(state)
    assert np.all(m["PFE"] >= m["EE"]) and m["EEPE"] >= m["EE"][m["t"] <= 1].mean()


def test_demo_portfolio():
    trades, csa = load_portfolio(Path(__file__).parents[1] / "portfolios" / "demo.yaml")
    mp = csa["mpor_days"] * 14 / 10 / 365
    base, t = grid(trades, mpors=(mp,))
    sim = HW.simulate(t, 4_000, seed=0)
    c = cube_mtm(trades, HW, MKT, sim)
    V0 = value(trades, HW, MKT, 0.0, 0.0, 1.0)[0].sum()
    without = exposure(c.sum(-1), sim["D"], t, base, V0)
    with_ = exposure(c.sum(-1), sim["D"], t, base, V0, {"H_C": csa["H_C"], "H_B": csa["H_B"],
                                                        "MTA": csa["MTA"], "mpor": mp})
    assert c.shape == (4_000, len(t), len(trades))
    assert np.all(with_["EE"] <= without["EE"] + 1e-6) and with_["EEPE"] < without["EEPE"]
