import itertools
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from xva.initial_margin import (LABELS, PARAMS, _aged_positions, _annuity, _forward, _position, _pv, _range, _schedule_rate,
                                backtest, calibrate, data_levels, im_profile, ladder, load_params, margin, params_table,
                                synthetic_im)
from xva.market import sample_market
from xva.products import swap
from xva.risk import rate_shock
from xva.validation import full_history

MKT = sample_market()
PAYER = swap("P", "irs", 1, 10e6, 0.025, 0.0, 10)
TABLE = load_params()
OWN = {"rw": TABLE["risk_weight_bp"].to_numpy(), "rho": TABLE[[f"rho_{v}" for v in LABELS]].to_numpy(), "phi": 0.98,
       "threshold": 1e12}
FLAT = {"rw": [10.0] * 12, "rho": 0.5 * (np.eye(12) + 1), "phi": 0.9, "threshold": 1e12}


def position(**kw):
    return {"name": "x", "side": 1, "notional": None, "dv01": None, "maturity": None} | kw


def one_vertex(vertex, amount=1000.0, row=1):
    s = np.zeros((2, 12))
    s[row, LABELS.index(vertex)] = amount
    return s


def test_parameters_on_file_are_the_calibration():
    c = calibrate(full_history())
    pd.testing.assert_frame_equal(TABLE, params_table(c["rw"], c["rho"]), check_dtype=False)
    assert PARAMS.read_text(encoding="utf-8").startswith("# Own calibration, not ISDA SIMM parameters.")


def test_correlations_on_file_are_usable():
    rho = OWN["rho"]
    assert np.allclose(rho, rho.T, atol=1e-6) and np.allclose(np.diag(rho), 1.0)
    assert rho.min() >= 0 and rho.max() <= 1  # the range relies on it
    assert np.linalg.eigvalsh(rho).min() > -1e-3


def test_ladder_sums_to_the_parallel_dv01():
    trades = [PAYER, swap("O", "ois", -1, 5e6, 0.02, 0.0, 3), swap("F", "irs", 1, 8e6, 0.025, 2.0, 5)]
    parallel = _pv(trades, rate_shock(MKT, np.ones_like, 1e-4)).sum() - _pv(trades, MKT).sum()
    assert ladder(trades, MKT).sum() == pytest.approx(parallel, rel=1e-3)
    assert not ladder([], MKT).any()


def test_ladder_of_a_payer_sits_at_maturity_on_the_index_curve():
    s = ladder([PAYER], MKT)
    assert np.abs(s).argmax() == 12 + LABELS.index("10y")
    assert s[1, LABELS.index("10y")] > 0


def test_a_running_coupon_carries_no_rate_risk():
    seasoned = swap("S", "irs", 1, 10e6, 0.025, -1.25, 10)
    fixing = float((1 / MKT.euribor6m(0.5) - 1) / 0.5)
    assert ladder([seasoned], MKT) == pytest.approx(ladder([replace(seasoned, last_fixing=fixing)], MKT))


def test_margin_by_hand():
    s = one_vertex("6m", 100.0) + one_vertex("2y", -50.0)
    assert margin(s, FLAT) == pytest.approx(np.sqrt(1000**2 + 500**2 - 2 * 0.5 * 1000 * 500))
    s = one_vertex("6m", 100.0, row=0) + one_vertex("6m", 100.0, row=1)
    assert margin(s, FLAT) == pytest.approx(np.sqrt(2 * 1000**2 + 2 * 0.9 * 1000**2))


def test_concentration_scales_only_above_the_threshold():
    s = one_vertex("6m")
    assert margin(s, FLAT) == pytest.approx(1e4)
    assert margin(s, FLAT | {"threshold": 1000.0}) == pytest.approx(1e4)
    assert margin(s, FLAT | {"threshold": 10.0}) == pytest.approx(1e4 * np.sqrt(1000 / 10))


@pytest.mark.parametrize("threshold", [1e12, 50.0])
def test_range_holds_over_every_sign(threshold):
    rng = np.random.default_rng(0)
    known, unknown = rng.normal(0, 100, (2, 12)), [rng.uniform(0, 100, (2, 12)) for _ in range(4)]
    p = OWN | {"threshold": threshold}
    central, low, high = _range(known, unknown, 0.0, p)
    cr = max(1.0, np.sqrt((abs(known.sum()) + sum(u.sum() for u in unknown)) / threshold))
    values = [margin(known + sum(x * u for x, u in zip(signs, unknown)), p, cr)
              for signs in itertools.product((1, -1), repeat=len(unknown))]
    assert low <= min(values) and max(values) <= high
    assert central == pytest.approx(np.sqrt(np.mean(np.square(values))))


def test_no_unknown_direction_no_range():
    central, low, high = _range(ladder([PAYER], MKT), [], 0.0, OWN)
    assert central == low == high == pytest.approx(margin(ladder([PAYER], MKT), OWN))


def test_each_level_uses_what_is_known():
    level, dv01, s = _position(position(dv01=1000.0, maturity=7.0), MKT, OWN)
    assert (level, dv01) == (2, 1000.0)
    assert s[1, LABELS.index("5y")] == pytest.approx(600) and s[1, LABELS.index("10y")] == pytest.approx(400)
    assert s[1].sum() == pytest.approx(1000) and s[0].sum() == 0
    level, _, s = _position(position(dv01=1000.0), MKT, OWN)
    assert level == 2 and s[1, int(np.argmax(OWN["rw"]))] == 1000.0
    level, dv01, _ = _position(position(notional=1e7, maturity=5.0), MKT, OWN)
    assert level == 3 and dv01 == pytest.approx(1e7 * sum(float(MKT.estr(T)) for T in range(1, 6)) * 1e-4)
    assert _position(position(notional=1e7), MKT, OWN)[0] == 4
    assert _position(position(notional=1e7, dv01=1000.0, maturity=5.0), MKT, OWN)[0] == 2


def test_notional_only_is_added_in_full():
    alone = synthetic_im([PAYER], [], MKT, OWN)["central"]
    r = synthetic_im([PAYER], [position(notional=1e7, side=None)], MKT, OWN)
    assert (r["central"], r["low"], r["high"]) == pytest.approx((alone + 4e5,) * 3)


def test_opposite_sides_net_and_a_blank_side_gives_a_range():
    q = {"dv01": 1000.0, "maturity": 5.0}
    assert synthetic_im([], [position(side=1, **q), position(side=-1, **q)], MKT, OWN)["central"] == pytest.approx(0, abs=1e-6)
    r = synthetic_im([PAYER], [position(side=None, **q)], MKT, OWN)
    payer = synthetic_im([PAYER], [position(side=1, **q)], MKT, OWN)["central"]
    receiver = synthetic_im([PAYER], [position(side=-1, **q)], MKT, OWN)["central"]
    assert r["low"] <= min(payer, receiver) <= r["central"] <= max(payer, receiver) <= r["high"]


def test_tables_describe_the_estimate():
    r = synthetic_im([PAYER], [position(dv01=1000.0, maturity=7.0), position(notional=2e7, maturity=4.0, side=None),
                               position(notional=1e7, side=None)], MKT, OWN)
    assert list(r["positions"]["data level"]) == ["DV01 known", "Notional and maturity", "Notional only"]
    assert list(r["positions"]["side"]) == ["payer", "unknown", "unknown"]
    assert r["coverage"]["share"].sum() == pytest.approx(1)
    assert r["coverage"]["positions"].sum() == 4
    assert r["ladder"]["direction unknown"].sum() == pytest.approx(r["positions"]["DV01 used"].iloc[1])


@pytest.mark.parametrize("duration, rate", [(0.5, 0.01), (1.99, 0.01), (2.0, 0.02), (4.99, 0.02), (5.0, 0.04), (20.0, 0.04)])
def test_regulatory_schedule(duration, rate):
    assert _schedule_rate(duration) == rate


def test_data_levels_price_what_is_missing():
    trades = [swap("A", "irs", 1, 10e6, 0.025, 0.0, 10), swap("B", "irs", -1, 20e6, 0.025, 0.0, 3)]
    t = data_levels(trades, MKT, OWN, np.array([3.0, -1.0])).set_index("data available")
    gross = 10e6 * 0.04 + 20e6 * 0.02
    assert t.loc["Notional", "IM"] == pytest.approx(0.04 * 30e6)
    assert t.loc["Notional, maturity (schedule)", "IM"] == pytest.approx(gross)
    assert t.loc["Notional, maturity, MtM (schedule)", "IM"] == pytest.approx((0.4 + 0.6 * 2 / 3) * gross)
    assert t["gap to full terms %"].iloc[0] == 0
    assert abs(t.loc["DV01, maturity, direction", "gap to full terms %"]) < 10
    row = t.loc["Notional, maturity"]
    assert row["low"] <= row["IM"] <= row["high"]


def test_profile_starts_today_and_runs_off():
    base = np.linspace(0.0, 10.5, 22)
    p = im_profile([PAYER], [], MKT, OWN, base, n=21)
    assert p["central"].iloc[0] == pytest.approx(synthetic_im([PAYER], [], MKT, OWN)["central"])
    assert p["central"].iloc[-1] == 0
    q = im_profile([PAYER], [position(dv01=1000.0, maturity=7.0, side=None), position(notional=1e7)], MKT, OWN, base, n=21)
    assert (q["low"] <= q["central"] + 1e-9).all() and (q["central"] <= q["high"] + 1e-9).all()
    assert q["central"].iloc[-1] == pytest.approx(4e5)  # no maturity: stays flat


def test_aged_positions_run_off():
    fwd = _forward(MKT, 3.0)
    out = _aged_positions([position(name="gone", dv01=1000.0, maturity=2.0), position(name="dv01", dv01=1000.0, maturity=7.0),
                           position(name="flat", dv01=1000.0), position(name="proxy", notional=1e6, maturity=7.0)], MKT, fwd, 3.0)
    assert [q["name"] for q in out] == ["dv01", "flat", "proxy"]
    assert out[0]["maturity"] == 4.0
    assert out[0]["dv01"] == pytest.approx(1000 * _annuity(4.0, fwd) / _annuity(7.0, MKT))
    assert out[1]["dv01"] == 1000.0 and out[2]["dv01"] is None


def test_backtest_on_one_vertex_is_near_one_percent():
    b = backtest(one_vertex("10y"), OWN, full_history())
    assert 0.005 <= b["sample"] <= 0.015
    assert b["worst"] >= b["margin"]
