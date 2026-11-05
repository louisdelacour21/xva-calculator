import io

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from xva import pipeline
from xva.inputs import InputError, build_config, default_settings, default_tables, parse_stress
from xva.onepager import one_pager
from xva.pipeline import baseline_table, config_diff, engine_hash, full_stack, results_workbook, run, run_id, sensitivity_table, stress_table

SMALL = default_settings() | {"paths": 2000, "grid_step_months": 3}
TRADE = {"name": "A", "type": "IRS", "side": "payer", "notional": 10e6, "fixed_rate_pct": 2.5, "start": "0", "tenor": 5}


def config(settings=SMALL, **tables):
    return build_config(default_tables() | {k: pd.DataFrame(v) for k, v in tables.items()}, settings)


@pytest.fixture(scope="module")
def demo():
    return run(build_config(default_tables(), default_settings()))


def test_demo_regression(demo):
    """The demo netting set, typed through the inputs: same trades, curves and credit as the validated engine."""
    assert demo["V0"] == pytest.approx(5_088_036, abs=1)
    for name, expected in {"CVA": 446_270, "DVA": 159_451, "FCA": 57_014, "FBA": 23_476, "Adjusted value": 4_767_680}.items():
        assert demo["xva"][name][0] == pytest.approx(expected, rel=0.01), name
    assert demo["pretrade"]["trade"].tolist() == ["New payer 10Y", "New receiver 5Y", "All together"]
    assert not demo["checks"] and not demo["fit"]["error (bp)"].abs().gt(5).any()


def test_adjusted_value_identity(demo):
    x = {k: v[0] for k, v in demo["xva"].items()}
    assert x["Adjusted value"] == pytest.approx(demo["V0"] - x["CVA"] + x["DVA"] - x["FCA"] + x["FBA"])
    assert x["Total XVA"] == pytest.approx(demo["V0"] - x["Adjusted value"])


def test_euler_contributions_sum_to_the_uncollateralised_cva():
    r = run(config(SMALL | {"csa_enabled": False}))
    assert r["trades"]["Euler CVA"].sum() == pytest.approx(r["xva"]["CVA"][0], rel=1e-9)
    assert r["diversification"] >= 0
    assert list(r["profiles"].columns) == ["EE without CSA", "PFE 97.5% without CSA"]  # no CSA series


def test_par_candidate_off_the_grid():
    r = run(config(candidates=[{"name": "par", "type": "IRS", "side": "payer", "notional": 10e6, "start": "0.4", "tenor": 7.3},
                               {"name": "fixed", "type": "OIS", "side": "receiver", "notional": 5e6, "fixed_rate_pct": 2.0, "tenor": 3.17}]))
    p = r["pretrade"]
    assert abs(p["MtM"].iloc[0]) < 1e-6 and p["par rate %"].iloc[0] == p["fixed rate %"].iloc[0]
    assert p["trade"].iloc[-1] == "All together" and p["MtM"].iloc[-1] == pytest.approx(p["MtM"].iloc[:2].sum())
    assert np.isfinite(p["ΔXVA"]).all() and (p["ΔXVA SE"] > 0).all()


def test_no_candidate():
    assert run(config(candidates=[{"name": None}]))["pretrade"] is None


def test_seasoned_trade_with_a_date_and_a_fixing():
    r = run(config(SMALL | {"valuation_date": "2026-01-15"},
                   trades=[TRADE | {"start": "2025-04-15", "last_fixing_pct": 3.0, "tenor": 5}]))
    assert np.isfinite(r["V0"]) and r["xva"]["CVA"][0] > 0
    assert r["trades"]["start"].iloc[0] == pytest.approx(-0.75, abs=0.01)


def test_cash_margined_runs_with_zero_mpor_and_ia():
    settings = SMALL | {"mpor_days": 0, "threshold_counterparty": 0.0, "threshold_bank": 0.0, "mta": 0.0}
    plain = run(config(settings))["xva"]["CVA"][0]
    assert plain < 1e-6 * run(config(SMALL))["xva"]["CVA"][0] + 1.0  # continuous margining: no gap to lose
    assert run(config(settings | {"independent_amount": -2e6}))["xva"]["CVA"][0] > plain  # posted IA is at risk


def test_pfe_level_and_grid_step():
    r = run(config(SMALL | {"pfe_level_pct": 90.0}))
    assert "PFE 90.0% with CSA" in r["profiles"].columns
    assert run(config(SMALL | {"grid_step_months": 1}))["dates"] > r["dates"]


def test_sigma_override_skips_the_calibration():
    r = run(config(SMALL | {"sigma_override_pct": 1.0}, vols=[]))
    assert r["sigma"] == 0.01 and r["fit"].empty


def test_impossible_credit_curve_is_a_clear_error():
    credit = [{"tenor": 1, "counterparty_spread_bp": 5000, "bank_spread_bp": 50},
              {"tenor": 5, "counterparty_spread_bp": 1, "bank_spread_bp": 50}]
    with pytest.raises(InputError, match="credit, counterparty spreads"):
        run(config(credit=credit))


def test_memory_guard_names_the_limit(monkeypatch):
    monkeypatch.setattr(pipeline, "MAX_MB", 1.0)
    with pytest.raises(InputError, match="at most"):
        run(config())


def test_stress_base_is_the_run_on_the_same_draws():
    cfg = config()
    table = stress_table(cfg, parse_stress(pd.DataFrame([{"scenario": "Up", "rates_bp": 100}, {"scenario": "Spreads", "counterparty_spread_mult": 2}])))
    assert table["scenario"].tolist() == ["Base", "Up", "Spreads"] and table["change in total XVA"].iloc[0] == 0
    assert table["CVA"].iloc[0] == pytest.approx(run(cfg)["xva"]["CVA"][0], rel=1e-12)
    assert table["MtM"].iloc[1] > table["MtM"].iloc[0]  # the demo book pays fixed on balance
    assert table["CVA"].iloc[2] > 1.5 * table["CVA"].iloc[0]


def test_sensitivities_and_results_workbook():
    cfg = config(trades=[TRADE], candidates=[{"name": None}])
    sens = sensitivity_table(cfg, 200)
    assert sens["bump"].iloc[-1] == "vega +1%" and "delta parallel" in set(sens["bump"])
    assert {"CVA", "CVA SE", "Total XVA"} <= set(sens.columns) and np.isfinite(sens["Total XVA"]).all()
    assert sens.set_index("bump").loc["delta parallel", "CVA"] > 0  # a payer gains exposure when rates rise

    r = run(cfg)
    stress = stress_table(cfg, [])
    book = load_workbook(io.BytesIO(results_workbook(r, stress, sens)))
    assert {"Summary", "Profiles", "Trades", "CS01", "Sensitivities", "Stress", "Capital", "Model fit"} <= set(book.sheetnames)
    assert "Pre-trade" not in book.sheetnames  # no candidate, no empty sheet
    assert book["Summary"]["A2"].value == "Mark-to-market" and book["Summary"]["B2"].value == pytest.approx(r["V0"])


def test_mva_and_kva_stack_on_top_of_the_baselines(demo):
    im, k = demo["im"], demo["kva"]
    assert im["low"] <= im["central"] <= im["high"]
    assert 0 < im["mva"]["low"] <= im["mva"]["central"] <= im["mva"]["high"]
    w = im["what-if"].set_index("adjustment")
    assert w.loc["CVA", "with IM (central)"] < w.loc["CVA", "without IM"]
    assert w.loc["Total", "without IM"] == pytest.approx(demo["xva"]["Total XVA"][0])
    sign = pd.Series({"CVA": 1, "DVA": -1, "FCA": 1, "FBA": -1, "MVA": 1})
    assert w.loc["Total", "with IM (central)"] == pytest.approx((sign * w.loc[sign.index, "with IM (central)"]).sum())
    assert k["value"] > 0 and k["profile"]["Capital"].iloc[0] == pytest.approx(demo["capital"]["Total capital"])
    assert full_stack(demo) == pytest.approx(demo["xva"]["Total XVA"][0] + im["mva"]["central"] + k["value"])
    assert {"Initial margin", "IM ladder", "IM what-if", "KVA profile"} <= set(load_workbook(io.BytesIO(results_workbook(demo))).sheetnames)


def test_controls_pass_on_the_demo(demo):
    v = demo["validation"]
    assert len(v) == 8 and (v["status"] == "pass").all(), v.to_string()


def test_controls_hold_on_seasoned_and_ois_trades():
    book = [TRADE | {"name": "Seasoned", "start": "-1.25", "last_fixing_pct": 3.1}, TRADE | {"name": "OIS", "type": "OIS", "start": "-0.5", "last_fixing_pct": 2.2},
            TRADE | {"name": "Plain OIS", "type": "OIS", "side": "receiver"}, TRADE | {"name": "Forward", "start": "2"}]
    v = run(config(trades=book))["validation"].set_index("check")["status"]
    assert v[v.index.str.startswith("Martingale")].item() == "pass"
    assert v[v.index.str.startswith("Discount")].item() == "pass"


def test_martingale_control_catches_a_wrong_cube(monkeypatch):
    real = pipeline.cube_mtm
    monkeypatch.setattr(pipeline, "cube_mtm", lambda *a: real(*a) * 1.02)
    v = run(config())["validation"].set_index("check")["status"]
    assert v[v.index.str.startswith("Martingale")].item() == "FAIL"


def test_run_id_follows_the_inputs_and_not_the_clock(demo):
    cfg = build_config(default_tables(), default_settings())
    assert demo["audit"]["run ID"] == run_id(cfg) != run_id(cfg | {"seed": cfg["seed"] + 1})
    assert demo["audit"]["engine"] == engine_hash()


def test_config_diff_and_baseline(demo):
    cfg = {"paths": 10, "warnings": ["x"], "curves": {"estr": [1.0, 2.0]}, "trades": [{"K": 1}]}
    new = {"paths": 20, "warnings": [], "curves": {"estr": [1.0, 3.0]}, "trades": [{"K": 1}, {"K": 2}]}
    assert config_diff(cfg, new) == ["curves.estr[1]: 2.0 → 3.0", "paths: 10 → 20", "trades: 1 items → 2 items"]
    assert config_diff(cfg, cfg) == []
    same = baseline_table(demo, demo)
    assert (same["change"] == 0).all() and "XVA including MVA and KVA" in set(same["measure"])


def test_workbook_has_controls_and_log(demo):
    sheets = load_workbook(io.BytesIO(results_workbook(demo, log=[demo["audit"], demo["audit"]]))).sheetnames
    assert {"Controls", "Run log"} <= set(sheets)


def test_one_page_report_is_self_contained_and_escaped(demo):
    cfg = build_config(default_tables(), default_settings())
    html = one_pager(demo | {"checks": ["<script>alert(1)</script>"]}, cfg)
    assert demo["audit"]["run ID"] in html and "<svg" in html and "not ISDA SIMM" in html
    assert "<script" not in html and "&lt;script&gt;" in html
    assert "http" not in html  # nothing is fetched when it is opened
