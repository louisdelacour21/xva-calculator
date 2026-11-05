import io
import json
from datetime import date

import pandas as pd
import pytest

from xva.initial_margin import SOURCE_OWN, SOURCE_USER, load_params
from xva.inputs import (InputError, _number, _years, build_config, default_settings, default_tables, excel, load_files,
                        parse_stress, preset_curves, tidy, workbook)

SETTINGS = default_settings() | {"valuation_date": "2026-01-15"}


def with_trades(*rows, **more):
    return default_tables() | {"trades": pd.DataFrame(rows)} | more


def problems(tables, settings=SETTINGS):
    with pytest.raises(InputError) as e:
        build_config(tables, settings)
    return e.value.errors


TRADE = {"name": "A", "type": "IRS", "side": "payer", "notional": 1e6, "fixed_rate_pct": 2.5, "start": "0", "tenor": 5}


@pytest.mark.parametrize("text, amount, expected", [
    ("1 000 000", True, 1e6), ("1,000,000", True, 1e6), ("1.000.000,5", True, 1000000.5), ("1,000.5", True, 1000.5),
    ("2,5", False, 2.5), ("2.5%", False, 2.5), ("50,000", True, 50000.0), ("50,000", False, 50.0),
    ("-1 000,5", True, -1000.5), ("1 000", True, 1000.0), (3, False, 3.0),
])
def test_number_reads_what_people_type(text, amount, expected):
    assert _number(text, amount) == expected


@pytest.mark.parametrize("bad", ["nan", "inf", "abc", "", True])
def test_number_rejects_the_rest(bad):
    with pytest.raises(ValueError):
        _number(bad)


def test_dates_and_years():
    today = date(2026, 1, 15)
    assert _years("3.5", today) == 3.5
    assert _years("-1", today) == -1.0
    expected = (date(2030, 1, 15) - today).days / 365
    assert _years("2030-01-15", today) == expected
    assert _years("15.01.2030", today) == expected  # would read as a number otherwise
    assert _years("15/01/2030", today) == expected


def test_aliases_and_tolerant_cells():
    errors = []
    df = tidy("trades", pd.DataFrame({"Trade ID": ["A"], "Product": ["swap"], "Direction": ["pay"], "Nominal": ["1 000 000"],
                                      "Strike": ["2,5"], "Start Date": ["2030-01-15"], "Maturity Date": ["15.01.2035"]}), errors)
    assert not errors
    row = df.iloc[0]
    assert (row["name"], row["type"], row["side"]) == ("A", "IRS", "payer")
    assert (row["notional"], row["fixed_rate_pct"]) == (1e6, 2.5)
    assert (row["start"], row["maturity"]) == ("2030-01-15", "15.01.2035")


def test_tidy_names_the_row_and_the_column():
    errors = []
    tidy("trades", pd.DataFrame({"name": ["A", "B"], "side": ["pay", "both"], "notional": ["x", 1]}), errors)
    assert any(e.startswith("trades row 2, side") for e in errors)
    assert any(e.startswith("trades row 1, notional") for e in errors)


def test_every_problem_is_listed_at_once():
    errors = problems(with_trades(
        TRADE | {"name": "neg", "notional": -5},
        TRADE | {"name": "pct", "fixed_rate_pct": 250},
        TRADE | {"name": "both", "maturity": "2030-01-15"},
        TRADE | {"name": "old", "start": "2010-01-15"},
        TRADE | {"name": "bad", "start": "soon"},
    ))
    text = "\n".join(errors)
    assert "row 1 (neg), notional" in text and "row 2 (pct), fixed_rate_pct" in text and "row 3 (both)" in text
    assert "row 4 (old): already matured" in text and "row 5 (bad), start" in text


def test_empty_netting_set_and_market_tables():
    t = default_tables()
    assert any("netting set is empty" in e for e in problems(t | {"trades": t["trades"].iloc[0:0]}))
    assert any("curves" in e for e in problems(t | {"curves": t["curves"].iloc[0:0]}))
    assert any("duplicate tenor" in e for e in problems(t | {"credit": pd.concat([t["credit"], t["credit"].iloc[:1]])}))
    assert any("vols" in e for e in problems(t | {"vols": t["vols"].iloc[0:0]}))
    assert build_config(t | {"vols": t["vols"].iloc[0:0]}, SETTINGS | {"sigma_override_pct": 1.0})["sigma"] == 0.01


def test_seasoned_trades():
    cfg = build_config(with_trades(TRADE | {"start": "-1", "last_fixing_pct": 3.0},
                                   TRADE | {"name": "B", "start": "2025-01-15", "tenor": None, "maturity": "2031-01-15"}), SETTINGS)
    a, b = cfg["trades"]
    assert (a["start"], a["tenor"], a["last_fixing"]) == (-1.0, 5.0, 0.03)
    assert b["start"] == pytest.approx(-1.0, abs=1e-9) and b["start"] + b["tenor"] == pytest.approx(5.0, abs=0.01)
    assert any("B" in w and "last_fixing_pct" in w for w in cfg["warnings"])


def test_blank_fixed_rate_means_par_for_candidates_only():
    t = default_tables()
    assert build_config(t, SETTINGS)["candidates"][0]["K"] is None
    assert any("fixed_rate_pct: required" in e for e in problems(with_trades(TRADE | {"fixed_rate_pct": None})))


def test_settings_bounds_and_rounding():
    t = default_tables()
    assert any("settings, paths" in e for e in problems(t, SETTINGS | {"paths": 5}))
    assert any("settings, sector" in e for e in problems(t, SETTINGS | {"sector": "unknown"}))
    assert any("settings, valuation_date" in e for e in problems(t, SETTINGS | {"valuation_date": "soon"}))
    assert build_config(t, SETTINGS | {"paths": 1001})["paths"] == 1000  # antithetic pairs


def test_csa_switch():
    t = default_tables()
    assert build_config(t, SETTINGS | {"csa_enabled": False})["csa"] is None
    assert build_config(t, SETTINGS | {"mpor_days": 20})["csa"]["mpor"] == pytest.approx(28 / 365)


def test_netting_set_selector():
    t = with_trades(TRADE | {"netting_set": "X"}, TRADE | {"name": "B", "netting_set": "Y"})
    assert [q["name"] for q in build_config(t, SETTINGS, "Y")["trades"]] == ["B"]


def test_excel_round_trip_keeps_everything():
    tables, settings = default_tables(), SETTINGS | {"grid_step_months": 1, "csa_enabled": True, "sigma_override_pct": None}
    read, back, notes = load_files([("inputs.xlsx", workbook(tables, settings, readme=True))])
    assert not notes and set(read) == set(tables)
    assert back["grid_step_months"] == 1 and back["grid_step_months"] is not True and back["csa_enabled"] is True
    assert back["sigma_override_pct"] is None
    assert json.dumps(build_config(read, settings), sort_keys=True) == json.dumps(build_config(tables, settings), sort_keys=True)


def test_workbook_writes_numeric_starts_as_numbers():
    tables = with_trades(TRADE | {"start": "1.5"}, TRADE | {"name": "B", "start": "2030-01-15"})
    sheet = pd.read_excel(io.BytesIO(workbook(tables, SETTINGS)), sheet_name="trades")
    assert sheet["start"].tolist() == [1.5, "2030-01-15"]


def test_workbook_ignores_extra_sheets_and_columns():
    data = excel({"trades": pd.DataFrame([TRADE | {"desk": "rates"}]), "Notes": pd.DataFrame({"a": [1]})})
    tables, _, notes = load_files([("book.xlsx", data)])
    assert set(tables) == {"trades"}
    assert any("'Notes' ignored" in n for n in notes) and any("desk" in n for n in notes)


def test_single_sheet_found_by_its_columns():
    tables, _, _ = load_files([("export.xlsx", excel({"Sheet1": pd.DataFrame([TRADE])}))])
    assert set(tables) == {"trades"}


def test_french_csv_named_after_the_table():
    data = "nom;type;side;notional;rate;start;tenor\nA;IRS;payeur;1 000 000;2,5;0;5\n".replace("nom", "name").replace("payeur", "payer")
    tables, _, _ = load_files([("mes_trades.csv", data.encode("latin-1"))])
    row = tables["trades"].iloc[0]
    assert (row["notional"], row["fixed_rate_pct"]) == (1e6, 2.5)


def test_csv_found_by_its_columns_with_a_byte_order_mark():
    tables, _, _ = load_files([("export.csv", "﻿name,type,side,notional,fixed_rate_pct,start,tenor\nA,OIS,rec,5000000,2.5,0,5\n".encode())])
    assert tables["trades"].iloc[0]["type"] == "OIS" and tables["trades"].iloc[0]["side"] == "receiver"


def test_long_integer_trade_id_is_kept():
    data = excel({"trades": pd.DataFrame([TRADE | {"name": 1234567890123}])})
    assert load_files([("book.xlsx", data)])[0]["trades"].iloc[0]["name"] == "1234567890123"


@pytest.mark.parametrize("name, data, message", [
    ("unknown.csv", b"a,b\n1,2\n", "cannot tell what"),
    ("broken.xlsx", b"not a workbook", "cannot read this file"),
    ("notes.xlsx", excel({"Notes": pd.DataFrame({"a": [1]}), "More": pd.DataFrame({"b": [1]})}), "no table found"),
    ("settings.csv", b"setting,value\nmagic,1\n", "unknown setting"),
    ("settings.csv", b"setting,value\npaths,1\n", "settings, paths"),
    ("trades.csv", b"name,side\nA,both\n", "trades row 1, side"),
])
def test_bad_files_are_refused_with_a_reason(name, data, message):
    with pytest.raises(InputError) as e:
        load_files([(name, data)])
    assert any(message in x for x in e.value.errors)


def test_settings_file_reads_french_numbers():
    _, settings, _ = load_files([("settings.csv", "setting;value\npfe_level_pct;97,5\ncsa_enabled;no\nsector;Financial\n".encode())])
    assert settings == {"pfe_level_pct": 97.5, "csa_enabled": False, "sector": "financial"}


def test_stress_scenarios():
    assert len(parse_stress(default_tables()["stress"])) == 5
    with pytest.raises(InputError, match="multipliers"):
        parse_stress(pd.DataFrame([{"scenario": "x", "vol_mult": 0}]))
    with pytest.raises(InputError, match="rates_bp"):
        parse_stress(pd.DataFrame([{"scenario": "x", "rates_bp": 5000}]))
    s = parse_stress(pd.DataFrame([{"scenario": None, "rates_bp": 50}]))[0]
    assert s == {"name": "Scenario 1", "rates_bp": 50.0, "vol": 1.0, "counterparty": 1.0, "bank": 1.0}


@pytest.mark.parametrize("preset", ["ecb", "09"])
def test_curve_presets_are_valid(preset):
    _, table = preset_curves(preset)
    cfg = build_config(default_tables() | {"curves": table}, SETTINGS)
    assert len(cfg["curves"]["tenors"]) > 3 and all(0 < t < 60 for t in cfg["curves"]["tenors"])


def with_positions(*rows):
    return default_tables() | {"positions": pd.DataFrame(rows)}


@pytest.mark.parametrize("row, message", [
    ({"name": "X"}, "give a notional or a DV01"),
    ({"name": "X", "dv01": -5}, "dv01: must be positive"),
    ({"name": "X", "notional": -5}, "notional: must be positive"),
    ({"name": "X", "notional": 1e6, "maturity": -1}, "already matured"),
    ({"name": "X", "notional": 1e6, "maturity": 90}, "beyond 60 years"),
    ({"name": "X", "notional": 1e6, "maturity": "soon"}, "neither a number of years nor a date"),
])
def test_positions_refused(row, message):
    assert any(message in e for e in problems(with_positions(row)))


def test_positions_are_read():
    cfg = build_config(with_positions({"name": "A", "side": "receiver", "pv01": "1 500", "maturity": "2031-01-15"},
                                      {"notional": 1e6}), SETTINGS)
    a, b = cfg["positions"]
    assert (a["side"], a["dv01"], a["notional"]) == (-1, 1500.0, None) and a["maturity"] == pytest.approx(5.0, abs=0.01)
    assert b["name"].startswith("position") and (b["side"], b["notional"], b["maturity"]) == (None, 1e6, None)


def test_positions_follow_the_netting_set():
    tables = with_trades(TRADE | {"netting_set": "N1"}, TRADE | {"name": "B", "netting_set": "N2"},
                         positions=pd.DataFrame([{"name": "in N1", "notional": 1e6, "netting_set": "N1"},
                                                 {"name": "in N2", "notional": 1e6, "netting_set": "N2"},
                                                 {"name": "blank", "notional": 1e6}]))
    assert [q["name"] for q in build_config(tables, SETTINGS, "N1")["positions"]] == ["in N1", "blank"]


def test_a_trades_file_with_a_dv01_column_stays_trades():
    data = "name,type,side,notional,fixed_rate_pct,start,tenor,dv01\nA,IRS,payer,1000000,2.5,0,5,500\n"
    assert set(load_files([("book.csv", data.encode())])[0]) == {"trades"}


def test_a_positions_file_is_recognised():
    assert set(load_files([("book.csv", b"name,dv01,maturity\nA,500,7\n")])[0]) == {"positions"}


def test_im_params_source_label():
    assert build_config(default_tables(), SETTINGS)["im"]["source"] == SOURCE_OWN
    changed = load_params()
    changed.loc[3, "risk_weight_bp"] = 99.0
    assert build_config(default_tables() | {"im_params": changed}, SETTINGS)["im"]["source"] == SOURCE_USER


def test_im_params_refused():
    assert any("one row per vertex" in e for e in problems(default_tables() | {"im_params": load_params().iloc[:-1]}))
    asymmetric = load_params()
    asymmetric.loc[0, "rho_30y"] = 0.99
    assert any("symmetric" in e for e in problems(default_tables() | {"im_params": asymmetric}))
    indefinite = load_params()  # 2w, 1m and 3m move together in the calibration: uncorrelating two of them is impossible
    indefinite.loc[0, "rho_1m"] = indefinite.loc[1, "rho_2w"] = 0.0
    assert any("positive semi-definite" in e for e in problems(default_tables() | {"im_params": indefinite}))
    above_one = load_params()
    above_one.loc[0, "rho_1m"] = above_one.loc[1, "rho_2w"] = 1.5
    assert problems(default_tables() | {"im_params": above_one})
