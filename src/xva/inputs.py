"""User inputs: tables typed in the app or read from CSV / Excel files, checked and turned into one configuration.

Everything crosses a trust boundary here: errors name the sheet, the row and the column, and nothing reaches the
engine unchecked. A workbook has one sheet per table (see SHEETS); a CSV holds one table, found by file name or columns.
"""
import io
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .demo import PORTFOLIO
from .initial_margin import LABELS, SOURCE_OWN, SOURCE_USER, load_params
from .market import ROOT, ecb_latest, swaption_vols
from .risk import RW_MAR50

TRADE_COLUMNS = ["name", "type", "side", "notional", "fixed_rate_pct", "start", "tenor", "maturity"]
SHEETS = {
    "trades": TRADE_COLUMNS + ["last_fixing_pct", "netting_set"],
    "candidates": TRADE_COLUMNS,
    "curves": ["tenor", "estr_pct", "euribor6m_pct"],
    "credit": ["tenor", "counterparty_spread_bp", "bank_spread_bp"],
    "vols": ["expiry", "tenor", "vol_bp"],
    "stress": ["scenario", "rates_bp", "vol_mult", "counterparty_spread_mult", "bank_spread_mult"],
    "positions": ["name", "side", "notional", "maturity", "dv01", "netting_set"],
    "im_params": ["vertex", "risk_weight_bp"] + [f"rho_{v}" for v in LABELS],
}
TEXT = {"name", "type", "side", "start", "maturity", "netting_set", "scenario", "vertex"}
ALIASES = {
    "trade": "name", "id": "name", "trade_id": "name", "product": "type", "direction": "side", "nominal": "notional",
    "amount": "notional", "notional_eur": "notional", "k": "fixed_rate_pct", "strike": "fixed_rate_pct",
    "rate": "fixed_rate_pct", "fixed_rate": "fixed_rate_pct", "start_date": "start", "maturity_date": "maturity",
    "end": "maturity", "end_date": "maturity", "last_fixing": "last_fixing_pct", "estr": "estr_pct", "€str": "estr_pct",
    "euribor": "euribor6m_pct", "euribor6m": "euribor6m_pct", "euribor_6m": "euribor6m_pct",
    "counterparty_bp": "counterparty_spread_bp", "bank_bp": "bank_spread_bp", "vol": "vol_bp", "normal_vol_bp": "vol_bp",
    "pv01": "dv01", "bpv": "dv01",
}
SIDES = {"payer": "payer", "pay": "payer", "p": "payer", "buy": "payer",
         "receiver": "receiver", "receive": "receiver", "rec": "receiver", "r": "receiver", "sell": "receiver"}
TYPES = {"irs": "IRS", "swap": "IRS", "euribor": "IRS", "euribor6m": "IRS", "ois": "OIS", "estr": "OIS", "€str": "OIS"}
SECTORS = sorted({s for s, _ in RW_MAR50})
QUALITIES = ["IG", "HY"]
SETTINGS = {  # numeric settings: (low, high), both inclusive
    "threshold_counterparty": (0, 1e13), "threshold_bank": (0, 1e13), "mta": (0, 1e13),
    "independent_amount": (-1e13, 1e13), "mpor_days": (0, 250), "mean_reversion": (1e-4, 1),
    "sigma_override_pct": (0.01, 10), "counterparty_recovery_pct": (0, 99), "bank_recovery_pct": (0, 99),
    "funding_spread_bp": (0, 5000), "paths": (200, 200_000), "seed": (0, 2**31), "grid_step_months": (1, 12),
    "pfe_level_pct": (50, 99.9), "im_subcurve_corr": (0, 1), "im_threshold_meur_bp": (1e-3, 1e6), "hurdle_pct": (0, 100),
    "ccr_risk_weight_pct": (0, 1250),
}
README = [
    "XVA calculator input workbook. Keep the sheet and column names; delete rows or add your own. Sheets you remove keep their current values in the app.",
    "",
    "trades: the netting set. side is the bank's view (payer = the bank pays fixed). type IRS = EURIBOR 6M, OIS = compounded €STR.",
    "   start: years from the valuation date, or a date; negative or past = seasoned trade. Give tenor (years) or maturity (years or a date), not both.",
    "   last_fixing_pct: seasoned trades only, the running EURIBOR fixing (IRS) or the annualised €STR accrued so far (OIS). netting_set: optional.",
    "candidates: new trades for the pre-trade comparison, same columns. A blank fixed rate means the par rate.",
    "curves: continuous zero rates in % per tenor in years, linear between pillars, flat beyond.",
    "credit: CDS par spreads in bp per tenor in years, for the counterparty and the bank.",
    "vols: ATM normal vols of swaptions in bp, expiry and tenor in years (whole years for the tenor).",
    "stress: scenarios, rates_bp = parallel shift, the others are multipliers (1 = unchanged).",
    "positions: positions known only in part, for the initial margin only (never simulated). Each row needs a notional or a dv01 (euros per bp, positive).",
    "   side blank = direction unknown, which gives a range. maturity: years or a date. netting_set blank = every netting set.",
    "im_params: risk weights (bp) and tenor correlations of the synthetic initial margin. Our own calibration on ECB history, not ISDA parameters; a licensed user can paste theirs.",
    "settings: setting / value pairs. Amounts in euros, sigma_override_pct blank = calibrate to the vols sheet, valuation_date blank = today.",
]


class InputError(ValueError):
    def __init__(self, errors):
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("; ".join(self.errors))


def _blank(v) -> bool:
    return v is None or v is pd.NaT or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip())


def _flag(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "1.0", "yes", "y", "on")


def _number(v, amount: bool = False) -> float:
    """Finite float from a cell: 1 000 000, 1,000,000, 1.000.000,5, 2,5 and 2.5% all read.
    A lone comma before three digits is a thousands separator in an amount, a decimal comma elsewhere."""
    if isinstance(v, bool):
        raise ValueError(f"{v} is not a number")
    if isinstance(v, (int, float, np.number)):
        x = float(v)
    else:
        s = re.sub(r"[\s%']", "", str(v))
        comma, dot = s.rfind(","), s.rfind(".")
        if comma >= 0 and dot >= 0:  # the last separator is the decimal one
            s = s.replace(",", "") if dot > comma else s.replace(".", "").replace(",", ".")
        elif s.count(",") > 1 or (amount and re.fullmatch(r"-?\d{1,3},\d{3}", s)):
            s = s.replace(",", "")
        elif s.count(".") > 1:
            s = s.replace(".", "")
        else:
            s = s.replace(",", ".")
        x = float(s)
    if not np.isfinite(x):
        raise ValueError(f"{v} is not a finite number")
    return x


def _column(c) -> str:
    c = re.sub(r"\s+", "_", str(c).strip().lower())
    return ALIASES.get(c, c)


def _text(col: str, v):
    if _blank(v):
        return None
    if isinstance(v, (datetime, date)):
        return (v.date() if isinstance(v, datetime) else v).isoformat()
    if isinstance(v, (int, float, np.number)):
        return f"{v:.15g}"  # trade ids can be long integers
    v = str(v).strip()
    table = {"side": SIDES, "type": TYPES}.get(col)
    if table and v.lower() not in table:
        raise ValueError(f"{v!r} is not one of {', '.join(sorted(set(table.values())))}")
    return table[v.lower()] if table else v


def tidy(table: str, df: pd.DataFrame, errors: list, notes: list | None = None) -> pd.DataFrame:
    """Known columns in sheet order, aliases renamed, numbers as floats, text cleaned, blank rows dropped."""
    df = df.rename(columns=_column)
    if table == "stress":
        df = df.rename(columns={"name": "scenario"})
    columns = SHEETS[table]
    if notes is not None and (extra := [str(c) for c in df.columns if c not in columns and not str(c).startswith("unnamed")]):
        notes.append(f"{table}: column(s) ignored: {', '.join(extra)}")
    df = df.loc[:, ~df.columns.duplicated()].reindex(columns=columns)
    df = df.loc[[not all(_blank(v) for v in r) for r in df.itertuples(index=False)]].reset_index(drop=True)
    out = {}
    for col in columns:
        cells = []
        for i, v in enumerate(df[col]):
            try:
                cells.append(_text(col, v) if col in TEXT else (np.nan if _blank(v) else _number(v, col in ("notional", "dv01"))))
            except ValueError as e:
                errors.append(f"{table} row {i + 1}, {col}: {e if col in TEXT else repr(v) + ' is not a number'}")
                cells.append(None if col in TEXT else np.nan)
        out[col] = pd.Series(cells, dtype=object if col in TEXT else float)
    return pd.DataFrame(out, columns=columns)


# --- Defaults ---------------------------------------------------------------------------------------------------


def default_tables() -> dict:
    d = yaml.safe_load(Path(PORTFOLIO).read_text(encoding="utf-8"))
    spreads = yaml.safe_load((ROOT / "spreads.yaml").read_text(encoding="utf-8"))
    cp, bk = spreads["counterparty"]["spreads_bp"], spreads["bank"]["spreads_bp"]
    tenors = [1, 2, 3, 5, 7, 10, 15, 20, 30]
    tables = {
        "trades": [{"name": q["name"], "type": q["type"], "side": q["side"], "notional": q["nominal"],
                    "fixed_rate_pct": round(q["K"] * 100, 6), "start": q["start"], "tenor": q["tenor"]} for q in d["trades"]],
        "candidates": [{"name": "New payer 10Y", "type": "IRS", "side": "payer", "notional": 50e6, "tenor": 10},
                       {"name": "New receiver 5Y", "type": "IRS", "side": "receiver", "notional": 25e6, "tenor": 5}],
        "curves": [{"tenor": T, "estr_pct": 2.5, "euribor6m_pct": 2.7} for T in tenors],
        "credit": [{"tenor": T, "counterparty_spread_bp": cp[T], "bank_spread_bp": bk[T]} for T in cp],
        "vols": [{"expiry": e, "tenor": n, "vol_bp": round(v * 1e4, 6)} for e, n, v in swaption_vols()],
        "stress": [{"scenario": n, "rates_bp": r, "vol_mult": v, "counterparty_spread_mult": c, "bank_spread_mult": b}
                   for n, r, v, c, b in [("Rates +100 bp", 100, 1, 1, 1), ("Rates -100 bp", -100, 1, 1, 1),
                                         ("Vols +20%", 0, 1.2, 1, 1), ("Counterparty spreads x2", 0, 1, 2, 1),
                                         ("Combined downturn", -100, 1.2, 2, 2)]],
        # one row per data level, so the demo shows the waterfall; initial margin only, XVA does not see them
        "positions": [{"name": "Legacy payer, DV01 known", "side": "payer", "maturity": 7, "dv01": 15_000},
                      {"name": "Broker swap, notional and maturity", "notional": 20e6, "maturity": 4},
                      {"name": "Unknown ticket, notional only", "notional": 10e6}],
        "im_params": load_params(),
    }
    return {k: tidy(k, pd.DataFrame(v), []) for k, v in tables.items()}


def default_settings() -> dict:
    csa = yaml.safe_load(Path(PORTFOLIO).read_text(encoding="utf-8"))["csa"]
    s = yaml.safe_load((ROOT / "spreads.yaml").read_text(encoding="utf-8"))
    return {
        "csa_enabled": True, "threshold_counterparty": float(csa["H_C"]), "threshold_bank": float(csa["H_B"]),
        "mta": float(csa["MTA"]), "independent_amount": 0.0, "mpor_days": csa["mpor_days"],
        "mean_reversion": 0.03, "sigma_override_pct": None,
        "counterparty_recovery_pct": s["counterparty"]["R"] * 100, "bank_recovery_pct": s["bank"]["R"] * 100,
        "funding_spread_bp": float(s["bank"]["funding_bp"]), "sector": s["counterparty"]["sector"],
        "quality": s["counterparty"]["quality"], "paths": 10_000, "seed": 0, "grid_step_months": 1,
        "pfe_level_pct": 97.5, "valuation_date": None,
        "im_subcurve_corr": 0.98, "im_threshold_meur_bp": 100.0, "hurdle_pct": 10.0, "ccr_risk_weight_pct": 100.0,
    }


def preset_curves(preset: str, basis_bp: float = 20.0) -> tuple[str, pd.DataFrame]:
    """(description, curves table) for the "ecb" (AAA as an €STR proxy, EURIBOR = AAA + basis) or "09" preset."""
    if preset == "ecb":
        day, z = ecb_latest()
        df = pd.DataFrame({"tenor": z["tenor"], "estr_pct": z["rate"], "euribor6m_pct": z["rate"] + basis_bp / 100})
        return f"ECB AAA zero curve of {day} as an €STR proxy, EURIBOR 6M = AAA + {basis_bp:g} bp", tidy("curves", df, [])
    df = pd.read_csv(ROOT / "curves_09.csv")
    return f"Project 09 bootstrapped curves of {df['date'].iloc[0]} (reconstructed quotes)", tidy("curves", df, [])


# --- Files ------------------------------------------------------------------------------------------------------


def _kind(name: str, df: pd.DataFrame) -> str | None:
    low = Path(name).stem.lower()
    for table in ["candidates", "trades", "curves", "credit", "vols", "stress", "settings", "positions", "im_params"]:
        if table in low:
            return table
    cols = {_column(c) for c in df.columns}
    for table, mark in [("settings", "setting"), ("stress", "scenario"), ("vols", "vol_bp"), ("curves", "estr_pct"),
                        ("credit", "counterparty_spread_bp"), ("im_params", "risk_weight_bp"), ("trades", "fixed_rate_pct"),
                        ("positions", "dv01"), ("trades", "notional")]:
        if mark in cols:
            return table
    return None


def _settings_of(df: pd.DataFrame, errors: list) -> dict:
    df = df.rename(columns=_column)
    if not {"setting", "value"} <= set(df.columns):
        errors.append("settings: expected the columns setting and value")
        return {}
    known = default_settings()
    out = {}
    for k, v in zip(df["setting"], df["value"]):
        if _blank(k):
            continue
        k = str(k).strip().lower()
        if k not in known:
            errors.append(f"settings: unknown setting {k!r}")
        else:
            out[k] = None if _blank(v) else v
    return _clean_settings(out, errors)


def _clean_settings(s: dict, errors: list) -> dict:
    """The given settings as numbers and canonical text; each problem named in errors."""
    out = dict(s)
    for k, v in s.items():
        if k in SETTINGS:
            low, high = SETTINGS[k]
            if v is None and k == "sigma_override_pct":
                continue
            try:
                out[k] = _number(v)
                if not low <= out[k] <= high:
                    errors.append(f"settings, {k}: expected a number between {low:g} and {high:g}")
            except (ValueError, TypeError):
                errors.append(f"settings, {k}: {v!r} is not a number")
        elif k == "csa_enabled":
            out[k] = _flag(v)
        elif k == "sector":
            out[k] = str(v).strip().lower()
            if out[k] not in SECTORS:
                errors.append(f"settings, sector: expected one of {', '.join(SECTORS)}")
        elif k == "quality":
            out[k] = str(v).strip().upper()
            if out[k] not in QUALITIES:
                errors.append(f"settings, quality: expected {' or '.join(QUALITIES)}")
        elif k == "valuation_date" and v is not None:
            try:
                out[k] = pd.to_datetime(v, dayfirst=not re.match(r"\d{4}-", str(v))).date().isoformat()
            except (ValueError, TypeError):
                errors.append(f"settings, valuation_date: {v!r} is not a date")
                out[k] = None
    return out


def load_files(files) -> tuple[dict, dict, list]:
    """({table: DataFrame}, {setting: value}, notes) found in uploaded files [(name, bytes)]; all checked or InputError."""
    found, errors, notes = {}, [], []
    for name, data in files:
        try:
            if name.lower().endswith((".xlsx", ".xlsm")):
                book = _sheets(data)
            else:
                text = data.decode("utf-8-sig", errors="replace") if data[:3] == b"\xef\xbb\xbf" else _decode(data)
                book = {name: pd.read_csv(io.StringIO(text), sep=None, engine="python", dtype=object)}
        except Exception as e:  # any corrupt file: tell the user rather than crash the page
            errors.append(f"{name}: cannot read this file ({e})")
            continue
        for sheet, df in book.items():
            if sheet.lower() == "readme":
                continue
            kind = _kind(sheet, df)
            if kind:
                found[kind] = df
            elif len(book) > 1:
                notes.append(f"{name}: sheet {sheet!r} ignored, its name and columns match no table")
            else:
                errors.append(f"{name}: cannot tell what {sheet!r} holds, name it after a table ({', '.join(SHEETS)}, settings)")
    if not found and not errors:
        errors.append("no table found in these files")
    settings = _settings_of(found.pop("settings"), errors) if "settings" in found else {}
    tables = {k: tidy(k, df, errors, notes) for k, df in found.items()}
    if errors:
        raise InputError(errors)
    return tables, settings, notes


def _sheets(data: bytes) -> dict:
    """{sheet: DataFrame} with the cells as typed (pandas.read_excel turns a 1 into True next to a TRUE, as in settings)."""
    from openpyxl import load_workbook

    out = {}
    for ws in load_workbook(io.BytesIO(data), data_only=True).worksheets:
        rows = list(ws.values)
        if rows:
            out[ws.title] = pd.DataFrame(rows[1:], columns=[f"unnamed {j}" if c is None else c for j, c in enumerate(rows[0])])
    return out


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")  # Excel "CSV" on a European Windows


def excel(sheets: dict) -> bytes:
    """Workbook from {name: DataFrame}: bold header, frozen first row, readable widths and numbers."""
    from openpyxl.styles import Font

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name[:31], index=False, header=name != "README")
            ws = w.sheets[name[:31]]
            if name != "README":
                ws.freeze_panes = "A2"
            for j, col in enumerate(ws.columns, start=1):
                if name != "README":
                    col[0].font = Font(bold=True)
                    ws.column_dimensions[col[0].column_letter].width = min(44, max(len(str(c.value or "")) for c in col) + 2)
                for c in col[1:]:
                    if isinstance(c.value, float):
                        c.number_format = "#,##0.00##"
    return buf.getvalue()


def workbook(tables: dict, settings: dict, readme: bool = False) -> bytes:
    """Inputs in the format load_files reads back: the template and the export of the current inputs."""
    sheets = {"README": pd.DataFrame({"text": README})} if readme else {}
    # start and maturity are text here (they may be dates): real numbers in Excel where they are numbers
    numbers = lambda df: df.assign(**{c: pd.to_numeric(df[c], errors="coerce").fillna(df[c]) for c in ("start", "maturity") if c in df})
    sheets |= {k: numbers(tables[k]) for k in SHEETS}
    return excel(sheets | {"settings": pd.DataFrame({"setting": list(settings), "value": list(settings.values())})})


# --- Checks -----------------------------------------------------------------------------------------------------


def _rows(df: pd.DataFrame) -> list:
    return [(i + 1, {k: None if _blank(v) else v for k, v in r.items()}) for i, (_, r) in enumerate(df.iterrows())]


def _years(text: str, today: date) -> float:
    if not re.fullmatch(r"\d{1,4}[./-]\d{1,2}[./-]\d{1,4}", text):  # 15.01.2030 would read as a number
        try:
            return _number(text)
        except ValueError:
            pass
    d = date.fromisoformat(text) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text) else pd.to_datetime(text, dayfirst=True).date()
    return (d - today).days / 365


def _trades(table: str, df: pd.DataFrame, today: date, errors: list, warnings: list) -> list:
    out = []
    for i, r in _rows(df):
        err = len(errors)
        where = f"{table} row {i}" + (f" ({r['name']})" if r["name"] else "")

        def when(col, default=None):
            if r[col] is None:
                return default
            try:
                return _years(r[col], today)
            except ValueError:
                errors.append(f"{where}, {col}: {r[col]!r} is neither a number of years nor a date")

        for col in ["type", "side", "notional"]:
            if r[col] is None:
                errors.append(f"{where}, {col}: required")
        if r["notional"] is not None and not 0 < r["notional"] <= 1e13:
            errors.append(f"{where}, notional: must be positive (the direction goes in side) and at most 1e13")
        K = r["fixed_rate_pct"]
        if K is None and table == "trades":
            errors.append(f"{where}, fixed_rate_pct: required")
        if K is not None and abs(K) > 25:
            errors.append(f"{where}, fixed_rate_pct: {K:g} is in percent, 2.5 means 2.5%")
        elif K is not None and 0 < abs(K) < 0.2:
            warnings.append(f"{where}: fixed rate of {K:g}%, is it a decimal? 2.5 means 2.5%")
        start, T = when("start", 0.0), when("maturity")
        tenor = r["tenor"]
        if (tenor is None) == (r["maturity"] is None):
            errors.append(f"{where}: give tenor or maturity, not both and not neither")
        elif len(errors) == err and start is not None:
            T = start + tenor if tenor is not None else T
            if T - start <= 0:
                errors.append(f"{where}: ends before it starts")
            elif T <= 0:
                errors.append(f"{where}: already matured")
            elif T > 60:
                errors.append(f"{where}: maturity beyond 60 years")
        if len(errors) > err:
            continue
        fix = r.get("last_fixing_pct")
        if start < 0 and fix is None:
            warnings.append(f"{where}: seasoned without last_fixing_pct, today's rate is used for the running coupon")
        out.append({"name": r["name"] or f"{table[:-1]} {i}", "type": r["type"].lower(),
                    "side": 1 if r["side"] == "payer" else -1, "nominal": r["notional"],
                    "K": None if K is None else K / 100, "start": start, "tenor": T - start,
                    "last_fixing": None if fix is None else fix / 100})
    return out


def _axis(table: str, col: str, rows: list, low: float, errors: list) -> list:
    """Strictly positive pillars without duplicates, in the table's order of appearance."""
    v = [r[col] for _, r in rows]
    for (i, r), x in zip(rows, v):
        if x is None or x < low:
            errors.append(f"{table} row {i}, {col}: must be at least {low:g}")
    known = [x for x in v if x is not None]
    if len(set(known)) < len(known):
        errors.append(f"{table}: duplicate {col} pillars")
    return v


def _required(table: str, col: str, rows: list, low: float, high: float, errors: list) -> list:
    for i, r in rows:
        if r[col] is None or not low <= r[col] <= high:
            errors.append(f"{table} row {i}, {col}: expected a number between {low:g} and {high:g}")
    return [r[col] for _, r in rows]


def _positions(df: pd.DataFrame, today: date, errors: list) -> list:
    """Positions known only in part, for the initial margin: each row needs a notional or a DV01."""
    out = []
    for i, r in _rows(df):
        err = len(errors)
        where = f"positions row {i}" + (f" ({r['name']})" if r["name"] else "")
        if r["notional"] is None and r["dv01"] is None:
            errors.append(f"{where}: give a notional or a DV01")
        if r["notional"] is not None and not 0 < r["notional"] <= 1e13:
            errors.append(f"{where}, notional: must be positive (the direction goes in side) and at most 1e13")
        if r["dv01"] is not None and not 0 <= r["dv01"] <= 1e13:
            errors.append(f"{where}, dv01: must be positive, in euros per bp (the direction goes in side)")
        T = None
        if r["maturity"] is not None:
            try:
                T = _years(r["maturity"], today)
            except ValueError:
                errors.append(f"{where}, maturity: {r['maturity']!r} is neither a number of years nor a date")
            else:
                if T <= 0:
                    errors.append(f"{where}: already matured")
                elif T > 60:
                    errors.append(f"{where}: maturity beyond 60 years")
        if len(errors) == err:
            out.append({"name": r["name"] or f"position {i}", "side": {"payer": 1, "receiver": -1}.get(r["side"]),
                        "notional": r["notional"], "dv01": r["dv01"], "maturity": T})
    return out


def _im_params(df: pd.DataFrame, s: dict, errors: list) -> dict | None:
    """Parameters of the synthetic initial margin, labelled: our own calibration unless the table was changed."""
    rows, err = _rows(df), len(errors)
    if [str(r["vertex"]).lower() for _, r in rows] != LABELS:
        errors.append(f"im_params: expected one row per vertex, in this order: {', '.join(LABELS)}")
        return None
    rw = _required("im_params", "risk_weight_bp", rows, 1e-6, 1e5, errors)
    rho = [_required("im_params", f"rho_{v}", rows, 0, 1, errors) for v in LABELS]
    if len(errors) > err:
        return None
    rw, rho = np.array(rw, dtype=float), np.array(rho, dtype=float)
    if not np.allclose(rho, rho.T, atol=1e-6) or not np.allclose(np.diag(rho), 1.0):
        errors.append("im_params: the correlations must be symmetric with ones on the diagonal")
    elif np.linalg.eigvalsh(rho).min() < -1e-3:
        errors.append("im_params: the correlation matrix is not positive semi-definite")
    own = load_params()
    same = np.allclose(rw, own["risk_weight_bp"]) and np.allclose(rho, own[[f"rho_{v}" for v in LABELS]].to_numpy())
    return {"rw": rw.tolist(), "rho": rho.tolist(), "phi": s["im_subcurve_corr"], "threshold": s["im_threshold_meur_bp"] * 1e6,
            "source": SOURCE_OWN if same else SOURCE_USER}


def build_config(tables: dict, settings: dict, netting_set: str | None = None) -> dict:
    """One JSON-able configuration for pipeline.run, or InputError listing every problem at once."""
    errors, warnings = [], []
    t = {k: tidy(k, tables[k], errors) for k in SHEETS if k != "stress"}
    s = _clean_settings(default_settings() | settings, errors)
    today = date.fromisoformat(s["valuation_date"]) if s["valuation_date"] else date.today()

    df = t["trades"]
    if netting_set is not None:
        df = df[df["netting_set"].fillna("") == netting_set]
    trades = _trades("trades", df, today, errors, warnings)
    if not trades and not errors:
        errors.append("trades: the netting set is empty, add at least one trade")
    candidates = _trades("candidates", t["candidates"], today, errors, warnings)
    if len({q["name"] for q in trades}) < len(trades):
        warnings.append("trades: some names appear twice, the tables below will be harder to read")

    rows = _rows(t["curves"])
    if not rows:
        errors.append("curves: at least one pillar is needed")
    tenors = _axis("curves", "tenor", rows, 1e-3, errors)
    estr = _required("curves", "estr_pct", rows, -25, 25, errors)
    euribor = _required("curves", "euribor6m_pct", rows, -25, 25, errors)
    curves = {}
    if not errors and rows:
        order = np.argsort(tenors)
        x = np.array(tenors)[order]
        y = {"estr": np.array(estr)[order] / 100, "euribor6m": np.array(euribor)[order] / 100}
        curves = {"tenors": x.tolist()} | {k: v.tolist() for k, v in y.items()}
        horizon = max((q["start"] + q["tenor"] for q in trades + candidates), default=0.0)
        if horizon > x[-1]:
            warnings.append(f"curves: last pillar at {x[-1]:g}Y for trades running to {horizon:.1f}Y, zero rates held flat beyond")
        for k, v in y.items():
            if np.any(np.diff(v * x) < -1e-9):
                warnings.append(f"curves: {k} implies negative forward rates between some pillars")

    rows = _rows(t["credit"])
    if not rows:
        errors.append("credit: at least one pillar is needed")
    cds = _axis("credit", "tenor", rows, 0.25, errors)
    cp = _required("credit", "counterparty_spread_bp", rows, 0, 5000, errors)
    bk = _required("credit", "bank_spread_bp", rows, 0, 5000, errors)

    rows = _rows(t["vols"])
    if not rows and s["sigma_override_pct"] is None:
        errors.append("vols: give at least one swaption, or set sigma_override_pct")
    expiry = _required("vols", "expiry", rows, 1 / 12, 60, errors)
    length = _required("vols", "tenor", rows, 1, 60, errors)
    bp = _required("vols", "vol_bp", rows, 0.1, 1000, errors)
    for (i, _), n in zip(rows, length):
        if n is not None and n != round(n):
            errors.append(f"vols row {i}, tenor: whole years only")

    df = t["positions"]
    if netting_set is not None:
        df = df[df["netting_set"].fillna(netting_set) == netting_set]
    positions = _positions(df, today, errors)
    im = _im_params(t["im_params"], s, errors)

    if errors:
        raise InputError(errors)
    days = int(s["mpor_days"])
    csa = None
    if _flag(s["csa_enabled"]):
        csa = {"H_C": s["threshold_counterparty"], "H_B": s["threshold_bank"], "MTA": s["mta"],
               "IA": s["independent_amount"], "mpor_days": days, "mpor": round(days * 7 / 5 / 365, 12)}
    return {
        "trades": trades, "candidates": candidates, "curves": curves, "csa": csa,
        "credit": {"tenors": cds, "counterparty_bp": cp, "bank_bp": bk, "counterparty_R": s["counterparty_recovery_pct"] / 100,
                   "bank_R": s["bank_recovery_pct"] / 100, "funding": s["funding_spread_bp"] * 1e-4},
        "vols": [[e, n, v * 1e-4] for e, n, v in zip(expiry, length, bp)],
        "a": s["mean_reversion"], "sigma": None if s["sigma_override_pct"] is None else s["sigma_override_pct"] / 100,
        "paths": int(s["paths"]) // 2 * 2, "seed": int(s["seed"]), "step": int(s["grid_step_months"]) / 12,
        "pfe_level": s["pfe_level_pct"] / 100, "sector": str(s["sector"]).lower(), "quality": str(s["quality"]).upper(),
        "valuation_date": today.isoformat(), "warnings": warnings, "positions": positions, "im": im,
        "hurdle": s["hurdle_pct"] / 100, "ccr_rw": s["ccr_risk_weight_pct"] / 100,
    }


def parse_stress(df: pd.DataFrame) -> list:
    """Scenarios of the stress table, or InputError."""
    errors = []
    rows = _rows(tidy("stress", df, errors))
    out = []
    for i, r in rows:
        name = r["scenario"] or f"Scenario {i}"
        rates = 0.0 if r["rates_bp"] is None else r["rates_bp"]
        m = [1.0 if r[k] is None else r[k] for k in ("vol_mult", "counterparty_spread_mult", "bank_spread_mult")]
        if abs(rates) > 1000:
            errors.append(f"stress row {i}, rates_bp: expected between -1000 and 1000")
        if any(not 0 < x <= 20 for x in m):
            errors.append(f"stress row {i}: multipliers must be above 0 and at most 20")
        out.append({"name": name, "rates_bp": rates, "vol": m[0], "counterparty": m[1], "bank": m[2]})
    if errors:
        raise InputError(errors)
    return out
