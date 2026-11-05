"""XVA calculator: your trades, curves, credit and CSA, typed in the page or uploaded as CSV / Excel.

  streamlit run app.py"""
import json
import sys
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

SRC = str(Path(__file__).parent / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
# Streamlit Cloud pulls new commits without restarting Python: drop the engine if its files changed,
# otherwise the new app.py runs against the old modules still in memory (ImportError after a rename)
stamp = max(f.stat().st_mtime_ns for f in Path(SRC, "xva").glob("*.py"))
if getattr(sys.modules.get("xva"), "stamp", None) != stamp:
    for name in [m for m in sys.modules if m.split(".")[0] == "xva"]:
        del sys.modules[name]
    import xva
    xva.stamp = stamp
from xva.inputs import (QUALITIES, SECTORS, SETTINGS, InputError, build_config, default_settings,  # noqa: E402
                        default_tables, load_files, parse_stress, preset_curves, workbook)
from xva.pipeline import results_workbook, run, sensitivity_table, stress_table  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed per series: EE and PFE keep theirs when the CSA is off
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
CURVE_PRESETS = {"Flat demo curves": None, "ECB AAA curve, latest date": "ecb", "Project 09 bootstrapped curves": "09"}
col = st.column_config

st.set_page_config(page_title="XVA calculator", layout="wide")
st.markdown("<style>[data-testid='stMetricValue']{font-size:1.45rem}</style>", unsafe_allow_html=True)


@st.cache_data(max_entries=8, show_spinner=False)
def run_cached(config: str, stamp: int) -> dict:
    return run(json.loads(config))


@st.cache_data(max_entries=8, show_spinner=False)
def stress_cached(config: str, scenarios: str, stamp: int) -> pd.DataFrame:
    return stress_table(json.loads(config), json.loads(scenarios))


@st.cache_data(max_entries=8, show_spinner=False)
def sens_cached(config: str, n: int, stamp: int) -> pd.DataFrame:
    return sensitivity_table(json.loads(config), n)


def money(v: float) -> str:
    if abs(v) < 1e3:
        return f"{v:,.0f} €"
    return f"{v / 1e6:,.2f} M€" if abs(v) >= 1e6 else f"{v / 1e3:,.1f} k€"


def show_errors(errors: list, where=st, title=""):
    where.error(title + "\n\n" + "\n".join(f"- {e}" for e in errors[:30]) + (f"\n- and {len(errors) - 30} more" if len(errors) > 30 else ""))


def show(df: pd.DataFrame, decimals: int = 0):
    """Read-only table: amounts with the reader's separators, rates and bp with fixed decimals."""
    df, conf = df.copy(), {}
    for c in df.select_dtypes("number"):
        low = c.lower()
        if low.endswith("%"):
            conf[c] = col.NumberColumn(format="%.4f")
        elif "bp" in low or low in ("start", "maturity", "tenor", "expiry", "years"):
            conf[c] = col.NumberColumn(format="%.2f")
        else:
            df[c] = df[c].round(decimals)
            conf[c] = col.NumberColumn(format="localized")
    st.dataframe(df, hide_index=True, column_config=conf)


def typed(label, key, scale=1.0, integer=False, **kw):
    """A number typed by the user, kept in the settings in the unit of the sheets."""
    low, high = SETTINGS[key]
    v = ss.settings[key]
    v = None if v is None else v / scale
    if integer:
        low, high, v = int(low), int(high), int(v)
        kw.setdefault("step", 1)
    out = st.number_input(label, low / scale if not integer else low, high / scale if not integer else high, v,
                          key=f"{key}_{ss.version}", **kw)
    ss.settings[key] = None if out is None else round(out * scale, 6)


def editor(name: str, conf: dict) -> pd.DataFrame:
    # room for every row and the add-row line, up to a page of 20 rows: no hidden scrolling on a short table
    return st.data_editor(ss.tables[name], key=f"{name}_{ss.version}", num_rows="dynamic", hide_index=True,
                          column_config=conf, height=35 * (min(len(ss.tables[name]), 20) + 2) + 3)


def profile_chart(p: pd.DataFrame) -> alt.LayerChart:
    names = list(p.columns)
    # safe field names: "PFE 97.5%" would read as a nested field in Vega
    wide = (p / 1e6).set_axis([f"s{i}" for i in range(len(names))], axis=1).reset_index()
    long = wide.melt("years", var_name="s", value_name="v").assign(series=lambda d: d["s"].map(lambda c: names[int(c[1:])]))
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("years:Q", title="Years"), y=alt.Y("v:Q", title="M€"),
        color=alt.Color("series:N", scale=alt.Scale(domain=names, range=COLORS[:len(names)]),
                        legend=alt.Legend(orient="top", title=None, labelLimit=240)))
    near = alt.selection_point(nearest=True, on="pointerover", fields=["years"], empty=False)
    tips = [alt.Tooltip("years:Q", title="Years", format=".2f")]
    tips += [alt.Tooltip(f"s{i}:Q", title=f"{n} (M€)", format=",.2f") for i, n in enumerate(names)]
    rule = alt.Chart(wide).mark_rule(color="#8a8984").encode(
        x="years:Q", opacity=alt.condition(near, alt.value(1), alt.value(0)), tooltip=tips).add_params(near)
    return lines + rule


ss = st.session_state
if "tables" not in ss:
    ss.tables, ss.settings = default_tables(), default_settings()
    ss.setdefault("version", 0)
    ss.ran, ss.good, ss.autorun, ss.curve_source = None, None, True, "Flat demo curves"
    ss.stress, ss.sens = None, None

st.title("XVA calculator")
st.caption("CVA, DVA, FCA and FBA of a netting set of EUR swaps under Hull-White 1F, with your own trades, curves, credit "
           "and CSA. Type in the tabs or upload a CSV or Excel file, then press Run. The demo inputs are synthetic.")
banner = st.container()

with st.sidebar:
    top = st.container()
    st.header("Collateral (CSA)")
    ss.settings["csa_enabled"] = st.toggle("Netting set under a CSA", ss.settings["csa_enabled"], key=f"csa_enabled_{ss.version}")
    off = not ss.settings["csa_enabled"]
    # 4 decimals: "10,000" with 3 reads as ten thousand under a comma-decimal browser locale
    typed("Threshold, counterparty (M€)", "threshold_counterparty", 1e6, step=1.0, format="%.4f", disabled=off)
    typed("Threshold, bank (M€)", "threshold_bank", 1e6, step=1.0, format="%.4f", disabled=off)
    typed("Minimum transfer amount (M€)", "mta", 1e6, step=0.1, format="%.4f", disabled=off)
    typed("Independent amount (M€)", "independent_amount", 1e6, step=1.0, format="%.4f", disabled=off,
          help="Held by the bank; negative if posted. Treated as unsegregated cash.")
    typed("Margin period of risk (business days)", "mpor_days", integer=True, disabled=off)
    st.header("Simulation")
    typed("Paths", "paths", integer=True, step=1000, help="Antithetic pairs: an odd number is rounded down.")
    typed("Seed", "seed", integer=True)
    typed("Grid step (months)", "grid_step_months", integer=True, help="Cash-flow dates are always on the grid.")
    typed("PFE level (%)", "pfe_level_pct", step=0.5, format="%.1f")
    day = ss.settings["valuation_date"]
    ss.settings["valuation_date"] = st.date_input(
        "Valuation date", date.fromisoformat(day) if day else date.today(), key=f"valuation_date_{ss.version}",
        help="Converts the dates of the trades into years.").isoformat()

names = ["Results", "Trades", "Market data", "Pre-trade", "Sensitivities", "Capital", "Stress", "Method"]
tabs = dict(zip(names, st.tabs(names)))
current, load_preset = {}, None

trade_columns = {
    "name": col.TextColumn("name"), "type": col.SelectboxColumn("type", options=["IRS", "OIS"]),
    "side": col.SelectboxColumn("side", options=["payer", "receiver"], help="The bank's view: payer = the bank pays fixed."),
    "notional": col.NumberColumn("notional (EUR)", min_value=0.0, format="localized"),
    "fixed_rate_pct": col.NumberColumn("fixed rate (%)", format="%.4f"),
    "start": col.TextColumn("start", help="Years from the valuation date, or a date. Negative or past = seasoned trade."),
    "tenor": col.NumberColumn("tenor (years)", format="%.2f", help="Give tenor or maturity, not both."),
    "maturity": col.TextColumn("maturity", help="Years from the valuation date, or a date."),
    "last_fixing_pct": col.NumberColumn("last fixing (%)", format="%.4f", help="Seasoned trades: running EURIBOR fixing, or annualised €STR accrued so far."),
    "netting_set": col.TextColumn("netting set", help="Optional: the selector in the sidebar then picks one."),
}

with tabs["Trades"]:
    st.subheader("Netting set")
    st.caption("Add rows at the bottom, select a row and press Delete to remove it. The same columns are read from a file.")
    current["trades"] = editor("trades", trade_columns)
    trades_slot = st.container()

with tabs["Market data"]:
    left, right = st.columns(2)
    with left:
        st.subheader("Curves")
        a, b, c = st.columns([3, 2, 1], vertical_alignment="bottom")
        resets = ss.get("resets", 0)  # not ss.version: these two keep their choice after a load
        preset = a.selectbox("Curve preset", list(CURVE_PRESETS), key=f"curve_preset_{resets}")
        basis = b.number_input("EURIBOR basis for ECB (bp)", -500.0, 500.0, 20.0, 1.0, key=f"curve_basis_{resets}")
        if c.button("Load", key=f"load_{ss.version}"):
            load_preset = (preset, basis)
        st.caption(f"Last loaded: {ss.curve_source}. Continuous zero rates, linear between pillars, flat beyond.")
        current["curves"] = editor("curves", {
            "tenor": col.NumberColumn("tenor (years)", format="%.2f"),
            "estr_pct": col.NumberColumn("€STR (%)", format="%.4f"),
            "euribor6m_pct": col.NumberColumn("EURIBOR 6M (%)", format="%.4f")})
    with right:
        st.subheader("Credit")
        a, b, c = st.columns(3)
        with a:
            typed("Counterparty recovery (%)", "counterparty_recovery_pct", step=5.0, format="%.1f")
        with b:
            typed("Bank recovery (%)", "bank_recovery_pct", step=5.0, format="%.1f")
        with c:
            typed("Funding spread (bp)", "funding_spread_bp", step=5.0, format="%.1f")
        st.caption("CDS par spreads, bootstrapped into piecewise-constant default intensities.")
        current["credit"] = editor("credit", {
            "tenor": col.NumberColumn("tenor (years)", format="%.2f"),
            "counterparty_spread_bp": col.NumberColumn("counterparty (bp)", format="%.1f"),
            "bank_spread_bp": col.NumberColumn("bank (bp)", format="%.1f")})
    st.subheader("Swaption vols and model")
    left, right = st.columns(2)
    with left:
        a, b = st.columns(2)
        with a:
            typed("Mean reversion a", "mean_reversion", step=0.005, format="%.4f")
        with b:
            typed("σ override (%)", "sigma_override_pct", step=0.05, format="%.4f", placeholder="Calibrated to the vols",
                  help="Leave blank to calibrate σ to the swaptions.")
        current["vols"] = editor("vols", {
            "expiry": col.NumberColumn("expiry (years)", format="%.2f"),
            "tenor": col.NumberColumn("tenor (years)", format="%d"),
            "vol_bp": col.NumberColumn("ATM normal vol (bp)", format="%.2f")})
    fit_slot = right.container()

with tabs["Pre-trade"]:
    st.subheader("Candidate trades")
    st.caption("Same columns as the trades. Blank fixed rate = par rate. Each candidate is priced on the paths of the netting set.")
    current["candidates"] = editor("candidates", {k: v for k, v in trade_columns.items()
                                                  if k not in ("last_fixing_pct", "netting_set")})
    pretrade_slot = st.container()

with tabs["Stress"]:
    st.subheader("Scenarios")
    st.caption("rates = parallel shift of both curves. Multipliers: 1 = unchanged. Everything is revalued in full on the same draws.")
    current["stress"] = editor("stress", {
        "scenario": col.TextColumn("scenario", width="large"), "rates_bp": col.NumberColumn("rates (bp)", format="%.0f"),
        "vol_mult": col.NumberColumn("vols ×", format="%.2f"),
        "counterparty_spread_mult": col.NumberColumn("counterparty spreads ×", format="%.2f"),
        "bank_spread_mult": col.NumberColumn("bank spreads ×", format="%.2f")})
    stress_slot = st.container()

with tabs["Capital"]:
    st.subheader("SA-CCR and reduced BA-CVA")
    a, b, _ = st.columns([1, 1, 2])
    ss.settings["sector"] = a.selectbox("Counterparty sector", SECTORS, SECTORS.index(ss.settings["sector"]), key=f"sector_{ss.version}")
    ss.settings["quality"] = b.selectbox("Credit quality", QUALITIES, QUALITIES.index(ss.settings["quality"]), key=f"quality_{ss.version}")
    capital_slot = st.container()

# Every input is known from here on

books = sorted({str(v) for v in current["trades"]["netting_set"].dropna() if str(v).strip()})
with top:
    book = st.selectbox("Netting set", books, key=f"book_{ss.version}") if books else None

try:
    cfg, errors = build_config(current, ss.settings, book), []
except InputError as e:
    cfg, errors = None, e.errors
key = json.dumps(cfg, sort_keys=True) if cfg else None

with top:
    clicked = st.button("Run", type="primary", disabled=cfg is None, width="stretch")
    status = st.empty()
    uploads = st.file_uploader("Upload CSV or Excel", type=["csv", "txt", "xlsx", "xlsm"], accept_multiple_files=True,
                               key=f"files_{ss.get('upload_count', 0)}", help="Your trades alone, or a full workbook: only the tables found are replaced. "
                               "The template below shows the format.")
    for note in ss.get("notes", []):
        st.caption(note)
    if clicked:
        ss.upload_errors = []
    if errs := ss.get("upload_errors"):
        show_errors(errs)
    a, b = st.columns(2)
    a.download_button("Template", lambda: workbook(default_tables(), default_settings(), readme=True),
                      "xva_input_template.xlsx", XLSX, on_click="ignore", width="stretch")
    b.download_button("My inputs", lambda s=dict(ss.settings): workbook(current, s), "xva_inputs.xlsx", XLSX, on_click="ignore",
                      width="stretch")
    if st.button("Reset to the demo", width="stretch"):
        version, resets = ss.version + 1, ss.get("resets", 0) + 1
        ss.clear()
        ss.version, ss.resets = version, resets  # new widget keys: the browser would send its old edits back under the same ones
        st.rerun()

if uploads:
    try:
        tables, settings, notes = load_files([(f.name, f.getvalue()) for f in uploads])
    except InputError as e:
        ss.upload_errors, ss.notes = e.errors, []  # a refused file leaves every input as it was
    else:
        ss.tables, ss.upload_errors, ss.autorun = current | tables, [], True
        ss.settings |= settings
        if "curves" in tables:
            ss.curve_source = ", ".join(f.name for f in uploads)
        found =", ".join(f"{k} ({len(v)} rows)" for k, v in tables.items()) + (", settings" if settings else "")
        ss.notes = [f"Loaded {found} from {', '.join(f.name for f in uploads)}."] + notes
        ss.version += 1  # new widget keys: the editors and fields show the new values
    ss.upload_count = ss.get("upload_count", 0) + 1  # new key: the uploader empties
    st.rerun()

if load_preset:
    label, basis = load_preset
    try:
        source, table = preset_curves(CURVE_PRESETS[label], basis) if CURVE_PRESETS[label] else (label, default_tables()["curves"])
    except (OSError, ValueError) as e:
        banner.error(f"This preset cannot be loaded: {e}")
    else:
        ss.tables, ss.curve_source, ss.autorun = current | {"curves": table}, source, True
        ss.version += 1
        st.rerun()

if clicked or (ss.autorun and cfg):
    ss.ran, ss.autorun = key, False

if errors:
    show_errors(errors, banner, f"{len(errors)} input problem(s) to fix before the next run:")

r = None
if ss.ran:
    try:
        with banner, st.spinner(f"Simulating {json.loads(ss.ran)['paths']:,} paths..."):
            r = run_cached(ss.ran, stamp)
        ss.good = ss.ran
    except Exception as e:  # numbers only the engine can reject: say so, keep the page and the last good results
        banner.error(f"The calculation failed: {e}")
        ss.ran = ss.good
        if ss.ran:
            r = run_cached(ss.ran, stamp)

if r and key != ss.ran:
    banner.warning("Inputs changed: the results shown are for the previous inputs." + (" Press Run." if key else ""))
    status.caption("Inputs changed")
elif r:
    status.caption(f"Up to date, computed in {r['seconds']:.1f} s" if r["seconds"] else "Up to date")

stress_result, sens_result = None, None
fresh_stress = ss.stress if ss.stress and ss.stress[0] == ss.ran else None
fresh_sens = ss.sens if ss.sens and ss.sens[0] == ss.ran else None

with stress_slot:
    go = st.button("Run the stress test", type="primary", disabled=r is None)
    st.caption("Takes about one run per scenario. A funding spread stays as typed: when bank spreads rise above it, the funding "
               "cost net of the bank's own default risk falls to zero, hence small FCA and FBA in that scenario.")
    if go:
        try:
            fresh_stress = ss.stress = (ss.ran, json.dumps(parse_stress(current["stress"])))
        except InputError as e:
            show_errors(e.errors)
    if fresh_stress:
        try:
            with st.spinner("Revaluing each scenario..."):
                stress_result = stress_cached(*fresh_stress, stamp)
            show(stress_result)
        except Exception as e:
            st.error(f"The stress test failed: {e}")

with tabs["Sensitivities"]:
    st.subheader("Rate deltas and vega")
    a, b, _ = st.columns([1, 1, 2], vertical_alignment="bottom")
    n = a.number_input("Paths for this table", 200, 50_000, 4_000, 1_000, key=f"sens_paths_{ss.version}")
    if b.button("Compute", type="primary", disabled=r is None):
        fresh_sens = ss.sens = (ss.ran, int(n) // 2 * 2)
    st.caption("Central differences with common random numbers: +1 bp on each pillar and on all, +1% of the vols. "
               "About 14 full revaluations, on demand. Sensitivities in EUR, with their Monte Carlo standard errors.")
    if fresh_sens:
        try:
            with st.spinner("Bumping and revaluing..."):
                sens_result = sens_cached(*fresh_sens, stamp)
            pillars = sens_result[sens_result["bump"].str.match(r"delta \d")]
            pillars = pillars.assign(bump=pillars["bump"].str.removeprefix("delta "))
            st.altair_chart(alt.Chart(pillars).mark_bar(color=COLORS[0], size=48).encode(
                x=alt.X("bump:N", sort=None, title="Pillar", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Total XVA:Q", title="Total XVA per +1 bp (EUR)"),
                tooltip=[alt.Tooltip("bump:N", title="Pillar"), alt.Tooltip("Total XVA:Q", title="Total XVA (EUR)", format=",.1f")],
            ).properties(height=260), width="stretch")
            show(sens_result, decimals=1)
        except Exception as e:
            st.error(f"The sensitivities failed: {e}")
    if r:
        st.subheader("Credit spread sensitivity (CS01)")
        st.caption("+1 bp on the CDS spreads at fixed exposures, from the main run.")
        show(r["cs01"], decimals=1)

with tabs["Results"]:
    if r is None:
        st.info("No results yet: fix the inputs above, then press Run.")
    else:
        x = r["xva"]
        ci = lambda k: f"± {money(1.96 * x[k][1])} at 95%"
        cols = st.columns(3)
        cols[0].metric("Mark-to-market", money(r["V0"]), help="Value without any adjustment, closed form.")
        cols[1].metric("Total XVA", money(x["Total XVA"][0]), help="CVA − DVA + FCA − FBA: the cost to the bank.")
        cols[1].caption(ci("Total XVA"))
        cols[2].metric("Adjusted value", money(x["Adjusted value"][0]), help="Mark-to-market minus total XVA.")
        cols[2].caption(ci("Adjusted value"))
        for c, k in zip(st.columns(4), ["CVA", "DVA", "FCA", "FBA"]):
            c.metric(k, money(x[k][0]))
            c.caption(ci(k))
        p = r["profiles"]
        st.subheader("Exposure profiles")
        st.altair_chart(profile_chart(p), width="stretch")
        st.caption(f"EEPE {money(r['EEPE'])}, peak PFE {money(r['peak PFE'])}. Saw-tooth: each cash-flow date drops the value. "
                   f"σ = {r['sigma']:.3%}, a = {r['a']:.3f}, {r['paths']:,} antithetic paths, {r['dates']} dates, about {r['mb']:,.0f} MB.")
        with st.expander("Profile data"):
            show((p / 1e3).round(1).rename(columns=lambda c: f"{c} (k€)").reset_index())
        st.subheader("Summary")
        rows = [{"measure": "Mark-to-market", "value (EUR)": r["V0"], "standard error": 0.0, "95% low": r["V0"], "95% high": r["V0"]}]
        rows += [{"measure": k, "value (EUR)": v, "standard error": e, "95% low": v - 1.96 * e, "95% high": v + 1.96 * e} for k, (v, e) in x.items()]
        show(pd.DataFrame(rows))
        with st.expander(f"Data checks ({len(r['checks'])})", expanded=bool(r["checks"])):
            for check in r["checks"]:
                st.warning(check)
            if not r["checks"]:
                st.write("Nothing to report: standard error under 5% of the CVA, inputs within the usual ranges.")
        st.download_button("Download the results (Excel)", lambda: results_workbook(r, stress_result, sens_result),
                           "xva_results.xlsx", XLSX, on_click="ignore")

with trades_slot:
    if r is not None:
        st.subheader("Analytics per trade")
        st.caption("Standalone CVA: the trade alone, uncollateralised. Euler CVA: its share of the uncollateralised netting-set CVA, "
                   "contributions sum to the total. Leave-one-out: the CVA of the netting set, CSA included, minus the CVA without the trade. "
                   "Start and maturity in years from the valuation date.")
        show(r["trades"])
        st.caption(f"Netting benefit: {money(r['diversification'])} (sum of standalone CVAs minus uncollateralised netting-set CVA).")

with fit_slot:
    if r is not None and len(r["fit"]):
        st.caption(f"Fit of the last run, σ = {r['sigma']:.3%}: model minus market vols.")
        show(r["fit"])

with pretrade_slot:
    if r is not None and r["pretrade"] is not None:
        st.subheader("Comparison")
        show(r["pretrade"])
        st.caption("ΔXVA > 0 costs the bank. CVA and FCA are costs, DVA and FBA benefits. Charge: ΔXVA spread over the annuity of the "
                   "trade. All-in rate: the par rate with the charge added to what the client pays. Last row: the candidates booked together.")
    elif r is not None:
        st.caption("Add a candidate and press Run.")

with capital_slot:
    if r is not None:
        k = r["capital"]
        for c, name in zip(st.columns(4), ["EAD", "SCVA", "BA-CVA capital (reduced)", "Accounting CVA"]):
            c.metric(name, money(k[name]))
        show(pd.DataFrame({"measure": list(k), "value": [f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}" for v in map(float, k.values())]}))
        st.caption("Regulatory charge, not a price: SA-CCR uses no simulation and no market vol, so it does not move with the model. "
                   "Single hedging set in EUR (CRE52), reduced BA-CVA with one counterparty (MAR50).")

with tabs["Method"]:
    st.subheader("Files")
    st.markdown(
        "- **Excel**: one sheet per table, as in the template. Only the tables found are replaced.\n"
        "- **CSV**: one table per file, recognised by the file name (`trades.csv`, `curves.csv`...) or its columns. "
        "Semicolons, French decimals and thousands separators are read.\n"
        "- Tables: `trades`, `candidates`, `curves`, `credit`, `vols`, `stress`, `settings`. Problems are listed with their row and column.\n"
        "- Files are read in memory and nothing is written to disk. This public demo runs on a shared free server: "
        "for a confidential book, clone the repository and run `streamlit run app.py` on your machine "
        "(set `XVA_MAX_MB` to raise the memory limit).")
    st.subheader("Model")
    st.markdown(
        "- Hull-White one-factor under Q, exact simulation, σ calibrated to ATM swaptions. Multi-curve: discounting on €STR, "
        "forwards on EURIBOR 6M with the basis held deterministic.\n"
        "- Swaps priced in closed form on each path. Seasoned trades: the running coupon comes from the last fixing.\n"
        "- CVA, DVA, FCA, FBA on discounted expected exposure, survival bootstrapped from CDS, trapezoid per grid interval.\n"
        "- CSA: thresholds, MTA, independent amount and margin period of risk, without initial margin.")
    st.subheader("Proxies and limitations")
    st.markdown(
        "- The demo curves, swaption vols and CDS spreads are synthetic. The ECB preset uses the AAA government curve as an €STR proxy.\n"
        "- On the demo netting set: vols ±20% → CVA +15.7% / −15.5%; spreads ±25% → CVA +22.6% / −23.5%; "
        "historical σ (P measure) instead of implied: CVA −23.5%.\n"
        "- One-factor model: slope risk is nearly invisible (rejected in the point-in-time backtest).\n"
        "- MTA added to the threshold, cash flows during the margin period of risk ignored.\n"
        "- Dates in ACT/365F years, no business-day calendar. Details in the validation report of the repository.")
