"""Demo: XVA of the demo netting set and cost of a new swap (pre-trade), same paths.

  streamlit run app.py"""
import sys
from pathlib import Path

import numpy as np
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
from xva.demo import PORTFOLIO  # noqa: E402
from xva.exposure import exposure, grid  # noqa: E402
from xva.hw1f import calibrate_q, swap_rate  # noqa: E402
from xva.market import swaption_vols, sample_market  # noqa: E402
from xva.products import cube_mtm, load_portfolio, swap, value  # noqa: E402
from xva.xva import adjustments, allocation_euler, load_credit, pre_trade  # noqa: E402

MPORS = {5: 7 / 365, 10: 14 / 365, 20: 28 / 365}  # business days -> calendar years, nodes on the grid
N = 10_000

st.set_page_config(page_title="XVA calculator", layout="wide")


@st.cache_resource
def engine():
    """Calibration, simulation and netting-set cube, once per server session."""
    market = sample_market()
    model, _ = calibrate_q(market.estr, market.euribor6m, swaption_vols())
    trades, c = load_portfolio(PORTFOLIO)
    base, t = grid(trades, mpors=tuple(MPORS.values()))
    sim = model.simulate(t, N, seed=0)
    cube = cube_mtm(trades, model, market, sim)
    V0s = value(trades, model, market, 0.0, 0.0, 1.0)[0]
    credit = load_credit(P=market.estr)
    alloc = allocation_euler(cube, sim["D"], t, base, V0s, credit)
    return market, model, credit, trades, c, base, sim, cube, V0s, alloc


market, model, credit, trades, c, base, sim, cube, V0s, alloc = engine()
V, V0 = cube.sum(-1), V0s.sum()

k = lambda v: f"{v / 1e3:,.1f} k€"

st.title("XVA calculator")
st.caption(f"EUR swaps under Hull-White 1F. Synthetic netting set of 7 swaps against a fictitious BBB counterparty; "
           f"σ(Q) = {model.sigma:.2%} calibrated to swaptions; {N:,} antithetic paths, "
           "fixed seed; SE = Monte Carlo standard error.")

with st.sidebar:
    st.header("CSA")
    with_csa = st.toggle("Collateral (CSA)", True)
    threshold = st.slider("Threshold, each way (M€)", 0.0, 30.0, c["H_C"] / 1e6, 0.5, disabled=not with_csa)
    mta = st.slider("MTA (M€)", 0.0, 2.0, c["MTA"] / 1e6, 0.1, disabled=not with_csa)
    mpor = st.select_slider("MPoR (business days)", list(MPORS), c["mpor_days"], disabled=not with_csa)
    st.header("New swap")
    side = st.radio("Direction (bank view)", ["payer", "receiver"], horizontal=True)
    nominal = st.slider("Notional (M€)", 10, 200, 50, 10)
    tenor = st.slider("Maturity (years)", 1, 10, 10)  # up to the last date of the netting-set grid
    atm = swap_rate(0.0, tenor, market.estr, market.euribor6m)[0]
    K = st.number_input("Fixed rate (%)", value=round(atm * 100, 3), step=0.05, format="%.3f") / 100
    st.caption(f"Par rate: {atm:.3%}")

csa = {"H_C": threshold * 1e6, "H_B": threshold * 1e6, "MTA": mta * 1e6, "mpor": MPORS[mpor]} if with_csa else None
expo = exposure(V, sim["D"], sim["t"], base, V0, csa)
xva = adjustments(expo, credit, V0)
new = swap("NEW", "irs", 1 if side == "payer" else -1, nominal * 1e6, K, 0.0, tenor)
pt = pre_trade(V, V0, new, model, market, sim, base, credit, csa)

st.subheader(f"Netting-set XVA ({'with' if with_csa else 'without'} CSA), V0 = {V0 / 1e6:,.2f} M€")
cols = st.columns(5)
for col, (q, (v, e)) in zip(cols, xva.items()):
    col.metric(q, k(v))
    col.caption(f"SE {k(e)}")

st.subheader(f"Pre-trade: {side} {nominal} M€, {tenor} years at {K:.3%}")
cols = st.columns(5)
for col, (q, (v, e)) in zip(cols, pt["Δ"].items()):
    col.metric(f"Δ{q}", k(v))
    col.caption(f"SE {k(e)}")
cols[4].metric("Compute time", f"{pt['seconds']:.2f} s")
cols[4].caption(f"New swap MtM {k(pt['V0'])}")
st.caption("ΔXVA computed on the netting-set paths already simulated: only the new swap is revalued. "
           "Convention: CVA and FCA are costs to the bank, DVA and FBA are benefits.")

g, d = st.columns([3, 2])
with g:
    st.subheader("Exposure profiles")
    e0 = exposure(V, sim["D"], sim["t"], base, V0, None)
    prof = pd.DataFrame({"EE without CSA": e0["EE"], "PFE 97.5% without CSA": e0["PFE"]}, index=base)
    if with_csa:
        prof["EE with CSA"], prof["PFE 97.5% with CSA"] = expo["EE"], expo["PFE"]
    st.line_chart(prof / 1e6, x_label="years", y_label="M€")
    m = lambda v: f"{v / 1e6:,.2f} M€"
    st.caption(f"EEPE {m(expo['EEPE'])}, peak PFE {m(expo['PFE'].max())} "
               "(saw-tooth: each cash-flow date drops the value).")
with d:
    st.subheader("Euler allocation of CVA")
    st.dataframe(pd.DataFrame({"trade": [tr.name for tr in trades], "CVA (k€)": alloc / 1e3})
                 .sort_values("CVA (k€)", ascending=False).round(1), hide_index=True)
    st.caption("Uncollateralised; contributions sum to the CVA.")

if xva["CVA"][1] > 0.05 * xva["CVA"][0]:
    st.warning(f"CVA standard error at {xva['CVA'][1] / xva['CVA'][0]:.1%} (> 5%).")

with st.expander("Proxies and limitations (read this)", expanded=True):
    st.markdown(
        "- **Flat curves** for €STR and EURIBOR 6M, until the bootstrapped curves of project 09 are plugged in.\n"
        "- **Synthetic swaption vols** (no free source): vols ±20% → CVA +15.7% / −15.5%.\n"
        "- **Synthetic CDS spreads**: spreads ±25% → CVA +22.6% / −23.5%.\n"
        "- Historical σ (P measure) instead of implied σ: CVA −23.5%.\n"
        "- One-factor model: slope risk is nearly invisible (rejected in the point-in-time backtest).\n"
        "- MTA added to the threshold, cash flows during the MPoR ignored. Details in the validation report.")
