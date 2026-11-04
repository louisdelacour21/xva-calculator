"""Demo: XVA of the demo netting set and cost of a new swap (pre-trade), same paths.

  streamlit run app.py"""
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from xva.demo import PORTEFEUILLE  # noqa: E402
from xva.exposure import exposition, grille  # noqa: E402
from xva.hw1f import calibrer_q, taux_swap  # noqa: E402
from xva.market import marche_test, vols_swaptions  # noqa: E402
from xva.products import charger_portefeuille, cube_mtm, swap, valeur  # noqa: E402
from xva.xva import ajustements, allocation_euler, charger_credit, pre_trade  # noqa: E402

MPORS = {5: 7 / 365, 10: 14 / 365, 20: 28 / 365}  # business days -> calendar years, nodes on the grid
N = 10_000

st.set_page_config(page_title="XVA calculator", layout="wide")


@st.cache_resource
def moteur():
    """Calibration, simulation and netting-set cube, once per server session."""
    marche = marche_test()
    modele, _ = calibrer_q(marche.estr, marche.euribor6m, vols_swaptions())
    trades, c = charger_portefeuille(PORTEFEUILLE)
    base, t = grille(trades, mpors=tuple(MPORS.values()))
    sim = modele.simuler(t, N, graine=0)
    cube = cube_mtm(trades, modele, marche, sim)
    V0s = valeur(trades, modele, marche, 0.0, 0.0, 1.0)[0]
    credit = charger_credit(P=marche.estr)
    alloc = allocation_euler(cube, sim["D"], t, base, V0s, credit)
    return marche, modele, credit, trades, c, base, sim, cube, V0s, alloc


marche, modele, credit, trades, c, base, sim, cube, V0s, alloc = moteur()
V, V0 = cube.sum(-1), V0s.sum()

k = lambda v: f"{v / 1e3:,.1f} k€"
NOMS = {"V ajustée": "Adjusted value"}

st.title("XVA calculator")
st.caption(f"EUR swaps under Hull-White 1F. Synthetic netting set of 7 swaps against a fictitious BBB counterparty; "
           f"σ(Q) = {modele.sigma:.2%} calibrated to swaptions; {N:,} antithetic paths, "
           "fixed seed; SE = Monte Carlo standard error.")

with st.sidebar:
    st.header("CSA")
    avec_csa = st.toggle("Collateral (CSA)", True)
    seuil = st.slider("Threshold, each way (M€)", 0.0, 30.0, c["H_C"] / 1e6, 0.5, disabled=not avec_csa)
    mta = st.slider("MTA (M€)", 0.0, 2.0, c["MTA"] / 1e6, 0.1, disabled=not avec_csa)
    mpor = st.select_slider("MPoR (business days)", list(MPORS), c["mpor_jours"], disabled=not avec_csa)
    st.header("New swap")
    sens = st.radio("Direction (bank view)", ["payer", "receiver"], horizontal=True)
    nominal = st.slider("Notional (M€)", 10, 200, 50, 10)
    tenor = st.slider("Maturity (years)", 1, 10, 10)  # up to the last date of the netting-set grid
    atm = taux_swap(0.0, tenor, marche.estr, marche.euribor6m)[0]
    K = st.number_input("Fixed rate (%)", value=round(atm * 100, 3), step=0.05, format="%.3f") / 100
    st.caption(f"Par rate: {atm:.3%}")

csa = {"H_C": seuil * 1e6, "H_B": seuil * 1e6, "MTA": mta * 1e6, "mpor": MPORS[mpor]} if avec_csa else None
expo = exposition(V, sim["D"], sim["t"], base, V0, csa)
xva = ajustements(expo, credit, V0)
nouveau = swap("NOUVEAU", "irs", 1 if sens == "payer" else -1, nominal * 1e6, K, 0.0, tenor)
pt = pre_trade(V, V0, nouveau, modele, marche, sim, base, credit, csa)

st.subheader(f"Netting-set XVA ({'with' if avec_csa else 'without'} CSA), V0 = {V0 / 1e6:,.2f} M€")
cols = st.columns(5)
for col, (q, (v, e)) in zip(cols, xva.items()):
    col.metric(NOMS.get(q, q), k(v))
    col.caption(f"SE {k(e)}")

st.subheader(f"Pre-trade: {sens} {nominal} M€, {tenor} years at {K:.3%}")
cols = st.columns(5)
for col, (q, (v, e)) in zip(cols, pt["Δ"].items()):
    col.metric(f"Δ{q}", k(v))
    col.caption(f"SE {k(e)}")
cols[4].metric("Compute time", f"{pt['secondes']:.2f} s")
cols[4].caption(f"New swap MtM {k(pt['V0'])}")
st.caption("ΔXVA computed on the netting-set paths already simulated: only the new swap is revalued. "
           "Convention: CVA and FCA are costs to the bank, DVA and FBA are benefits.")

g, d = st.columns([3, 2])
with g:
    st.subheader("Exposure profiles")
    e0 = exposition(V, sim["D"], sim["t"], base, V0, None)
    prof = pd.DataFrame({"EE without CSA": e0["EE"], "PFE 97.5% without CSA": e0["PFE"]}, index=base)
    if avec_csa:
        prof["EE with CSA"], prof["PFE 97.5% with CSA"] = expo["EE"], expo["PFE"]
    st.line_chart(prof / 1e6, x_label="years", y_label="M€")
    m = lambda v: f"{v / 1e6:,.2f} M€"
    st.caption(f"EEPE {m(expo['EEPE'])}, peak PFE {m(expo['PFE'].max())} "
               "(saw-tooth: each cash-flow date drops the value).")
with d:
    st.subheader("Euler allocation of CVA")
    st.dataframe(pd.DataFrame({"trade": [tr.nom for tr in trades], "CVA (k€)": alloc / 1e3})
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
