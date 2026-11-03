"""Démo : XVA du netting set démo et coût d'un nouveau swap (pré-trade), mêmes chemins.

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

MPORS = {5: 7 / 365, 10: 14 / 365, 20: 28 / 365}  # jours ouvrés -> années calendaires, nœuds présents sur la grille
N = 10_000

st.set_page_config(page_title="Calculateur XVA", layout="wide")


@st.cache_resource
def moteur():
    """Calibration, simulation et cube du netting set, une fois par session serveur (environ 2 s)."""
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

fr = lambda s: s.replace(",", " ").replace(".", ",")  # 1,234.5 -> 1 234,5
k = lambda v: fr(f"{v / 1e3:,.1f}") + " k€"

st.title("Calculateur XVA")
st.caption(f"Swaps EUR sous Hull-White 1F. Netting set synthétique de 7 swaps face à une contrepartie fictive BBB ; "
           f"σ(Q) = {fr(f'{modele.sigma:.2%}')} calibré sur les swaptions ; {fr(f'{N:,}')} chemins antithétiques, "
           "graine fixe ; ET = erreur-type Monte Carlo.")

with st.sidebar:
    st.header("CSA")
    avec_csa = st.toggle("Collatéral (CSA)", True)
    seuil = st.slider("Seuil, chaque sens (M€)", 0.0, 30.0, c["H_C"] / 1e6, 0.5, disabled=not avec_csa)
    mta = st.slider("MTA (M€)", 0.0, 2.0, c["MTA"] / 1e6, 0.1, disabled=not avec_csa)
    mpor = st.select_slider("MPoR (jours ouvrés)", list(MPORS), c["mpor_jours"], disabled=not avec_csa)
    st.header("Nouveau swap")
    sens = st.radio("Sens (vu de la banque)", ["payeur", "receveur"], horizontal=True)
    nominal = st.slider("Nominal (M€)", 10, 200, 50, 10)
    tenor = st.slider("Maturité (ans)", 1, 10, 10)  # jusqu'à la dernière date de la grille du netting set
    atm = taux_swap(0.0, tenor, marche.estr, marche.euribor6m)[0]
    K = st.number_input("Taux fixe (%)", value=round(atm * 100, 3), step=0.05, format="%.3f") / 100
    st.caption(f"Taux au pair : {fr(f'{atm:.3%}')}")

csa = {"H_C": seuil * 1e6, "H_B": seuil * 1e6, "MTA": mta * 1e6, "mpor": MPORS[mpor]} if avec_csa else None
expo = exposition(V, sim["D"], sim["t"], base, V0, csa)
xva = ajustements(expo, credit, V0)
nouveau = swap("NOUVEAU", "irs", 1 if sens == "payeur" else -1, nominal * 1e6, K, 0.0, tenor)
pt = pre_trade(V, V0, nouveau, modele, marche, sim, base, credit, csa)

st.subheader(f"XVA du netting set ({'avec' if avec_csa else 'sans'} CSA), V0 = {fr(f'{V0 / 1e6:,.2f}')} M€")
cols = st.columns(5)
for col, (q, (v, e)) in zip(cols, xva.items()):
    col.metric(q, k(v))
    col.caption(f"ET {k(e)}")

st.subheader(f"Pré-trade : {sens} {nominal} M€, {tenor} ans à {fr(f'{K:.3%}')}")
cols = st.columns(5)
for col, (q, (v, e)) in zip(cols, pt["Δ"].items()):
    col.metric(f"Δ{q}", k(v))
    col.caption(f"ET {k(e)}")
cols[4].metric("Temps de calcul", fr(f"{pt['secondes']:.2f}") + " s")
cols[4].caption(f"MtM du nouveau swap {k(pt['V0'])}")
st.caption("ΔXVA calculé sur les chemins du netting set déjà simulés : seul le nouveau swap est réévalué. "
           "Convention : CVA et FCA sont des coûts pour la banque, DVA et FBA des gains.")

g, d = st.columns([3, 2])
with g:
    st.subheader("Profils d'exposition")
    e0 = exposition(V, sim["D"], sim["t"], base, V0, None)
    prof = pd.DataFrame({"EE sans CSA": e0["EE"], "PFE 97,5 % sans CSA": e0["PFE"]}, index=base)
    if avec_csa:
        prof["EE avec CSA"], prof["PFE 97,5 % avec CSA"] = expo["EE"], expo["PFE"]
    st.line_chart(prof / 1e6, x_label="années", y_label="M€")
    m = lambda v: fr(f"{v / 1e6:,.2f}") + " M€"
    st.caption(f"EEPE {m(expo['EEPE'])}, PFE maximale {m(expo['PFE'].max())} "
               "(dents de scie : chaque date de flux fait tomber la valeur).")
with d:
    st.subheader("Allocation d'Euler de la CVA")
    st.dataframe(pd.DataFrame({"trade": [tr.nom for tr in trades], "CVA (k€)": alloc / 1e3})
                 .sort_values("CVA (k€)", ascending=False).round(1), hide_index=True)
    st.caption("Sans collatéral ; la somme des contributions égale la CVA.")

if xva["CVA"][1] > 0.05 * xva["CVA"][0]:
    st.warning("Erreur-type de la CVA à " + fr(f"{xva['CVA'][1] / xva['CVA'][0]:.1%}") + " (> 5 %).")

with st.expander("Proxys et limites (à lire)", expanded=True):
    st.markdown(
        "- **Courbes plates** €STR et EURIBOR 6M en attendant les courbes bootstrapées du projet 09.\n"
        "- **Vols de swaptions synthétiques** (aucune source gratuite) : vols ±20 % → CVA +15,7 % / −15,5 %.\n"
        "- **Spreads CDS synthétiques** : spreads ±25 % → CVA +22,6 % / −23,5 %.\n"
        "- σ historique (mesure P) au lieu de σ implicite : CVA −23,5 %.\n"
        "- Modèle à un facteur : risque de pente quasi invisible (rejeté au backtest point-in-time).\n"
        "- MTA ajouté au seuil, flux pendant la MPoR ignorés. Détail dans le rapport de validation.")
