"""Graphiques et tableaux de validation (étape 6) : python -m xva.rapport  ->  rapport/figures/*.png, rapport/*.csv"""
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .demo import PORTEFEUILLE
from .hw1f import calibrer_p, calibrer_q
from .market import historique_bce, marche_test, vols_swaptions
from .products import charger_portefeuille
from .validation import backtest_pit, convergence, etude_pq, historique_complet
from .xva import charger_credit

SORTIE = Path(__file__).resolve().parents[2] / "rapport"


def main():
    (SORTIE / "figures").mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    marche = marche_test()
    modele, _ = calibrer_q(marche.estr, marche.euribor6m, vols_swaptions())
    credit = charger_credit(P=marche.estr)
    trades, _ = charger_portefeuille(PORTEFEUILLE)

    t0 = time.perf_counter()
    conv, pente = convergence(trades, modele, marche, credit)
    conv.to_csv(SORTIE / "convergence.csv", index=False)
    print(f"Convergence (sans CSA), pente log ET / log N = {pente:.3f} ({time.perf_counter() - t0:.0f} s)\n"
          f"{conv.round(1).to_string(index=False)}")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].errorbar(conv["N"], conv["CVA"] / 1e3, yerr=1.96 * conv["ET"] / 1e3, fmt="o-", capsize=3)
    ax[0].set(xscale="log", xlabel="chemins N", ylabel="CVA (k€), IC 95 %", title="CVA du portefeuille démo")
    ax[1].loglog(conv["N"], conv["ET"], "o-", label=f"antithétiques (pente {pente:.2f})")
    ax[1].loglog(conv["N"], conv["ET sans antithétiques"], "s--", label="sans antithétiques")
    ax[1].loglog(conv["N"], conv["ET"].iloc[0] * np.sqrt(conv["N"].iloc[0] / conv["N"]), ":", c="grey",
                 label="1/√N")
    ax[1].set(xlabel="chemins N", ylabel="erreur-type (€)", title="Erreur-type")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(SORTIE / "figures" / "convergence.png", dpi=130)

    sigma_p = calibrer_p(historique_bce())
    pq = etude_pq(trades, marche, vols_swaptions(), credit, sigma_p)
    pq.to_csv(SORTIE / "etude_pq.csv")
    print(f"\nÉtude P/Q (sans CSA, 10 000 chemins, mêmes tirages)\n{pq.astype(float).round(4).to_string()}")
    fig, ax = plt.subplots(figsize=(8, 4))
    e = pq["écart"].astype(float).iloc[1:] * 100
    ax.barh(e.index, e, color=["#b2182b" if v < 0 else "#2166ac" for v in e])
    ax.axvline(0, c="k", lw=0.8)
    ax.set(xlabel="écart de CVA contre la référence Q (%)", title="CVA : mesure P, vols, retour à la moyenne, spreads")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(SORTIE / "figures" / "etude_pq.png", dpi=130)

    t0 = time.perf_counter()
    pits, ks = backtest_pit(historique_complet())
    ks.to_csv(SORTIE / "backtest_pit.csv", index=False)
    print(f"\nBacktest PIT sous P, historique BCE 2004-2026 ({time.perf_counter() - t0:.0f} s)\n"
          f"{ks.round(3).to_string(index=False)}")
    fig, ax = plt.subplots(len(pits), 3, figsize=(11, 7), sharex=True)
    for i, (h, df) in enumerate(pits.items()):
        for j, p in enumerate(df.columns):
            ax[i, j].hist(df[p], bins=10, range=(0, 1), density=True, color="#4c72b0")
            ax[i, j].axhline(1, c="k", ls=":", lw=0.8)
            ax[i, j].set_title(f"{p}, {h} (n = {len(df)})", fontsize=9)
    fig.suptitle("PIT des valeurs réalisées dans la loi simulée (uniforme si le modèle est juste)")
    fig.tight_layout()
    fig.savefig(SORTIE / "figures" / "backtest_pit.png", dpi=130)
    print(f"\nFigures dans {SORTIE / 'figures'}")


if __name__ == "__main__":
    main()
