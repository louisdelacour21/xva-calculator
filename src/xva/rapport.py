"""Validation charts and tables (step 6) : python -m xva.rapport  ->  rapport/figures/*.png, rapport/*.csv"""
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .demo import PORTEFEUILLE
from .exposure import grille
from .hw1f import calibrer_p, calibrer_q, swap_coupons, taux_swap
from .market import historique_bce, marche_test, vols_swaptions
from .products import charger_portefeuille, cube_mtm, indice, swap
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
    ax[0].set(xscale="log", xlabel="paths N", ylabel="CVA (k€), 95% CI", title="Demo netting-set CVA")
    ax[1].loglog(conv["N"], conv["ET"], "o-", label=f"antithetic (slope {pente:.2f})")
    ax[1].loglog(conv["N"], conv["ET sans antithétiques"], "s--", label="without antithetics")
    ax[1].loglog(conv["N"], conv["ET"].iloc[0] * np.sqrt(conv["N"].iloc[0] / conv["N"]), ":", c="grey",
                 label="1/√N")
    ax[1].set(xlabel="paths N", ylabel="standard error (€)", title="Standard error")
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
    ax.set(xlabel="CVA change vs the Q reference (%)", title="CVA: P measure, vols, mean reversion, spreads")
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
    fig.suptitle("PIT of realised values in the simulated distribution (uniform if the model is right)")
    fig.tight_layout()
    fig.savefig(SORTIE / "figures" / "backtest_pit.png", dpi=130)
    ee_swaption(modele, marche)
    print(f"\nFigures dans {SORTIE / 'figures'}")


def ee_swaption(modele, marche, nominal=10e6, tenor=5, n=40_000):
    """EE* d'un payeur seul au pair contre la swaption payeuse fermée (Jamshidian) aux dates de coupon fixe."""
    K = taux_swap(0.0, tenor, marche.estr, marche.euribor6m)[0]
    tr = [swap("P", "irs", 1, nominal, K, 0.0, tenor)]
    _, t = grille(tr, mpors=())
    sim = modele.simuler(t, n, graine=0)
    ee = sim["D"] * np.maximum(cube_mtm(tr, modele, marche, sim)[..., 0], 0)
    u = np.arange(1, tenor)
    sw = []
    for v in u:
        n1, temps, coupons = swap_coupons(v, tenor - v, K, marche.estr, marche.euribor6m)  # nominal unitaire
        sw.append(nominal * modele.jamshidian(v, temps, coupons, n1))
    k = [indice(t, v) for v in u]
    ecart = (ee.mean(0)[k] - sw) / nominal * 1e4
    et = ee.std(0)[k] / np.sqrt(n) / nominal * 1e4
    pd.DataFrame({"t": u, "EE*": ee.mean(0)[k], "swaption": sw, "écart (bp)": ecart, "ET (bp)": et}) \
        .to_csv(SORTIE / "ee_swaption.csv", index=False)
    print(f"\nEE* contre swaption ({n:,} chemins) : écart max {np.abs(ecart).max():.2f} bp du nominal, "
          f"ET {et.max():.2f} bp")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, ee.mean(0) / 1e3, label="simulated EE* (Monte Carlo)")
    ax.plot(u, np.array(sw) / 1e3, "o", c="#b2182b", label="closed-form payer swaption (Jamshidian)")
    ax.set(xlabel="years", ylabel="k€", title=f"Single payer {nominal / 1e6:.0f} M€ {tenor}y at par, no CSA")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SORTIE / "figures" / "ee_swaption.png", dpi=130)


if __name__ == "__main__":
    main()
