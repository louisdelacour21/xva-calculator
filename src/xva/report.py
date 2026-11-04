"""Validation charts and tables (step 6): python -m xva.report  ->  report/figures/*.png, report/*.csv"""
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .demo import PORTFOLIO
from .exposure import grid
from .hw1f import calibrate_p, calibrate_q, swap_coupons, swap_rate
from .market import ecb_history, swaption_vols, sample_market
from .products import cube_mtm, index, load_portfolio, swap
from .validation import backtest_pit, convergence, full_history, pq_study
from .xva import load_credit

OUT = Path(__file__).resolve().parents[2] / "report"


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    market = sample_market()
    model, _ = calibrate_q(market.estr, market.euribor6m, swaption_vols())
    credit = load_credit(P=market.estr)
    trades, _ = load_portfolio(PORTFOLIO)

    t0 = time.perf_counter()
    conv, slope = convergence(trades, model, market, credit)
    conv.to_csv(OUT / "convergence.csv", index=False)
    print(f"Convergence (no CSA), slope log SE / log N = {slope:.3f} ({time.perf_counter() - t0:.0f} s)\n"
          f"{conv.round(1).to_string(index=False)}")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].errorbar(conv["N"], conv["CVA"] / 1e3, yerr=1.96 * conv["SE"] / 1e3, fmt="o-", capsize=3)
    ax[0].set(xscale="log", xlabel="paths N", ylabel="CVA (k€), 95% CI", title="Demo netting-set CVA")
    ax[1].loglog(conv["N"], conv["SE"], "o-", label=f"antithetic (slope {slope:.2f})")
    ax[1].loglog(conv["N"], conv["SE without antithetics"], "s--", label="without antithetics")
    ax[1].loglog(conv["N"], conv["SE"].iloc[0] * np.sqrt(conv["N"].iloc[0] / conv["N"]), ":", c="grey",
                 label="1/√N")
    ax[1].set(xlabel="paths N", ylabel="standard error (€)", title="Standard error")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "convergence.png", dpi=130)

    sigma_p = calibrate_p(ecb_history())
    pq = pq_study(trades, market, swaption_vols(), credit, sigma_p)
    pq.to_csv(OUT / "pq_study.csv")
    print(f"\nP/Q study (no CSA, 10,000 paths, same draws)\n{pq.astype(float).round(4).to_string()}")
    fig, ax = plt.subplots(figsize=(8, 4))
    e = pq["change"].astype(float).iloc[1:] * 100
    ax.barh(e.index, e, color=["#b2182b" if v < 0 else "#2166ac" for v in e])
    ax.axvline(0, c="k", lw=0.8)
    ax.set(xlabel="CVA change vs the Q reference (%)", title="CVA: P measure, vols, mean reversion, spreads")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "pq_study.png", dpi=130)

    t0 = time.perf_counter()
    pits, ks = backtest_pit(full_history())
    ks.to_csv(OUT / "backtest_pit.csv", index=False)
    print(f"\nPIT backtest under P, ECB history 2004-2026 ({time.perf_counter() - t0:.0f} s)\n"
          f"{ks.round(3).to_string(index=False)}")
    fig, ax = plt.subplots(len(pits), 3, figsize=(11, 7), sharex=True)
    for i, (h, df) in enumerate(pits.items()):
        for j, p in enumerate(df.columns):
            ax[i, j].hist(df[p], bins=10, range=(0, 1), density=True, color="#4c72b0")
            ax[i, j].axhline(1, c="k", ls=":", lw=0.8)
            ax[i, j].set_title(f"{p}, {h} (n = {len(df)})", fontsize=9)
    fig.suptitle("PIT of realised values in the simulated distribution (uniform if the model is right)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "backtest_pit.png", dpi=130)
    ee_swaption(model, market)
    print(f"\nFigures in {OUT / 'figures'}")


def ee_swaption(model, market, nominal=10e6, tenor=5, n=40_000):
    """EE* of a single payer at par against the closed-form payer swaption (Jamshidian) at the fixed coupon dates."""
    K = swap_rate(0.0, tenor, market.estr, market.euribor6m)[0]
    tr = [swap("P", "irs", 1, nominal, K, 0.0, tenor)]
    _, t = grid(tr, mpors=())
    sim = model.simulate(t, n, seed=0)
    ee = sim["D"] * np.maximum(cube_mtm(tr, model, market, sim)[..., 0], 0)
    u = np.arange(1, tenor)
    sw = []
    for v in u:
        n1, times, coupons = swap_coupons(v, tenor - v, K, market.estr, market.euribor6m)  # unit notional
        sw.append(nominal * model.jamshidian(v, times, coupons, n1))
    k = [index(t, v) for v in u]
    gap = (ee.mean(0)[k] - sw) / nominal * 1e4
    se = ee.std(0)[k] / np.sqrt(n) / nominal * 1e4
    pd.DataFrame({"t": u, "EE*": ee.mean(0)[k], "swaption": sw, "gap (bp)": gap, "SE (bp)": se}) \
        .to_csv(OUT / "ee_swaption.csv", index=False)
    print(f"\nEE* against swaption ({n:,} paths): max gap {np.abs(gap).max():.2f} bp of notional, "
          f"SE {se.max():.2f} bp")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, ee.mean(0) / 1e3, label="simulated EE* (Monte Carlo)")
    ax.plot(u, np.array(sw) / 1e3, "o", c="#b2182b", label="closed-form payer swaption (Jamshidian)")
    ax.set(xlabel="years", ylabel="k€", title=f"Single payer {nominal / 1e6:.0f} M€ {tenor}y at par, no CSA")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "ee_swaption.png", dpi=130)


if __name__ == "__main__":
    main()
