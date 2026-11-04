"""XVA table of the demo netting set (with and without CSA), Euler allocation, pre-trade, sensitivities and capital:
python -m xva.demo"""
import time
from pathlib import Path

import yaml

from .exposure import exposure, grid
from .hw1f import calibrate_q
from .market import ROOT, swaption_vols, sample_market
from .products import cube_mtm, load_portfolio, swap, value
from .risk import RW_MAR50, ba_cva_reduced, cs01, effective_maturity, sa_ccr, scva, sensitivities
from .xva import adjustments, allocation_euler, load_credit, pre_trade

PORTFOLIO = Path(__file__).resolve().parents[2] / "portfolios" / "demo.yaml"


def main(n: int = 10_000, seed: int = 0):
    market = sample_market()
    model, _ = calibrate_q(market.estr, market.euribor6m, swaption_vols())
    credit = load_credit(P=market.estr)
    trades, c = load_portfolio(PORTFOLIO)
    csa = {"H_C": c["H_C"], "H_B": c["H_B"], "MTA": c["MTA"], "mpor": c["mpor_days"] * 14 / 10 / 365}
    base, t = grid(trades, mpors=(csa["mpor"],))
    t0 = time.perf_counter()
    sim = model.simulate(t, n, seed=seed)
    cube = cube_mtm(trades, model, market, sim)
    print(f"σ(Q) = {model.sigma:.4%}, {n} paths, {len(t)} dates, cube in {time.perf_counter() - t0:.1f} s")
    V0s = value(trades, model, market, 0.0, 0.0, 1.0)[0]
    V, V0 = cube.sum(-1), V0s.sum()
    print(f"V0 = {V0 / 1e6:,.2f} M€")
    for name, k in (("no CSA", None), ("CSA", csa)):
        r = adjustments(exposure(V, sim["D"], t, base, V0, k), credit, V0)
        print(f"\n{name}\n" + "\n".join(f"  {q:14s} {v / 1e3:10,.1f} k€  (SE {e / 1e3:,.1f})" for q, (v, e) in r.items()))
    alloc = allocation_euler(cube, sim["D"], t, base, V0s, credit)
    print("\nEuler allocation of the CVA (no CSA):")
    for tr, a in sorted(zip(trades, alloc), key=lambda z: -z[1]):
        print(f"  {tr.name:7s} {a / 1e3:10,.1f} k€")
    new = swap("NEW", "irs", 1, 50e6, 0.026, 0.0, 10)
    for name, k in (("no CSA", None), ("CSA", csa)):
        r = pre_trade(V, V0, new, model, market, sim, base, credit, k)
        d = "  ".join(f"Δ{q} {v / 1e3:,.1f} k€ (SE {e / 1e3:,.1f})" for q, (v, e) in r["Δ"].items())
        print(f"\nPre-trade {new.name} payer 50 M€ 10Y 2.60%, {name}: {d}  in {r['seconds']:.2f} s")

    t0 = time.perf_counter()
    sens = sensitivities(trades, market, swaption_vols(), credit, csa, n=n, seed=seed)
    sens |= cs01(exposure(V, sim["D"], t, base, V0, csa), credit, market.estr)
    print(f"\nSensitivities with CSA, € per +1 bp (vega: +1% relative of vols), same draws, "
          f"{time.perf_counter() - t0:.0f} s:")
    for shock, r in sens.items():
        print(f"  {shock:16s} " + "  ".join(f"{q} {v:9,.0f} ({e:,.0f})" for q, (v, e) in r.items()))

    q = yaml.safe_load((ROOT / "spreads.yaml").read_text(encoding="utf-8"))["counterparty"]
    csa_sa = {"H_C": c["H_C"], "H_B": c["H_B"], "MTA": c["MTA"], "mpor_days": c["mpor_days"]}
    M, rw = effective_maturity(trades), RW_MAR50[(q["sector"], q["quality"])]
    print(f"\nCapital (CRE52 / MAR50), M = {M:.2f} years, RW = {rw:.1%} ({q['sector']}, {q['quality']}):")
    for name, k in (("no CSA", None), ("CSA", csa_sa)):
        r = sa_ccr(trades, V0, k)
        K = ba_cva_reduced([scva(rw, [(M, r["EAD"])])])
        print(f"  {name:8s} add-on {r['add-on'] / 1e3:8,.1f} k€  RC {r['RC'] / 1e3:8,.1f} k€  "
              f"multiplier {r['multiplier']:.3f}  EAD {r['EAD'] / 1e6:6,.2f} M€  BA-CVA {K / 1e3:8,.1f} k€")


if __name__ == "__main__":
    main()
