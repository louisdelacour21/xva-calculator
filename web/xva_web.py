"""Same computation as app.py, for the static page (Pyodide in a Web Worker): start() once, compute(p) on each change."""
import numpy as np

from xva.demo import PORTFOLIO
from xva.exposure import exposure, grid
from xva.hw1f import calibrate_q, swap_rate
from xva.market import swaption_vols, sample_market
from xva.products import cube_mtm, load_portfolio, swap, value
from xva.xva import adjustments, allocation_euler, load_credit, pre_trade

MPORS = {5: 7 / 365, 10: 14 / 365, 20: 28 / 365}
N = 10_000
SHOWN_PATHS = 120  # drawn on the page's fan chart; the metrics always use all N
_s = {}


def start(n=N):
    market = sample_market()
    model, _ = calibrate_q(market.estr, market.euribor6m, swaption_vols())
    trades, c = load_portfolio(PORTFOLIO)
    base, t = grid(trades, mpors=tuple(MPORS.values()))
    sim = model.simulate(t, n, seed=0)
    cube = cube_mtm(trades, model, market, sim)
    V0s = value(trades, model, market, 0.0, 0.0, 1.0)[0]
    credit = load_credit(P=market.estr)
    alloc = allocation_euler(cube, sim["D"], t, base, V0s, credit)
    V, V0 = cube.sum(-1), float(V0s.sum())
    shown = np.random.default_rng(0).choice(n, min(SHOWN_PATHS, n), replace=False)  # random: antithetic pairs stay unbiased
    _s.update(market=market, model=model, credit=credit, base=base, sim=sim, V=V, V0=V0, shown=shown)
    e0 = exposure(V, sim["D"], sim["t"], base, V0, None)
    return {"N": n, "sigma": float(model.sigma), "V0": V0, "base": [float(b) for b in base],
            "atm": {T: float(swap_rate(0.0, T, market.estr, market.euribor6m)[0]) for T in range(1, 11)},
            "csa": {"threshold": c["H_C"] / 1e6, "mta": c["MTA"] / 1e6, "mpor": c["mpor_days"]},
            "EE0": e0["EE"].tolist(), "PFE0": e0["PFE"].tolist(),
            "alloc": [[tr.name, float(a)] for tr, a in zip(trades, alloc)]}


def compute(p):
    """p: with_csa, threshold, mta (M€), mpor (days), side, nominal (M€), tenor (years), K (decimal)."""
    s = _s
    csa = ({"H_C": p["threshold"] * 1e6, "H_B": p["threshold"] * 1e6, "MTA": p["mta"] * 1e6,
            "mpor": MPORS[p["mpor"]]} if p["with_csa"] else None)
    expo = exposure(s["V"], s["sim"]["D"], s["sim"]["t"], s["base"], s["V0"], csa)
    new = swap("NEW", "irs", 1 if p["side"] == "payer" else -1, p["nominal"] * 1e6, p["K"], 0.0, p["tenor"])
    pt = pre_trade(s["V"], s["V0"], new, s["model"], s["market"], s["sim"], s["base"], s["credit"], csa)
    return {"xva": adjustments(expo, s["credit"], s["V0"]), "delta": pt["Δ"], "V0_new": pt["V0"],
            "seconds": pt["seconds"], "EE": expo["EE"].tolist(), "PFE": expo["PFE"].tolist(),
            "EEPE": float(expo["EEPE"]), "paths": np.round(expo["E"][s["shown"]]).tolist()}


if __name__ == "__main__":
    import json
    d = start(2_000)
    r = compute({"with_csa": True, **d["csa"], "side": "payer", "nominal": 50, "tenor": 10, "K": d["atm"][10]})
    json.dumps([d, r])  # everything must go to the page as JSON
    assert len(r["EE"]) == len(d["base"]) and r["xva"]["CVA"][0] > 0
    assert len(r["paths"]) == SHOWN_PATHS and len(r["paths"][0]) == len(d["base"])
    print("ok", round(r["xva"]["CVA"][0] / 1e3, 1), "k€")
