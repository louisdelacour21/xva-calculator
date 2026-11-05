"""One calculation from a checked configuration (inputs.build_config): plain numbers and tables, no engine object,
so results can be cached by the app and exported.
"""
import hashlib
import json
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .exposure import exposure, grid
from .hw1f import HW1F, calibrate_q, hw_normal_vol
from .initial_margin import backtest, data_levels, im_profile, ladder, synthetic_im
from .inputs import InputError, excel
from .market import Market, ZeroCurve
from .products import cube_mtm, par_rate, swap, value
from .risk import RW_MAR50, ba_cva_reduced, capital_profile, cs01, effective_maturity, rate_shock, sa_ccr, scva, sensitivities
from .validation import full_history, live_checks
from .xva import Credit, adjustments, allocation_euler, bootstrap, contributions, kva, mva, pre_trade

MAX_MB = float(os.environ.get("XVA_MAX_MB", 600))  # Streamlit Community Cloud gives about 1 GB per app
ADJUSTMENTS = ("CVA", "DVA", "FCA", "FBA")
IM_LABEL = "Synthetic SIMM-style initial margin, not ISDA SIMM®, no affiliation"


def _market(cfg) -> Market:
    c = cfg["curves"]
    return Market(ZeroCurve(c["tenors"], c["estr"]), ZeroCurve(c["tenors"], c["euribor6m"]))


def _credit(cfg, P, counterparty=1.0, bank=1.0) -> Credit:
    c = cfg["credit"]

    def survival(who, spreads, R, mult):
        try:
            return bootstrap({T: s * mult * 1e-4 for T, s in zip(c["tenors"], spreads)}, P, R)
        except ValueError as e:
            raise InputError(f"credit, {who} spreads: {e}") from None

    return Credit(survival("counterparty", c["counterparty_bp"], c["counterparty_R"], counterparty),
                  survival("bank", c["bank_bp"], c["bank_R"], bank), c["funding"])


def _model(market, quotes, a, sigma) -> HW1F:
    try:
        return HW1F(a, sigma, market.estr) if sigma else calibrate_q(market.estr, market.euribor6m, quotes, a)[0]
    except ValueError as e:
        raise InputError(f"model: the calibration failed on these curves and vols ({e})") from None


def _trade(q, model, market):
    t = swap(q["name"], q["type"], q["side"], q["nominal"], q["K"] or 0.0, q["start"], q["tenor"], q["last_fixing"])
    return replace(t, K=par_rate(t, model, market)[0]) if q["K"] is None else t


def _memory(paths: int, dates: int, trades: int) -> float:
    """Arrays of paths × dates held at once, in MB: the cube, then about 8 more (draws, netting-set value, exposures)."""
    mb = paths * dates * (trades + 8) * 8 / 1e6
    if mb > MAX_MB:
        most = int(MAX_MB * 1e6 / (dates * (trades + 8) * 8))
        raise InputError(f"this run needs about {mb:,.0f} MB ({paths:,} paths × {dates} dates × {trades} trades), the limit "
                         f"here is {MAX_MB:,.0f} MB: use at most {most:,} paths or a longer grid step")
    return mb


def _setup(cfg) -> dict:
    """Everything shared by the run and the stress test: same grid and same seed, hence the same draws."""
    market = _market(cfg)
    quotes = [tuple(q) for q in cfg["vols"]]
    model = _model(market, quotes, cfg["a"], cfg["sigma"])
    trades = [_trade(q, model, market) for q in cfg["trades"]]
    candidates = [_trade(q, model, market) for q in cfg["candidates"]]
    csa = cfg["csa"]
    base, t = grid(trades + candidates, mpors=(csa["mpor"],) if csa else (), step=cfg["step"])
    return {"market": market, "quotes": quotes, "model": model, "trades": trades, "candidates": candidates,
            "csa": csa, "base": base, "t": t, "mb": _memory(cfg["paths"], len(t), len(trades + candidates))}


def _estimates(adj: dict, V0: float) -> dict:
    """Total XVA cost next to the four adjustments; it has the standard error of the adjusted value."""
    a, se = adj["Adjusted value"]
    return {k: adj[k] for k in ADJUSTMENTS} | {"Total XVA": (V0 - a, se), "Adjusted value": (a, se)}


def _pretrade(ctx, V, V0, sim, credit) -> pd.DataFrame:
    rows = []
    args = (ctx["model"], ctx["market"], sim, ctx["base"], credit, ctx["csa"])
    for c in ctx["candidates"] + ([ctx["candidates"]] if len(ctx["candidates"]) > 1 else []):
        pt = pre_trade(V, V0, c, *args)
        d = {k: pt["Δ"][k][0] for k in ADJUSTMENTS}
        total = d["CVA"] - d["DVA"] + d["FCA"] - d["FBA"]
        row = {"trade": "All together", "side": "", "notional": np.nan, "fixed rate %": np.nan, "par rate %": np.nan}
        running = np.nan
        all_in = np.nan
        if not isinstance(c, list):
            par, annuity = par_rate(c, ctx["model"], ctx["market"])
            running = total / (annuity * c.nominal)
            all_in = par - c.side * running  # the bank pays less, or receives more, to cover its cost
            row = {"trade": c.name, "side": "payer" if c.side > 0 else "receiver", "notional": c.nominal,
                   "fixed rate %": c.K * 100, "par rate %": par * 100}
        rows.append(row | {"MtM": pt["V0"]} | {f"Δ{k}": v for k, v in d.items()}
                    | {"ΔXVA": total, "ΔXVA SE": pt["total"][1], "charge (bp running)": running * 1e4,
                       "all-in rate %": all_in * 100})
    return pd.DataFrame(rows)


def _trade_table(ctx, cube, sim, V, V0s, credit, cva) -> tuple[pd.DataFrame, float]:
    t, base, csa = ctx["t"], ctx["base"], ctx["csa"]
    euler = allocation_euler(cube, sim["D"], t, base, V0s, credit)
    alone, without = [], []
    for j in range(cube.shape[2]):
        alone.append(contributions(exposure(cube[:, :, j], sim["D"], t, base, V0s[j]), credit)["CVA"].mean())
        rest = exposure(V - cube[:, :, j], sim["D"], t, base, V0s.sum() - V0s[j], csa)
        without.append(cva - contributions(rest, credit)["CVA"].mean())
    table = pd.DataFrame({
        "trade": [q.name for q in ctx["trades"]], "type": [q.type.upper() for q in ctx["trades"]],
        "side": ["payer" if q.side > 0 else "receiver" for q in ctx["trades"]],
        "notional": [q.nominal for q in ctx["trades"]], "fixed rate %": [q.K * 100 for q in ctx["trades"]],
        "start": [q.start for q in ctx["trades"]], "maturity": [q.maturity for q in ctx["trades"]],
        "MtM": V0s, "standalone CVA": alone, "Euler CVA": euler, "leave-one-out ΔCVA": without,
    })
    return table, float(np.sum(alone) - np.sum(euler))


def _margin(ctx, cfg, V, V0s, sim, credit, without) -> dict:
    """Synthetic initial margin, the funding cost of posting it (MVA), and the XVA if it were held and posted both ways."""
    market, trades, base, positions, p = ctx["market"], ctx["trades"], ctx["base"], cfg["positions"], cfg["im"]
    im = synthetic_im(trades, positions, market, p)
    profile = im_profile(trades, positions, market, p, base)
    charge = {k: mva(profile[k].to_numpy(), im[k], base, credit, market.estr) for k in ("central", "low", "high")}
    V0 = float(V0s.sum())
    margined = adjustments(exposure(V, sim["D"], ctx["t"], base, V0, ctx["csa"], cfg["pfe_level"], im=profile["central"].to_numpy()), credit, V0)
    rows = [{"adjustment": k, "without IM": without[k][0], "with IM (central)": margined[k][0]} for k in ADJUSTMENTS]
    rows.append({"adjustment": "MVA", "without IM": 0.0, "with IM (central)": charge["central"]})
    what_if = pd.DataFrame(rows)
    sign = np.array([1, -1, 1, -1, 1])
    what_if.loc[len(what_if)] = ["Total", sign @ what_if["without IM"], sign @ what_if["with IM (central)"]]
    what_if["change"] = what_if["with IM (central)"] - what_if["without IM"]
    return im | {"levels": data_levels(trades, market, p, V0s), "profile": profile, "mva": charge, "what-if": what_if,
                 "backtest": backtest(ladder(trades, market), p, full_history()), "source": p["source"], "phi": p["phi"],
                 "threshold": p["threshold"]}


def run_id(cfg: dict) -> str:
    """Same inputs (seed included), same ID."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def engine_hash() -> str:
    """Of the engine's source files, so it also works where git is absent; line endings do not count."""
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def _audit(cfg, r) -> dict:
    return {"run ID": run_id(cfg), "engine": engine_hash(), "computed (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "paths": cfg["paths"], "seed": cfg["seed"], "MtM": r["V0"], "Total XVA": r["xva"]["Total XVA"][0],
            "XVA incl. MVA and KVA": full_stack(r), "seconds": round(r["seconds"], 1)}


def run(cfg: dict) -> dict:
    t0 = time.perf_counter()
    ctx = _setup(cfg)
    market, model, trades, csa, base, t = (ctx[k] for k in ("market", "model", "trades", "csa", "base", "t"))
    credit = _credit(cfg, market.estr)
    sim = model.simulate(t, cfg["paths"], seed=cfg["seed"])
    cube = cube_mtm(trades, model, market, sim)
    V0s = value(trades, model, market, 0.0, 0.0, 1.0)[0]
    V, V0 = cube.sum(-1), float(V0s.sum())
    level = cfg["pfe_level"]
    expo = exposure(V, sim["D"], t, base, V0, csa, level)
    xva = _estimates(adjustments(expo, credit, V0), V0)
    bare = exposure(V, sim["D"], t, base, V0, None, level) if csa else expo
    pfe = f"PFE {level:.1%}"
    profiles = pd.DataFrame({"EE without CSA": bare["EE"], f"{pfe} without CSA": bare["PFE"]}, index=pd.Index(base, name="years"))
    if csa:
        profiles["EE with CSA"], profiles[f"{pfe} with CSA"] = expo["EE"], expo["PFE"]
    table, diversification = _trade_table(ctx, cube, sim, V, V0s, credit, xva["CVA"][0])

    fit = pd.DataFrame([{"expiry": e, "tenor": n, "market (bp)": v * 1e4,
                         "model (bp)": hw_normal_vol(model, e, n, market.euribor6m) * 1e4} for e, n, v in ctx["quotes"]])
    if len(fit):
        fit["error (bp)"] = fit["model (bp)"] - fit["market (bp)"]

    sensi = cs01(expo, credit, market.estr)
    cs = pd.DataFrame([{"bump": k} | {q: sensi[k][q][0] for q in ADJUSTMENTS} for k in sensi])

    ccr = sa_ccr(trades, V0, csa)
    M = effective_maturity(trades)
    rw = RW_MAR50[(cfg["sector"], cfg["quality"])]
    charge = scva(rw, [(M, ccr["EAD"])])
    ba = ba_cva_reduced([charge])
    ccr_capital = 0.08 * cfg["ccr_rw"] * ccr["EAD"]
    capital = {"Add-on": ccr["add-on"], "Replacement cost": ccr["RC"], "Multiplier": ccr["multiplier"], "EAD": ccr["EAD"],
               "Unmargined EAD (cap)": ccr["unmargined EAD"], "Effective maturity (years)": M, "BA-CVA risk weight": rw,
               "SCVA": charge, "BA-CVA capital (reduced)": ba, "CCR risk weight": cfg["ccr_rw"],
               "CCR capital (8% × risk weight × EAD)": ccr_capital, "Total capital": ccr_capital + ba, "Accounting CVA": xva["CVA"][0]}
    dates = np.linspace(0.0, base[-1], 25)
    K = capital_profile(trades, np.interp(dates, np.r_[0.0, t], np.r_[V0, V.mean(0)]), csa, rw, cfg["ccr_rw"], dates)
    run_off = pd.DataFrame({"Capital": K}, index=pd.Index(dates, name="years"))
    margin = _margin(ctx, cfg, V, V0s, sim, credit, xva)

    checks = list(cfg["warnings"])
    cva, se = xva["CVA"]
    if cva > 0 and se > 0.05 * cva:
        checks.append(f"CVA standard error at {se / cva:.1%} of the CVA, above 5%: raise the number of paths")
    if len(fit) and not cfg["sigma"] and np.abs(fit["error (bp)"]).max() > 5:
        checks.append(f"calibration misses a swaption by {np.abs(fit['error (bp)']).max():.1f} bp: one σ cannot fit these vols")
    validation = live_checks(trades, market, sim, V, credit, bare, float(table["Euler CVA"].sum()), fit, cfg["sigma"], xva["CVA"],
                             cfg["im"]["rho"], margin["backtest"])
    r = {
        "V0": V0, "xva": xva, "profiles": profiles, "EEPE": expo["EEPE"], "peak PFE": float(expo["PFE"].max()),
        "trades": table, "diversification": diversification, "pretrade": _pretrade(ctx, V, V0, sim, credit) if ctx["candidates"] else None,
        "cs01": cs, "capital": capital, "fit": fit, "sigma": model.sigma, "a": model.a, "checks": checks,
        "dates": len(t), "mb": ctx["mb"], "paths": cfg["paths"], "pfe_level": level, "csa": csa is not None,
        "im": margin, "kva": {"value": kva(K, dates[1:], credit, market.estr, cfg["hurdle"]), "hurdle": cfg["hurdle"], "profile": run_off},
        "validation": validation, "seconds": time.perf_counter() - t0,
    }
    return r | {"audit": _audit(cfg, r)}


def sensitivity_table(cfg: dict, n: int) -> pd.DataFrame:
    """Deltas per pillar and vega at n paths, with standard errors. Slow: on demand only."""
    market = _market(cfg)
    quotes = [tuple(q) for q in cfg["vols"]]
    model = _model(market, quotes, cfg["a"], cfg["sigma"])
    trades = [_trade(q, model, market) for q in cfg["trades"]]
    _memory(n, len(grid(trades, mpors=(cfg["csa"]["mpor"],) if cfg["csa"] else (), step=cfg["step"])[1]), len(trades))
    out = sensitivities(trades, market, quotes, _credit(cfg, market.estr), cfg["csa"], n=n, seed=cfg["seed"],
                        a=cfg["a"], sigma=cfg["sigma"], step=cfg["step"])
    rows = []
    for bump, d in out.items():
        row = {"bump": bump}
        for q in ADJUSTMENTS:
            row[q], row[f"{q} SE"] = d[q]
        row["Total XVA"] = row["CVA"] - row["DVA"] + row["FCA"] - row["FBA"]
        rows.append(row)
    return pd.DataFrame(rows)


def stress_table(cfg: dict, scenarios: list) -> pd.DataFrame:
    """Full revaluation per scenario on the draws of the run: curves, σ, survivals, exposures all move."""
    ctx = _setup(cfg)
    rows = []
    for sc in [{"name": "Base", "rates_bp": 0.0, "vol": 1.0, "counterparty": 1.0, "bank": 1.0}] + scenarios:
        market = rate_shock(ctx["market"], np.ones_like, sc["rates_bp"] * 1e-4)
        quotes = [(e, n, v * sc["vol"]) for e, n, v in ctx["quotes"]]
        try:
            model = _model(market, quotes, cfg["a"], cfg["sigma"] * sc["vol"] if cfg["sigma"] else None)
            credit = _credit(cfg, market.estr, sc["counterparty"], sc["bank"])
        except InputError as e:
            raise InputError(f"stress, {sc['name']}: {e}") from None
        sim = model.simulate(ctx["t"], cfg["paths"], seed=cfg["seed"])
        V = cube_mtm(ctx["trades"], model, market, sim).sum(-1)
        V0 = float(value(ctx["trades"], model, market, 0.0, 0.0, 1.0)[0].sum())
        xva = _estimates(adjustments(exposure(V, sim["D"], ctx["t"], ctx["base"], V0, ctx["csa"]), credit, V0), V0)
        rows.append({"scenario": sc["name"], "MtM": V0} | {k: v[0] for k, v in xva.items()})
    df = pd.DataFrame(rows)
    df["change in total XVA"] = df["Total XVA"] - df["Total XVA"].iloc[0]
    return df


def full_stack(r: dict) -> float:
    """Total XVA with the funding cost of the initial margin and the cost of capital."""
    return r["xva"]["Total XVA"][0] + r["im"]["mva"]["central"] + r["kva"]["value"]


def headline(r: dict) -> dict:
    return ({"Mark-to-market": r["V0"]} | {k: v[0] for k, v in r["xva"].items()}
            | {"MVA (central)": r["im"]["mva"]["central"], "KVA": r["kva"]["value"], "XVA including MVA and KVA": full_stack(r),
               "Initial margin (central)": r["im"]["central"], "EEPE": r["EEPE"], "Peak PFE": r["peak PFE"], "Total capital": r["capital"]["Total capital"]})


def baseline_table(base: dict, r: dict) -> pd.DataFrame:
    df = pd.DataFrame({"baseline": headline(base), "current": headline(r)}).rename_axis("measure").reset_index()
    df["change"] = (df["current"] - df["baseline"]).round(0) + 0.0  # whole euros as shown; + 0.0: no "-0" from a few cents of noise
    df["change %"] = df["change"] / df["baseline"].abs().where(df["baseline"] != 0) * 100
    return df


def config_diff(a, b, path: str = "") -> list[str]:
    """One line per input that differs, "path: old → new"."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [line for k in sorted(a.keys() | b.keys()) if k != "warnings"
                for line in config_diff(a.get(k), b.get(k), f"{path}.{k}" if path else k)]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [line for i, (u, v) in enumerate(zip(a, b)) for line in config_diff(u, v, f"{path}[{i}]")]
    size = lambda v: f"{len(v)} items" if isinstance(v, list) else v
    return [] if a == b else [f"{path}: {size(a)} → {size(b)}"]


def results_workbook(r: dict, stress: pd.DataFrame | None = None, sens: pd.DataFrame | None = None, log: list | None = None) -> bytes:
    """Everything the app shows, as an Excel file."""
    summary = [{"measure": "Mark-to-market", "value": r["V0"], "standard error": 0.0}]
    summary += [{"measure": k, "value": v, "standard error": e} for k, (v, e) in r["xva"].items()]
    im = r["im"]
    summary += [{"measure": "MVA (synthetic IM, central)", "value": im["mva"]["central"]}, {"measure": "KVA", "value": r["kva"]["value"]},
                {"measure": "XVA including MVA and KVA", "value": full_stack(r)}]
    summary += [{"measure": "EEPE", "value": r["EEPE"]}, {"measure": f"Peak PFE {r['pfe_level']:.1%}", "value": r["peak PFE"]},
                {"measure": "σ (Hull-White)", "value": r["sigma"]}, {"measure": "Mean reversion", "value": r["a"]},
                {"measure": "Paths", "value": r["paths"]}]
    margin = [("Label", IM_LABEL), ("Parameters", im["source"]), ("Central", im["central"]), ("Low", im["low"]), ("High", im["high"]),
              ("MVA, central", im["mva"]["central"]), ("MVA, low", im["mva"]["low"]), ("MVA, high", im["mva"]["high"]),
              ("Backtest exceedance, sample", im["backtest"]["sample"]), ("Backtest exceedance, full history", im["backtest"]["history"])]
    sheets = {"Summary": pd.DataFrame(summary), "Profiles": r["profiles"].reset_index(), "Trades": r["trades"],
              "Pre-trade": r["pretrade"], "CS01": r["cs01"], "Sensitivities": sens, "Stress": stress,
              "Capital": pd.DataFrame({"measure": list(r["capital"]), "value": [float(v) for v in r["capital"].values()]}),
              "Initial margin": pd.DataFrame(margin, columns=["measure", "value"]), "IM ladder": im["ladder"], "IM positions": im["positions"],
              "IM coverage": im["coverage"], "IM data levels": im["levels"], "IM what-if": im["what-if"], "IM profile": im["profile"].reset_index(),
              "KVA profile": r["kva"]["profile"].reset_index(), "Model fit": r["fit"], "Checks": pd.DataFrame({"check": r["checks"]}),
              "Controls": r["validation"], "Run log": pd.DataFrame(log or [r["audit"]])}
    return excel({k: v for k, v in sheets.items() if v is not None and len(v)})
