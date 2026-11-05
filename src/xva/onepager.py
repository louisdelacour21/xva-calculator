"""One-page run report: a self-contained HTML file to print to PDF, nothing fetched and no dependency.
Light colours on purpose: it is meant for paper, whatever the theme of the app.
"""
from html import escape

import numpy as np

from .pipeline import IM_LABEL, headline

LIMITS = ("Demo inputs are synthetic unless replaced. Not an accounting or regulatory figure. One-factor rates model: slope risk is nearly invisible. "
          "Basel figures are simplified (one hedging set, reduced BA-CVA). No business-day calendar. MTA added to the threshold, cash flows during the margin period ignored. "
          f"{IM_LABEL}: delta only, own parameters calibrated on public ECB rates.")
COLORS = ["#2a78d6", "#eb6834"]
PAGE = """@page{size:A4;margin:11mm}
body{margin:0;font:9.5px/1.35 Arial,Helvetica,sans-serif;color:#0b0b0b;background:#fff}
main{max-width:720px;margin:auto}
h1{font-size:16px;margin:0}h2{font-size:10.5px;margin:9px 0 3px;border-bottom:1px solid #0b0b0b;padding-bottom:1px}
p{margin:2px 0;color:#52514e}
table{border-collapse:collapse;width:100%}td,th{padding:1px 4px 1px 0;border-bottom:1px solid #e4e2dd;text-align:left;font-weight:400;vertical-align:top}
th{color:#52514e}td.n{text-align:right;font-variant-numeric:tabular-nums}tr.b td{font-weight:700}
.two{display:grid;grid-template-columns:1fr 1fr;gap:0 18px}
.key i{display:inline-block;width:14px;height:2px;margin:0 4px 3px 10px;vertical-align:middle}.key i:first-child{margin-left:0}
svg{display:block;width:100%;height:auto}svg text{font:9px Arial,Helvetica,sans-serif;fill:#52514e}svg text.d{fill:#0b0b0b}"""


def _eur(v) -> str:
    return f"{v:,.0f}"


def _table(rows, numeric=(), head=(), bold=()) -> str:
    """Rows of cells; every cell is escaped, names and warnings come from the user's file."""
    cell = lambda j, c: f"<td{' class=n' if j in numeric else ''}>{escape(str(c))}</td>"
    top = f"<tr>{''.join(f'<th>{escape(h)}</th>' for h in head)}</tr>" if head else ""
    body = "".join(f"<tr{' class=b' if row[0] in bold else ''}>{''.join(cell(j, c) for j, c in enumerate(row))}</tr>" for row in rows)
    return f"<table>{top}{body}</table>"


def _ceiling(v: float) -> tuple[float, float]:
    """(top of the axis, gridline step), the step being 2, 5, 10 or 20 × 10^(k-1)."""
    v = max(v, 1e-6)
    step = next(s * 10 ** np.floor(np.log10(v)) for s in (0.2, 0.5, 1, 2) if v / (s * 10 ** np.floor(np.log10(v))) <= 6)
    return float(np.ceil(v / step) * step), float(step)


def _chart(profiles, csa: bool) -> str:
    cols = [c for c in profiles.columns if c.endswith(" with CSA" if csa else " without CSA")]
    W, H, left, top, bottom = 720, 170, 34, 12, 18
    x_max = float(profiles.index[-1])
    y_max, step = _ceiling(float(profiles[cols].to_numpy().max()) / 1e6)
    X = lambda t: left + (W - left - 4) * t / x_max
    Y = lambda v: H - bottom - (H - top - bottom) * v / y_max
    parts = []
    for v in np.arange(0, y_max + step / 2, step):
        parts.append(f'<line x1="{left}" x2="{W - 4}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#e4e2dd"/><text x="{left - 4}" y="{Y(v) + 3:.1f}" text-anchor="end">{v:g}</text>')
    for v in range(0, int(x_max) + 1, max(int(x_max) // 10, 1)):
        parts.append(f'<text x="{X(v):.1f}" y="{H - 5}" text-anchor="middle">{v}</text>')
    for c, color in zip(cols, COLORS):
        y = profiles[c].to_numpy() / 1e6
        points = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in zip(profiles.index, y))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/>')
        peak = int(np.argmax(y))
        parts.append(f'<text class="d" x="{min(X(profiles.index[peak]), W - 60):.1f}" y="{Y(y[peak]) - 3:.1f}" text-anchor="middle">{escape(c.split()[0])} {y[peak]:.2f} M€</text>')
    key = "".join(f'<i style="background:{color}"></i>{escape(c)}' for c, color in zip(cols, COLORS))
    alt = "Exposure over time in million euros: " + ", ".join(f"{c} peaks at {profiles[c].max() / 1e6:.2f}" for c in cols)
    return f'<p class="key">{key} &nbsp;(M€, against years)</p><svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(alt)}">{"".join(parts)}</svg>'


def one_pager(r: dict, cfg: dict) -> str:
    a, im, k = r["audit"], r["im"], r["capital"]
    total = {"Total XVA", "XVA including MVA and KVA"}
    head = [(n, _eur(v)) for n, v in headline(r).items() if n not in ("Initial margin (central)", "EEPE", "Peak PFE", "Total capital")]
    head += [("EEPE", _eur(r["EEPE"])), (f"Peak PFE {r['pfe_level']:.1%}", _eur(r["peak PFE"]))]
    capital = [(n, _eur(k[n])) for n in ("EAD", "CCR capital (8% × risk weight × EAD)", "BA-CVA capital (reduced)", "Total capital")]
    margin = [("Central", _eur(im["central"])), ("Low to high", f"{_eur(im['low'])} to {_eur(im['high'])}"),
              ("Backtest exceedance, sample", f"{im['backtest']['sample']:.2%} of {im['backtest']['sample days']:,} days")]
    margin += [(f"Covered by: {row['data level']}", f"{row['share']:.1%}") for row in im["coverage"].to_dict("records")]
    csa = cfg["csa"]
    terms = (f"thresholds {_eur(csa['H_C'])} / {_eur(csa['H_B'])}, MTA {_eur(csa['MTA'])}, independent amount {_eur(csa['IA'])}, margin period {csa['mpor_days']} days"
             if csa else "none")
    c = cfg["credit"]
    assumptions = [("Trades", f"{len(cfg['trades'])} in the netting set"), ("Model", f"Hull-White 1F, a = {r['a']:.3f}, σ = {r['sigma']:.3%}"),
                   ("Simulation", f"{r['paths']:,} antithetic paths, {r['dates']} dates, seed {cfg['seed']}"), ("CSA", terms),
                   ("Credit", f"recoveries {c['counterparty_R']:.0%} / {c['bank_R']:.0%}, funding spread {c['funding'] * 1e4:,.0f} bp"),
                   ("Hurdle rate", f"{r['kva']['hurdle']:.1%}")]
    checks = "".join(f"<p>{escape(w)}</p>" for w in r["checks"][:4])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>XVA run {escape(a['run ID'])}</title><style>{PAGE}</style></head><body><main>
<h1>XVA run report</h1>
<p>Valuation date {escape(cfg['valuation_date'])}. Computed {escape(a['computed (UTC)'])} UTC. Run ID {escape(a['run ID'])}, engine {escape(a['engine'])}. Amounts in EUR.</p>
<div class="two"><div><h2>Headline</h2>{_table(head, numeric=(1,), bold=total)}</div>
<div><h2>Capital</h2>{_table(capital, numeric=(1,), bold=('Total capital',))}<h2>Initial margin</h2>{_table(margin, numeric=(1,))}<p>{escape(IM_LABEL)}. Parameters: {escape(im['source'])}.</p></div></div>
<h2>Exposure</h2>{_chart(r['profiles'], r['csa'])}
<div class="two"><div><h2>Controls on this run</h2>{_table(r['validation'].values.tolist(), head=('Check', 'Value', 'Limit', 'Status'))}</div>
<div><h2>Key assumptions</h2>{_table(assumptions)}<h2>Data checks</h2>{checks or '<p>Nothing to report.</p>'}</div></div>
<h2>Limitations</h2><p>{escape(LIMITS)}</p>
</main></body></html>
"""
