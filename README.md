# XVA calculator: CVA, DVA, FVA, MVA and KVA of a netting set of EUR swaps

Vectorised Hull-White 1F Monte Carlo engine: collateralised exposure profiles, CVA/DVA/FVA,
synthetic initial margin that still works when trade data is missing, MVA and KVA, pre-trade incremental CVA,
sensitivities, and eight controls recomputed on every run. Validated against QuantLib and the closed-form swaption price.

## Progress

| # | Step | Status |
|---|------|--------|
| 1 | Repo, curve interface, HW1F, exact simulation | done (flat curves until project 09) |
| 2 | Jamshidian, Q calibration, P comparison | done |
| — | Plug in the project 09 curves | done in the app, as a snapshot (`market/curves_09.csv`); TARGET2 schedules still to do |
| 3 | Vectorised 6M IRS / OIS; grid, netting, CSA, metrics | done |
| 4 | Credit, CVA/DVA/FVA, incremental | done |
| 5 | CRN sensitivities, SA-CCR, BA-CVA | done |
| 6 | PIT backtest, convergence, P/Q study | done |
| 7 | Streamlit, GIF; report, README, LinkedIn | done |
| 8 | Interactive app: typed inputs, CSV/Excel upload, editable trades and market data, capital and stress tabs | done |
| 9 | Synthetic initial margin with a missing-data waterfall, MVA, KVA, run controls, baseline comparison, run log, one-page report; side navigation, night mode | done |

## Demo (`app.py`)

Online: https://xva-calculator-dwrrvgykssdappmhxqokdqu.streamlit.app/ (first load ~1 min if the app was asleep). Locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Type your inputs or upload a file, then press Run. Everything is computed live (20 to 25 s for 10,000 paths on a PC,
then cached). The inputs stay editable and a changed input never overwrites the results on screen until you press Run.

| Page | What it does |
|---|---|
| Results | Mark-to-market, total XVA, adjusted value; CVA, DVA, FCA, FBA with 95% intervals; XVA stack including MVA and KVA; EEPE, peak PFE, EE/NEE/PFE profile, data checks; comparison with a pinned baseline; Excel and one-page HTML exports |
| Trades | Editable netting set (IRS or OIS, payer or receiver, forward-starting or seasoned, years or real dates); per trade: standalone CVA, Euler CVA, leave-one-out ΔCVA, netting benefit |
| Market data | Editable curves with presets (flat, ECB AAA curve, project 09 bootstrapped curves), CDS spreads, recoveries, funding spread, swaption vols, mean reversion, σ override, fit table |
| Pre-trade | Candidate trades compared on the paths already simulated: ΔCVA, ΔDVA, ΔFVA, ΔXVA, charge in bp, all-in rate |
| Sensitivities | On demand: deltas per pillar and parallel, vega, with standard errors (common random numbers); CS01 per CDS pillar |
| Initial margin | Synthetic margin with its low-to-high range, coverage by data level, what the same book gives at each level, margin over the life of the book, MVA, editable parameters |
| Capital | SA-CCR EAD, SCVA and reduced BA-CVA by sector and credit quality, next to the accounting CVA; capital profile and KVA |
| Stress | Editable scenarios (rates, vols, counterparty and bank spreads), full revaluation on the same draws |
| Controls | Eight checks recomputed on this run, run ID, engine hash, run log of the session |
| Method | File formats, model, proxies and limitations |

The left sidebar holds the page menu, Run, your files, the CSA (thresholds, MTA, independent amount, margin period of risk,
on/off), the simulation (paths, seed, grid step, PFE level, valuation date) and the night mode button.

### Your own data

**Template** gives the format, **My inputs** exports what is on screen, and the results export to Excel from the Results tab.

- **Excel**: one sheet per table (`trades`, `candidates`, `curves`, `credit`, `vols`, `stress`, `positions`, `im_params`, `settings`). Only the tables found are replaced, so a file with your trades alone is enough.
- **CSV**: one table per file, recognised by the file name or its columns. Semicolons, decimal commas and thousands separators are read, and common column names too (`nominal`, `strike`, `maturity_date`...).
- **Trades**: `name, type, side, notional, fixed_rate_pct, start, tenor` or `maturity`. `start` and `maturity` are years from the valuation date or dates; a past start makes a seasoned trade, with `last_fixing_pct` for the running coupon. An optional `netting_set` column adds a selector.
- **Positions with partial data**: `name, side, notional, maturity, dv01, netting_set`, for the initial margin only. Leave blank what you do not know. A file that replaces the trades without this table clears the demo positions, and says so.
- **Margin parameters**: `vertex, risk_weight_bp, rho_2w ... rho_30y`.

A bad file is refused as a whole, with the row and column of each problem, and the inputs stay as they were.
Files are read in memory and never written to disk. The public demo runs on a shared free server: for a confidential
book, clone the repository and run it on your machine. `XVA_MAX_MB` raises the memory limit (600 MB by default).

![Pre-trade](report/figures/pre_trade.gif)

Validation report: [`report/validation.pdf`](report/validation.pdf) (self-review, limitations, recommendations;
independent review still to do).

## Curves and the project 09 plug-in (`src/xva/market.py`)

The engine knows a single interface: a callable `t -> P(0, t)`, `t` in ACT/365F years.
A `Market` object carries two curves, `estr` (discounting) and `euribor6m` (projection).

- `sample_market()` returns two flat curves 20 bp apart, which already exercises the multi-curve set-up (β ≠ 1).
- `ZeroCurve(tenors, rates)` takes continuous zero rates per pillar: what the app's editor, files and presets use.
- `market/curves_09.csv` is a snapshot of the project 09 curves (`scripts/snapshot_curves_09.py`), levels from its
  reconstructed quotes, so the online demo does not need the `ratecurves` package.
- `from_curve_09(curve)` wraps a `Curve` / `CurveHW` from `ratecurves` (method `P(date)`) for a direct plug-in:
  `Market(from_curve_09(estr), from_curve_09(euribor6m))`.
- Left to do: replace the regular year-fraction schedules in `swap_coupons` with the TARGET2 dates from 09
  (marked `shortcut:` in the code).

## Hull-White 1F (`src/xva/hw1f.py`)

- Closed-form ZC price `P(t, T | x)`, exact joint simulation of `(x, ∫x)` on an irregular grid,
  discount factor `D(0, t) = P(0, t)·exp(-I - V/2)` so `E[D] = P(0, t)` without bias,
  antithetic variates.
- Jamshidian valid for coupons of any sign (the 6M/€STR basis creates small negative coupons).
- Q calibration: `a = 0.03` fixed, `σ` fitted in normal vol to the 10-year co-terminal ATM swaptions
  (`market/swaption_vols.yaml`, **synthetic** for lack of a free source).
- P calibration (comparison only): `σ·B(τ)/τ` fitted to the annualised standard deviations of daily changes
  in the ECB 1, 5 and 10-year spot rates (one year of history, `market/ecb/`).

First figures (flat curves, synthetic vols): σ(Q) = 0.84%, max vol error 3.6 bp;
σ(P) = 0.58%, i.e. **-31%**: a CVA calibrated on history would be understated by about as much.

## Products (`src/xva/products.py`)

- EURIBOR 6M IRS (deterministic 6M/€STR basis) and €STR OIS, payer or receiver, spot or forward start.
- `value(trades, model, market, t, x_t, D_t, state)`: the `value(t, state)` interface where the neural pricer
  will plug in. Each date = one ZC matrix (paths × maturities) × one weight matrix
  (maturities × trades); only the current coupon, fixed on the path, is added trade by trade.
- `cube_mtm`: MtM cube paths × dates × trades, kept for incremental CVA.
- Fix to the project sheet: the running €STR leg is worth `exp(∫_s^t r) − P(t, e)`, not `exp(∫r)·P(t, e) − P(t, e)`
  (error caught by the martingale test).
- Seasoned trades: the coupon running at the valuation date comes from `last_fixing_pct` (the EURIBOR fixing, or the
  annualised €STR accrued so far), then the remaining cash flows are valued as usual.

## Exposure (`src/xva/exposure.py`)

- Grid: monthly ∪ cash-flow dates ∪ `t − MPoR` nodes.
- Classic collateral `C(t) = max(V(t−M) − H_C, 0) − max(−V(t−M) − H_B, 0)`, MTA added to the thresholds
  (documented approximation), cash flows during the MPoR ignored. An independent amount shifts the collateral held.
- EE, NEE, PFE (97.5% by default, adjustable), discounted EE* and NEE* with standard errors, EEPE (effective EE over one year).

Demo netting set (`portfolios/demo.yaml`, 7 fictitious swaps, 435 M€ notional, Q-calibrated σ, 10,000 paths,
240 dates, cube in 4 s on a PC):

| | EEPE | Max EE | Max PFE |
|---|---|---|---|
| No CSA | 6.2 M€ | 9.3 M€ | 36.8 M€ |
| CSA (H = 10 M€, MTA 0.5 M€, MPoR 10 d) | 5.3 M€ | 5.4 M€ | 13.9 M€ |

## Credit and XVA (`src/xva/xva.py`)

- Survival: piecewise-constant λ bootstrapped on 1, 3, 5, 7, 10-year CDS spreads, R = 40%,
  MidPoint convention (default mid-period, accrued coupon paid at default), regular quarterly CDS
  (no IMM dates). **Synthetic** spreads in `market/spreads.yaml` (BBB counterparty, A-rated bank).
- CVA, DVA, FCA, FBA per the project sheet formulas, EE* at the middle of each interval by trapezoid.
  Funding: `ℓ = max(s_F − (1 − R_B)·λ_B, 0)`, so FBA does not double-count DVA.
- Standard errors computed per antithetic pair.
- `pre_trade`: ΔXVA of a new trade on the same paths, only its MtM is computed.
- `allocation_euler`: marginal CVA per trade (uncollateralised), exact sum = CVA.

```bash
python -m xva.demo
```

Demo netting set, Q-calibrated σ, 10,000 paths (k€, standard error in brackets):

| | CVA | DVA | FCA | FBA | Adjusted value (V0 = 5,090) |
|---|---|---|---|---|---|
| No CSA | 659.7 (3.2) | 227.8 (2.4) | 81.2 (0.3) | 34.2 (0.4) | 4,609.1 |
| CSA | 446.3 (0.7) | 159.5 (1.1) | 57.0 (< 0.1) | 23.5 (0.2) | 4,767.7 |

The CSA removes 32% of the CVA. Euler allocation (no CSA): IRS-01 394 k€, IRS-03 126 k€, IRS-05 103 k€.
Pre-trade for a 50 M€ 10-year payer IRS at 2.60%: ΔCVA = 146.5 k€ (SE 0.9) without CSA, 28.0 k€ (SE 0.2) with CSA,
in 1.4 to 1.6 s.

## Synthetic initial margin, MVA, KVA (`src/xva/initial_margin.py`)

**Synthetic SIMM-style initial margin, not ISDA SIMM®, no affiliation with ISDA.** It keeps the public functional form of a
sensitivity-based margin (risk weights, tenor correlations, sub-curve correlation, concentration factor), delta only, one
currency. The ISDA parameter tables are not in this repository: page 1 of the ISDA documents limits their use and
distribution, so the parameters here are our own. `scripts/calibrate_im.py` writes them to `market/im_params.csv`
from the public ECB AAA history. They are not tuned to match ISDA and this is not a regulatory margin. A user with their own
licence can type official parameters in the `im_params` table.

What is known of a position sets how it is margined:

| Level | Known | Method |
|---|---|---|
| 1 | Full terms | Exact ladder of the trade, by revaluation |
| 2 | DV01 | Unsigned ladder on the 6M curve, split linearly at maturity; without a maturity, on the vertex with the highest risk weight |
| 3 | Notional and maturity | Same, with DV01 = notional × annuity × 1 bp |
| 4 | Notional only | BCBS-IOSCO schedule (MGN20.16), worst bucket 4%, added without offset |

A blank direction gives a range, because the margin K is a norm: central `√(k² + ΣK(uᵢ)²)`, the root mean square over
independent signs; high `k + K(Σu)`; low `max(k − K(Σu), 0)`. The page also re-margines the trades of the book as if less were
known about them, which prices the missing data on that book, and backtests the margin against 10-business-day ECB moves
(the backtest shows the exposure of the ladder, not of a live portfolio).

- MVA: the margin is aged along the life of the book on forward curves, then funded at the funding spread (`xva.mva`), with its own range.
- KVA: capital profile `capital_profile` (SA-CCR and reduced BA-CVA on the mean future value), then `xva.kva` at the hurdle rate (10% by default).
- **Total XVA stays CVA − DVA + FCA − FBA**; "XVA including MVA and KVA" is shown next to it. Pre-trade and stress do not cover MVA and KVA.

Demo netting set, 10,000 paths: margin 8.37 M€ (7.89 to 8.83 M€), of which 80.7% on full terms; MVA 95.5 k€; KVA 999.2 k€;
XVA including MVA and KVA 1,415.0 k€. Same trades margined on notional only: +143%, which is the cost of not knowing them.

## Run controls and audit trail (`live_checks`, `src/xva/pipeline.py`, `src/xva/onepager.py`)

Every run recomputes eight checks on its own numbers, on the Controls page and in the exports: discount factors against the
curve, martingale of the MtM against the PV of later cash flows, CDS repricing, calibration error, Euler sum, CVA standard error,
margin correlation matrix, margin backtest. A failed check shows a warning on Results.

- Run ID: hash of the inputs, so the same inputs give the same ID whatever the date. Engine hash: hash of the code in `src/xva`.
- Baseline: pin a run, change an input, press Run: every headline figure moves side by side with the list of inputs that differ.
- Run log: the last 20 runs of the session, on the Controls page and in the Excel export. It lasts as long as the browser session.
- One-page report: a self-contained HTML file with nothing fetched, to print to PDF on one A4 page.

## Sensitivities and capital (`src/xva/risk.py`)

- Deltas: 1 bp bump to the zero rates of both curves, as triangles at the 1, 2, 5, 10, 30-year pillars
  (they sum to the parallel bump), central, **same draws** (common random numbers), σ recalibrated at each bump.
- Vega: swaption vols ±1% relative, σ recalibrated, same draws.
- CS01: counterparty CDS spreads ±1 bp per pillar, re-bootstrap, same exposures (no resimulation).
- SA-CCR rates (CRE52): one EUR hedging set, buckets < 1, 1-5, > 5 years, delta ±1, margined and unmargined MF,
  multiplier, margined EAD capped by the unmargined one.
- Reduced BA-CVA (MAR50): `K = 0.65·√[(0.5·ΣSCVA)² + 0.75·ΣSCVA²]`. Approximations: effective maturity
  = notional-weighted residual maturity (floor 1 year); risk weights per sector and credit quality from MAR50
  (demo counterparty: IG industrials, 3%).

Demo netting set with CSA, 10,000 paths, € per +1 bp (standard error in brackets), 46 s for 14 revaluations:

| Bump | 1Y | 2Y | 5Y | 10Y | Parallel |
|---|---|---|---|---|---|
| CVA delta | −62 | −589 | −1,007 | 4,725 | 3,068 (26) |

| CDS pillar | 1Y | 3Y | 5Y | 7Y | 10Y | Parallel |
|---|---|---|---|---|---|---|
| CVA CS01 | 17 | 92 | 289 | 1,055 | 1,300 | 2,754 (3) |

Sum of pillar deltas = parallel to within 1 €, same for the CS01s. CVA vega (+1% of vols): 1,123 €.
Standard errors stay below 1% of the sensitivity thanks to common draws.

Capital: SA-CCR EAD 16.9 M€ (add-on 7.0 M€, RC 5.1 M€), reduced BA-CVA 1.29 M€, against an accounting CVA of 0.66 M€.
With the demo CSA (10 M€ threshold), RC = threshold + MTA = 10.5 M€: the margined EAD (17.6 M€) exceeds the unmargined one
and is capped. A high-threshold CSA does not reduce SA-CCR capital.

## Validation (`src/xva/validation.py`, `python -m xva.report`)

`pip install -e .[report]` then `python -m xva.report` (a few minutes): CSV tables and figures in `report/`.

**Convergence** (demo netting set, no CSA): standard error in `N^-0.504`, antithetics divide the variance
by 4.1 (i.e. 4 times fewer paths). 10,000 paths: CVA 653.0 k€ (SE 3.1), 1.4 s; 50,000: 655.5 k€ (SE 1.4), 6.6 s.

![Convergence](report/figures/convergence.png)

**EE* against the closed-form swaption** (single 10 M€ 5-year payer at par, 40,000 paths): max error 0.46 bp of notional
at the fixed coupon dates, below the standard error (0.87 bp).

![EE against swaption](report/figures/ee_swaption.png)

**P measure against Q** (same draws, 10,000 paths, change vs the CVA under Q):

| Case | σ | CVA | Change |
|---|---|---|---|
| Q, implied σ (reference) | 0.84% | 653.0 k€ | |
| P, historical σ (1 year ECB) | 0.58% | 499.5 k€ | −23.5% |
| Vols +20% / −20% | 1.01% / 0.67% | 755.2 / 552.0 k€ | +15.7% / −15.5% |
| a = 0.01 / 0.05 (σ recalibrated) | 0.76% / 0.92% | 649.8 / 655.9 k€ | −0.5% / +0.4% |
| Spreads ±25% | 0.84% | 800.8 / 499.3 k€ | +22.6% / −23.5% |

A CVA computed under P would be a quarter lower: the implied vol premium weighs as much as a 25% shock
to spreads. Mean reversion matters little once σ is recalibrated on the same swaptions.

**Point-in-time backtest of exposures under P**: daily ECB AAA curves 2004-2026 (5,665 days, free ECB
API). At each date, σ(P) calibrated on the previous year (no look-ahead), distribution of the value
at the horizon simulated from that day's curve, PIT = rank of the realised value. Non-overlapping windows.
Three netting sets starting at the horizon: 5-year payer, 10-year receiver, steepener (10-year payer against
2-year receiver of similar DV01). Single AAA curve, the only long and free history.

| Horizon | Windows | KS power (true vol ×1.25) | KS payer 5Y | KS receiver 10Y | KS steepener |
|---|---|---|---|---|---|
| 1 month | 257 | 55% | 0.108 (p 0.005) | 0.127 (p < 0.001) | 0.444 (p < 0.001) |
| 3 months | 85 | 21%, indicative | 0.162 (p 0.02) | 0.153 (p 0.03) | 0.462 (p < 0.001) |
| 1 year | 21 | 10%, indicative | 0.268 (p 0.08) | 0.244 (p 0.14) | 0.416 (p 0.001) |

![PIT](report/figures/backtest_pit.png)

Reading: on directional positions, PITs pile up at both ends (at 1 month, 6 to 10% of
realisations below the 2.5% quantile instead of 2.5%): historical tails are fatter than the
HW1F Gaussian and σ(P) on a rolling year reacts late to regime changes (2008, 2022).
On the steepener, almost all realisations fall outside the simulated distribution: with a single factor,
2-year and 10-year rates move together and the model sees almost no risk on a DV01-neutral position.
This is the known limit of HW1F; a portfolio exposed to the slope would need a two-factor model.
At 3 months and 1 year, power is too low to conclude; those rows are indicative.

## Tests

```bash
pip install -e .[test]
pytest
```

161 tests, about 2.5 min on a PC:
- HW1F: martingale `E[D] = P(0, t)`, moments of `x` and `∫x`, forward ZC consistency, ZC option and
  Jamshidian swaption against QuantLib 1.44, Jamshidian against Monte Carlo in multi-curve, parity,
  Q and P calibrations recovering a known σ, project 09 adapter.
- Products: multi-curve IRS against QuantLib (< 1e-10 of notional), OIS against QuantLib (1e-8, daily
  TARGET compounding); `E[D·V(t)]` = value of remaining cash flows, including mid-period;
  EE* of a single payer = swaption on the residual swap (± 3 SE); PFE within the binomial CI of the closed-form quantile.
- Exposure: sub-additive netting, mirror swaps → 0, infinite threshold = uncollateralised, perfect collateral
  at zero MPoR → 0, EE increasing in MPoR and √MPoR law (± 15% away from cash-flow dates), demo with and without CSA.
- XVA: the bootstrap reprices the spreads; par spread against QuantLib `MidPointCdsEngine` (< 0.1 bp;
  the project sheet's `IsdaCdsEngine` replaced by MidPoint, the engine's convention); CVA of a single swap = sum
  of closed-form swaptions (< 2% and 3 SE); sum of Euler allocations = CVA; pre-trade = full recompute;
  bank CVA = counterparty DVA; ℓ = 0 → no FVA.
- Risk: triangles summing to 1; sum of pillar deltas = parallel (< 5%); delta and vega signs
  of a payer; sum of CS01s = parallel; SA-CCR by hand (one trade, two buckets with multiplier, margined,
  cap); reduced BA-CVA by hand (one and two counterparties).
- Validation: uniform PITs (KS not rejected) when realised curves are drawn from the model itself;
  KS power at nominal level with no misspecification and increasing with the number of windows; convergence slope −0.5.
- Inputs: Excel round trip keeps every value, CSV variants (semicolons, decimal commas, aliases), each bad file or cell
  refused with its row and column, seasoned trades and dates, curve presets, stress scenarios.
- Pipeline: demo regression, adjusted value identity, Euler sum, memory guard, impossible credit curve, stress base
  = run on the same draws, results workbook.
- Initial margin: each waterfall level by hand, range ordering, positive semi-definite correlations, parameter file against the calibration, schedule figures, MVA and KVA.
- Controls: all pass on the demo and on seasoned and OIS trades, the martingale check fails on a deliberately wrong cube, run ID follows the
  inputs and not the clock, baseline and input diff, report self-contained with user text escaped.
- App (`streamlit.testing`): first load, typed numbers, a changed input waits for Run, CSA off, stress, sensitivities,
  preset then reset, margin page, controls and baseline. File upload, night mode and the side menu need a real browser, so they were tested by hand with Playwright.
