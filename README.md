# Vol Surface Research Lab

A reproducible Python options research engine for SPY option-chain cleaning, IV
surface construction, static-arbitrage diagnostics, SABR calibration, Heston
calibration, robustness testing, and failure-mode analysis. The repository is a
package-first research artifact with deterministic scripts, tests, saved input
data, and generated outputs.

## Key Result

On the one saved SPY option-chain snapshot that passes quality checks
(yfinance, retrieved 2026-07-12 20:17 UTC, expiries 2026-07-13 to 2026-07-20),
per-expiry SABR median RMSE is `0.018955` on the full OTM surface (595 quotes,
6 expiries) and `0.007655` on the filtered calibration universe (206 quotes,
3 expiries). RMSE is in implied-volatility units (decimal), so `0.0077` means
about 0.77 vol points.

Read this as the same model fit to two different quote sets, not as a model
improvement. Filtering removes the hardest quotes:

| Step | Quotes | Expiries | Median SABR RMSE | Source |
|---|---:|---:|---:|---|
| Full OTM surface | 595 | 6 | 0.018955 | `outputs/stage4_5_sabr_robustness/robustness_summary.csv`, `base_stage4` |
| Drop expiries under 2 trading days | 311 | 3 | 0.012705 | same file, `exclude_T_lt_2_trading_days` |
| Also require relative spread <= 0.25, volume > 0, open interest > 0 | 206 | 3 | 0.007655 | same file, `recommended` |

Of the 0.0113 drop, 0.0063 comes from dropping the three shortest expiries and
0.0051 from dropping wide-spread or untraded quotes on the remaining three. The filter
was chosen on this same snapshot, and each fit is in-sample. On the 206-quote
universe SABR has lower RMSE than global and per-expiry Heston on all three
expiries (`outputs/stage5_heston/model_comparison_by_expiry.csv`,
`outputs/stage5_5_heston_improvement/heston_variant_comparison_by_expiry.csv`).
The Heston fit is numerically fragile at these 3 to 7 calendar day maturities (see
Limitations), so the size of the Heston gap is not reliable. This is a
reproducible calibration and diagnostics framework, not a trading signal or a
broad model-ranking claim.

## What This Project Proves

- Black-Scholes pricing, Greeks, and IV inversion were implemented and tested.
- Real SPY option-chain snapshot ingestion and deterministic regeneration were
  added.
- Quote filtering, forward extraction, and OTM IV surface construction were
  implemented.
- Static-arbitrage diagnostics (strike monotonicity, call convexity, calendar
  monotonicity) were run on the cleaned quotes. They flag violations (138 rows
  in `outputs/stage3_real_snapshot/arbitrage_report.csv`); they do not remove
  or repair them.
- SABR calibration was stress-tested across expiry filters, liquidity filters,
  beta choices, and weighting modes.
- Heston pricing/calibration was implemented and compared on the same filtered
  universe.
- Heston underperformance was investigated through sanity checks, synthetic
  recovery, quadrature sensitivity, multistart, Feller/regularized variants, and
  per-expiry diagnostics. Synthetic recovery passed, which rules out a gross
  pricer or calibrator bug at the tested maturities (0.05 to 0.25 years). It
  does not explain the underfit on the real 3 to 7 calendar day maturities; the
  quadrature check points to numerical error there.

## Headline Metrics

| Metric | Value |
|---|---:|
| Tests | 121 passed (rerun 2026-10-12, Python 3.10, macOS) |
| Raw snapshot rows | 1169 |
| Clean rows | 957 |
| OTM surface rows | 595 |
| Recommended universe rows | 206 |
| Recommended universe expiries | 3 |
| SABR median RMSE, full OTM surface (595 quotes, 6 expiries) | 0.018955 |
| SABR median RMSE, filtered universe (206 quotes, 3 expiries) | 0.007655 |
| Static-arbitrage violations flagged | 138 |
| Heston global RMSE, Stage 5 single-snapshot path | 0.117433 |
| Synthetic Heston recovery RMSE | 0.000043 |
| Quadrature sensitivity 32 vs 96 node mean IV diff | 0.052438 |

## Cross-Date Vol Surface Validation

The cross-date extension adds a manifest-driven pipeline for running the same
cleaning, filtering, SABR calibration, Heston calibration, and same-universe
comparison across multiple saved SPY option-chain snapshots.
The comparison artifact uses per-expiry Heston fits where available and records
global, per-expiry, fallback, and failed calibration scopes explicitly.

The current checked manifest contains five saved SPY yfinance pulls. The first
snapshot is usable for model comparison; four later pulls are preserved as
low-quality failed snapshots because quote cleaning left no same-strike call/put
pairs for forward extraction. The fingerprint audit shows the failed pulls are
near-duplicate stale quote structures: they have different raw file hashes and
timestamps, but the same spot, expiries, strike set, row counts, positive-mid
count, and post-filter call/put overlap. They are retained as data-quality
diagnostics, not independent evidence.

The failed pulls are true low-quality quote snapshots under the current filters,
not schema, timezone, or call/put pairing bugs. Each contains both calls and
puts across six expiries, but only four rows have positive mids after bid/ask
checks, leaving zero overlapping call/put strikes after filtering.

This is not a trading strategy, does not support a broad model-ranking claim, and
depends on snapshot quality, maturity coverage, and filtering rules. All
cross-date outputs are reproducible from saved data. Existing normalized CSV
snapshots can be added with:

```bash
python scripts/add_cross_date_snapshot.py \
  --snapshot-id spy_YYYY_MM_DD_HHMM \
  --raw-csv path/to/normalized_option_chain.csv \
  --metadata-json path/to/metadata.json \
  --market-regime saved_yfinance_snapshot \
  --notes "Saved SPY snapshot; not vendor-certified."
```

Live SPY yfinance snapshots can be collected and quality-gated with:

```bash
python scripts/collect_spy_snapshot.py \
  --market-regime active_market_hours \
  --notes "Saved SPY snapshot; yfinance is not vendor-certified."
```

The collector saves the raw CSV and metadata locally without overwriting. It
appends the manifest only if pre-ingestion checks pass unless
`--allow-low-quality` is explicit. Tests never download live data.

Current cross-date artifact summary:

| Metric | Value |
|---|---:|
| Saved snapshots processed | 5 |
| Usable model-comparison snapshots | 1 |
| Total raw rows processed | 7,017 |
| Total recommended calibration rows | 206 |
| SABR expiry wins vs Heston | 3 |
| Heston expiry wins vs SABR | 0 |
| Median global Heston RMSE, cross-date path | 0.117464 |
| Generalization status | insufficient snapshots |
| Conclusion stability | insufficient |

The Stage 5 single-snapshot path reports Heston global RMSE `0.117433`; the
cross-date regeneration path reports median global Heston RMSE `0.117464` under
its saved-manifest execution settings. Both are single-usable-snapshot
diagnostics and should not be read as cross-date model evidence.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python scripts/generate_stage3_real_snapshot_outputs.py
python scripts/generate_stage4_sabr_outputs.py
python scripts/generate_stage4_5_sabr_robustness_outputs.py
python scripts/generate_stage5_heston_outputs.py
python scripts/generate_stage5_5_heston_improvement_outputs.py
python scripts/generate_cross_date_outputs.py
python scripts/generate_final_report_outputs.py
python -m pytest
python -m ruff check .
```

## Repository Map

```text
src/volsurface/        Python package
tests/                 pytest suite
data/snapshots/        saved SPY yfinance snapshot
data/raw_snapshots/    cross-date snapshot manifest
data/processed_snapshots/ generated per-snapshot processed tables
outputs/               deterministic generated reports
artifacts/cross_date/  cross-date validation artifacts
docs/                  research notes and technical documentation
scripts/               output regeneration scripts
notebooks/             optional exploration area
```

## Final Report

- [Research note](docs/research_note.md)
- [Project summary](docs/project_summary.md)
- [Technical case study](docs/technical_case_study.md)
- [Technical FAQ](docs/technical_faq.md)
- [Cross-date validation note](docs/cross_date_validation.md)
- [Final report outputs](outputs/final_report/)

## Limitations

- One usable snapshot. Five saved yfinance pulls; one is usable for model
  comparison and the other four are near-duplicate stale quote structures
  (`artifacts/cross_date/aggregate_summary.csv`: `usable_snapshot_count` 1,
  `generalization_status` insufficient_snapshots). Nothing here is cross-date
  or out-of-sample evidence.
- One short-dated regime. All six expiries are within eight calendar days of
  the snapshot; the calibrated three are 3 to 7 calendar days out
  (T = 0.0086 to 0.0196 years, ACT/365). No
  monthly or longer expiries.
- Stale, off-hours quotes. The snapshot was retrieved on Sunday 2026-07-12 at
  20:17 UTC, so quotes are from the prior session. yfinance is not
  vendor-certified.
- Before/after RMSE uses different quote sets (595 vs 206 quotes, 6 vs 3
  expiries). The filter was chosen on the same snapshot, and fits are
  in-sample.
- Expiry clock. `src/volsurface/time.py` measures time to 00:00 UTC on the
  expiry date, not the 16:00 ET close, which shortens every maturity by about
  20 hours. An audit rerun with expiry at 20:00 UTC
  (`scripts/audit_sensitivity_checks.py`, `outputs/audit/expiry_clock_sabr_summary.csv`)
  gives full-surface median SABR RMSE 0.014315 instead of 0.018955, cuts the
  2026-07-13 expiry RMSE from 0.364 to 0.039, and admits a fourth expiry
  (273 quotes, median 0.006035). Much of the "ultra-short expiries are
  unstable" diagnosis is this clock convention. The headline outputs have not
  been regenerated with the change.
- Heston numerics. Calibration uses 32 Gauss-Laguerre nodes; at the fitted
  parameters, 32 vs 96 nodes differ by 0.052 mean and 0.262 max in IV
  (`outputs/stage5_5_heston_improvement/quadrature_sensitivity.csv`).
  Recalibrating global Heston at 96 nodes gives RMSE 0.0795 instead of 0.1174,
  with theta at its lower bound and kappa near its upper bound; at 192 nodes
  the pricer overflows (`outputs/audit/heston_quadrature_global_fit.csv`).
  SABR stays lower either way, but the Heston RMSE level is not converged.
- Synthetic Heston recovery uses maturities 0.05 to 0.25 years, far longer
  than the real ones, so it does not validate the pricer in the real regime.
- Flat 5% rate; dividends enter only through the put-call-parity forward;
  American exercise is not modeled (SPY options are American).
- No trading signal, no hedging or P&L test.
- `pytest` regenerates some tracked artifacts in place; rerunning on another
  machine changes floating-point digits (for example cross-date global Heston
  RMSE 0.117464 vs 0.117424) without changing any reported conclusion.

Remaining optional work: more snapshots during market hours, the 16:00 ET
expiry clock, a converged Heston pricer, historical validation and
arbitrage-aware interpolation.
