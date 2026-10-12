# Project Summary

> Note (expiry-clock fix, 2026-10-12): the numbers in this document predate the expiry-clock fix (time to
> expiry now runs to the 16:00 New York close, not 00:00 UTC). Current numbers are in README.md and
> outputs/; old vs new is in outputs/audit/expiry_clock_sabr_summary.csv.


## Project Scope

A reproducible Python options research engine for SPY option-chain ingestion,
quote filtering, forward extraction, OTM implied-volatility surface
construction, static-arbitrage diagnostics, SABR calibration, Heston
calibration, and same-universe model comparison.

## Technical Purpose

The project shows the full path from noisy option-chain rows to calibrated model
diagnostics. It emphasizes reproducibility, robustness checks, and honest
failure analysis rather than a polished model-only notebook.

## Key Numbers

| Metric | Value |
|---|---:|
| Tests | 121 passed |
| Raw snapshot rows | 1169 |
| Clean rows | 957 |
| OTM surface rows | 595 |
| Recommended calibration universe rows | 206 |
| Recommended calibration universe expiries | 3 |
| SABR median RMSE before filtering | 0.018955 |
| SABR median RMSE after filtering | 0.007655 |
| Heston global RMSE | 0.117433 |
| Synthetic Heston recovery RMSE | 0.000043 |

## Main Finding

On this saved SPY yfinance snapshot, per-expiry SABR fit the recommended
calibration universe much better than global or per-expiry Heston variants.
Heston sanity checks and synthetic recovery passed, so the underfit appears to
come from model/setup limitations and numerical sensitivity, not a basic
implementation failure.

## Cross-Date Extension

The project now includes a manifest-driven cross-date validation pipeline under
`scripts/generate_cross_date_outputs.py`. It supports multiple saved SPY
option-chain snapshots and writes reproducible artifacts under
`artifacts/cross_date/`.
The Heston artifact records global, global-by-expiry, per-expiry, and failed
calibration scopes instead of silently dropping unavailable fits.

The current checked manifest contains five saved yfinance pulls. One is usable
for model comparison; four later pulls are retained as failed low-quality
snapshots because quote cleaning left no same-strike call/put pairs for forward
extraction. A fingerprint audit shows the failed pulls are near-duplicate stale
quote structures, not independent model-comparison evidence. The generated
cross-date outputs therefore validate saved-pull handling, quality gating, and
failure preservation, but do not establish cross-date generalization.
Additional normalized CSV snapshots can be added with
`scripts/add_cross_date_snapshot.py`, and live SPY yfinance snapshots can be
collected with `scripts/collect_spy_snapshot.py`.

The failed snapshots are classified as true low-quality quote snapshots: calls,
puts, expiries, strikes, and option labels parse correctly, but quote filters
leave only four positive-mid rows and zero overlapping call/put strikes after
filtering.

## Technical Observations

- Filtering choices can dominate calibration quality.
- Same-universe comparison is essential for fair model diagnostics.
- A model can pass synthetic recovery and still underfit a real snapshot.
- Parameter plausibility and fit quality can move in opposite directions.

## Limitations

Only one usable model-comparison snapshot, not vendor-certified data, no
historical validation, no trading signal, no broad model-ranking claim, and
incomplete treatment of American exercise and dividends.

## Links To Outputs

- `outputs/final_report/final_key_metrics.csv`
- `outputs/final_report/final_model_comparison.csv`
- `outputs/stage4_5_sabr_robustness/recommended_calibration_universe.csv`
- `outputs/stage5_heston/model_comparison_by_expiry.csv`
- `outputs/stage5_5_heston_improvement/heston_improvement_summary.csv`
- `artifacts/cross_date/model_comparison_summary.csv`
