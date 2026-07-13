# Project Summary

## What I Built

A reproducible Python options research engine for SPY option-chain ingestion,
quote filtering, forward extraction, OTM implied-volatility surface
construction, static-arbitrage diagnostics, SABR calibration, Heston
calibration, and same-universe model comparison.

## Why It Matters

The project shows the full path from noisy option-chain rows to calibrated model
diagnostics. It emphasizes reproducibility, robustness checks, and honest
failure analysis rather than a polished model-only notebook.

## Key Numbers

| Metric | Value |
|---|---:|
| Tests | 101 passed |
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

## What I Learned

- Filtering choices can dominate calibration quality.
- Same-universe comparison is essential for fair model diagnostics.
- A model can pass synthetic recovery and still underfit a real snapshot.
- Parameter plausibility and fit quality can move in opposite directions.

## Limitations

Single point-in-time snapshot, not vendor-certified data, no historical
validation, no trading signal, no universal model ranking, and incomplete
treatment of American exercise and dividends.

## Links To Outputs

- `outputs/final_report/final_key_metrics.csv`
- `outputs/final_report/final_model_comparison.csv`
- `outputs/stage4_5_sabr_robustness/recommended_calibration_universe.csv`
- `outputs/stage5_heston/model_comparison_by_expiry.csv`
- `outputs/stage5_5_heston_improvement/heston_improvement_summary.csv`
