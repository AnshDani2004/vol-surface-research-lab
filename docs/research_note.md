# Research Note: Vol Surface Research Lab

> Note (expiry-clock fix, 2026-10-12): the numbers in this document predate the expiry-clock fix (time to
> expiry now runs to the 16:00 New York close, not 00:00 UTC). Current numbers are in README.md and
> outputs/; old vs new is in outputs/audit/expiry_clock_sabr_summary.csv.


## 1. Executive Summary

`vol-surface-research-lab` is a reproducible Python research engine for option
chain cleaning, implied-volatility surface construction, static-arbitrage
diagnostics, SABR calibration, Heston calibration, and model failure analysis.

The central result is not that SABR is universally better than Heston. The
central result is that, on this saved short-dated SPY snapshot, per-expiry SABR
fit the filtered local smile much better than global or per-expiry Heston
variants, while Heston sanity checks and synthetic recovery suggest the
implementation was not simply broken.

## 2. Motivation

Volatility models often look clean in notebooks and messy on actual option
chains. This project was designed to make that mess visible: quote filters,
forward extraction, OTM surface construction, calibration warnings, robustness
checks, and same-universe model comparison all live in a normal Python package
with deterministic tests and regenerated outputs.

## 3. Data Source And Snapshot Limitations

The original model-comparison data is a saved SPY yfinance snapshot from a
single point in time. The cross-date manifest now contains five saved SPY
yfinance pulls with 7,017 total raw rows, but only one currently passes quality
checks for model comparison. Four later saved pulls are retained as failed
low-quality snapshots because quote cleaning left no same-strike call/put pairs
for forward extraction. A fingerprint audit shows those failed pulls are
near-duplicate stale quote structures rather than independent
model-comparison observations.

The data is not vendor-certified, and the project does not make claims about
model performance across dates, underlyings, or market regimes.

## 4. Pipeline Overview

The pipeline moves from raw chain rows to a filtered calibration universe:

1. Load and normalize option-chain rows.
2. Compute mid prices and relative spreads.
3. Filter invalid, stale-looking, crossed, and wide-spread quotes.
4. Estimate forwards with put-call parity.
5. Select OTM/ATM rows for observed IV surface construction.
6. Run static-arbitrage diagnostics on observed quotes.
7. Calibrate SABR per expiry.
8. Stress-test SABR across filters, beta choices, and weighting modes.
9. Calibrate Heston on the recommended calibration universe.
10. Compare SABR and Heston on the same filtered rows.
11. Diagnose Heston failure modes with synthetic recovery and variants.

## 5. Stage 1: Black-Scholes, Greeks, IV Solver

Stage 1 implemented European Black-Scholes call/put pricing, Greeks, put-call
parity checks, and implied-volatility inversion with bisection and Newton
methods. The tests include finite-difference Greek checks and IV recovery from
known generated prices.

## 6. Stage 2: Chain Cleaning And Forward Extraction

Stage 2 added schema normalization, mid-price calculation, quote filters,
ACT/365 time-to-expiry, put-call-parity forward extraction, forward log
moneyness, and OTM surface labels. This created the deterministic data pipeline
used by later stages.

## 7. Stage 3: Observed IV Surface And Static-Arbitrage Diagnostics

Stage 3 converted cleaned quotes into observed implied-volatility tables and OTM
surface rows. It also added simple observed-quote static-arbitrage diagnostics:
monotonicity, convexity, and simplified calendar checks. These diagnostics are
screening tools, not full arbitrage-free interpolation.

## 8. Stage 3.5: Saved SPY Snapshot

Stage 3.5 added the saved SPY yfinance snapshot. The snapshot produced:

| Metric | Value |
|---|---:|
| Raw snapshot rows | 1169 |
| Clean rows | 957 |
| OTM surface rows | 595 |

This made the rest of the project reproducible without new option-chain
downloads.

## 9. Stage 4: SABR Calibration

Stage 4 added fixed-beta Hagan lognormal SABR calibration per expiry. On the raw
saved snapshot, SABR exposed instability in ultra-short expiries and noisy
quotes. The raw median RMSE IV was `0.018955`, with high warning counts and
bound-hitting behavior.

## 10. Stage 4.5: SABR Robustness And Recommended Universe

Stage 4.5 tested expiry filters, liquidity filters, beta choices, and weighting
modes. The selected recommended calibration universe contains:

| Metric | Value |
|---|---:|
| Rows | 206 |
| Expiries | 3 |
| SABR median RMSE IV | 0.007655 |
| Warning count | 3 |

Filtering reduced SABR median RMSE from `0.018955` to `0.007655` and reduced
warning count from `16` to `3`.

## 11. Stage 5: Heston Calibration And Same-Universe Comparison

Stage 5 implemented Heston characteristic-function pricing, put pricing by
parity, implied-volatility conversion, and global Heston calibration. It then
recomputed SABR on the exact same recommended calibration universe.

| Model | RMSE IV | Weighted RMSE IV |
|---|---:|---:|
| SABR per-expiry, median by expiry | 0.007655 | 0.007401 |
| Global Heston | 0.117433 | 0.127041 |

Global Heston produced suspicious diagnostics: `theta` near the lower bound,
Feller violation, and `v0/theta` mismatch.

## 12. Stage 5.5: Heston Improvement And Fairness Diagnostics

Stage 5.5 tested whether the Stage 5 underfit was caused by implementation
issues, quadrature sensitivity, initialization, global-model rigidity, or
parameter-instability constraints.

Key results:

| Diagnostic | Result |
|---|---|
| Heston-to-Black-Scholes limit | Price sanity check passed |
| Synthetic Heston recovery | RMSE IV `0.000043` |
| Quadrature sensitivity | 32 vs 96 node mean IV diff `0.052438` |
| Improved global multistart | No material weighted-RMSE improvement |
| Feller-penalized variants | Feller violations remained |
| Regularized Heston | Cleaner parameters, worse fit |
| Per-expiry Heston | Improved Heston fit, still underfit SABR |

## 13. Cross-Date Vol Surface Validation Extension

The cross-date extension adds a saved-snapshot manifest and deterministic
pipeline for rerunning the same cleaning, filtering, SABR calibration, Heston
calibration, and same-universe comparison across multiple local SPY snapshots.

The current checked manifest contains five saved pulls, but only one is
usable for model comparison. The generated cross-date outputs therefore validate
multi-snapshot mechanics, saved-pull collection, quality gating, and failure
preservation, but do not establish cross-date generalization. The extension is
not a trading strategy, not a broad model-ranking claim, and depends on snapshot
quality, maturity coverage, and filtering rules.

## 14. Final Findings

The strongest empirical finding is narrow and honest: on this single saved
short-dated SPY snapshot, per-expiry SABR fit the recommended calibration
universe much better than Heston variants. Heston sanity checks passed, so the
failure is more consistent with model/setup limitations and numerical
sensitivity than with an obviously broken implementation.

## 15. What Failed

- Raw SABR calibration was unstable on ultra-short expiries.
- Global Heston underfit the same filtered rows used for SABR.
- Feller-penalized Heston did not fully stabilize the Feller condition within
  the deterministic runtime cap.
- Regularization improved parameter plausibility but worsened fit.
- Heston quadrature sensitivity was material for the Stage 5 parameter set.

## 16. What Survived Robustness Checks

- The package structure and deterministic output regeneration.
- Black-Scholes pricing, Greeks, parity, and IV inversion tests.
- The saved snapshot regeneration path.
- The filtered SABR recommended universe.
- Heston implementation sanity checks and synthetic recovery.
- The conclusion that Heston underfit SABR on this same-universe comparison.
- The cross-date pipeline mechanics on the current saved-snapshot manifest.

## 17. Limitations

- Only one usable model-comparison snapshot.
- Saved yfinance SPY snapshot is not vendor-certified data.
- No historical validation.
- No trading signal.
- American exercise and dividends are not fully modeled.
- No rough volatility, local volatility, neural networks, or new model families.
- Heston quadrature sensitivity requires caution.
- The SABR/Heston result is not a broad model-ranking claim.

## 18. Reproduction Commands

```bash
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

## 19. Future Work

- Historical validation across many dates and market regimes.
- Vendor-certified chain ingestion.
- More careful dividend and American-exercise treatment.
- Independent Heston pricing checks and stricter quadrature controls.
- Optional arbitrage-aware interpolation.
