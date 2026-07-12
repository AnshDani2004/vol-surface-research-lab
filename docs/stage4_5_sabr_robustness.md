# Stage 4.5: SABR Robustness

## Scope

Stage 4.5 does not try to make SABR look good. It diagnoses which SABR
calibration claims survive changes in expiry filters, liquidity filters, beta
choice, and weighting.

This stage uses the saved Stage 3.5 SPY yfinance snapshot and the Stage 4 SABR
machinery. It does not add Heston, model comparison against Heston, historical
backtesting, live-data dependency in deterministic regeneration, trading claims,
or final resume claims.

## Input Source

Main input:

- `outputs/stage3_real_snapshot/otm_surface_table.csv`
- enriched with liquidity fields from `outputs/stage3_real_snapshot/clean_chain.csv`

Preserved fields include bid, ask, mid, bid-ask spread, relative spread, volume,
open interest, optional last-trade timestamp, time to expiry, log moneyness,
surface-side label, and implied volatility.

## Why Robustness Was Needed

Stage 4 showed parameter-bound pressure and large residuals, especially for the
nearest expiry. Stage 4.5 checks whether those warnings are driven by
ultra-short expiries, wide or low-liquidity quotes, beta choice, weighting
choice, or parameter-bound behavior.

## Scenarios Tested

Expiry filters:

- `all_expiries`
- `exclude_T_lt_1_trading_day`
- `exclude_T_lt_2_trading_days`
- `exclude_T_lt_5_trading_days`

Liquidity filters:

- `base_stage4`
- `relative_spread_lte_0_25`
- `relative_spread_lte_0_10`
- `volume_gt_0`
- `open_interest_gt_0`
- `volume_gt_0_and_open_interest_gt_0`
- `relative_spread_lte_0_25_and_volume_gt_0_and_open_interest_gt_0`

Beta values:

- `0.0`
- `0.5`
- `1.0`

Weighting modes:

- `unweighted`
- `spread_weighted`
- `vega_weighted`

Vega weighting uses Black-Scholes vega at the market IV with a zero-rate
weighting convention, then normalizes and clips weights to avoid exploding
optimizer scales.

## Actual Commands

Generated outputs:

```bash
python scripts/generate_stage4_5_sabr_robustness_outputs.py
```

Validated package:

```bash
python -m pytest
python -m ruff check .
```

Actual validation output:

```text
...............................................................          [100%]
63 passed in 3.15s
```

```text
All checks passed!
```

## Expiry Filter Sensitivity

| Scenario | Expiries | Fit Points | Success | Skipped | Median RMSE | Median MAE | Median Weighted RMSE | Max Max Abs Error | Warnings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_expiries | 6 | 595 | 6 | 0 | 0.018955 | 0.014687 | 0.018754 | 0.758247 | 16 |
| exclude_T_lt_1_trading_day | 4 | 424 | 4 | 0 | 0.015219 | 0.012734 | 0.014033 | 0.085668 | 7 |
| exclude_T_lt_2_trading_days | 3 | 311 | 3 | 0 | 0.012705 | 0.011025 | 0.011731 | 0.059228 | 5 |
| exclude_T_lt_5_trading_days | 0 | 0 | 0 | 0 |  |  |  |  | 0 |

## Liquidity Filter Sensitivity

| Scenario | Rows Before | Rows After | Removed | Expiries | Median RMSE | Median MAE | Median Weighted RMSE | Max Max Abs Error | Warnings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_stage4 | 595 | 595 | 0 | 6 | 0.018955 | 0.014687 | 0.018754 | 0.758247 | 16 |
| relative_spread_lte_0_25 | 595 | 360 | 235 | 6 | 0.008402 | 0.007283 | 0.008039 | 0.233334 | 11 |
| relative_spread_lte_0_10 | 595 | 259 | 336 | 6 | 0.002755 | 0.002312 | 0.002713 | 0.185163 | 11 |
| volume_gt_0 | 595 | 581 | 14 | 6 | 0.018997 | 0.014538 | 0.018850 | 0.758247 | 16 |
| open_interest_gt_0 | 595 | 595 | 0 | 6 | 0.018955 | 0.014687 | 0.018754 | 0.758247 | 16 |
| volume_gt_0_and_open_interest_gt_0 | 595 | 581 | 14 | 6 | 0.018997 | 0.014538 | 0.018850 | 0.758247 | 16 |
| relative_spread_lte_0_25_and_volume_gt_0_and_open_interest_gt_0 | 595 | 359 | 236 | 6 | 0.008382 | 0.007268 | 0.008007 | 0.233334 | 11 |

## Beta Sensitivity

Aggregate beta sensitivity:

| Universe | Beta | Fit Points | Success | Median RMSE | Median MAE | Median Weighted RMSE | Max Max Abs Error | Warnings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base_stage4 | 0.0 | 595 | 6 | 0.153747 | 0.141359 | 0.157413 | 1.052107 | 23 |
| base_stage4 | 0.5 | 595 | 6 | 0.018955 | 0.014687 | 0.018754 | 0.758247 | 16 |
| base_stage4 | 1.0 | 595 | 6 | 0.019620 | 0.015180 | 0.019597 | 0.415100 | 14 |
| recommended | 0.0 | 206 | 3 | 0.116841 | 0.113653 | 0.118134 | 0.176147 | 9 |
| recommended | 0.5 | 206 | 3 | 0.007655 | 0.006613 | 0.007401 | 0.024268 | 3 |
| recommended | 1.0 | 206 | 3 | 0.007887 | 0.006806 | 0.007623 | 0.025030 | 3 |

## Weighting Sensitivity

| Universe | Weighting | Fit Points | Success | Median RMSE | Median MAE | Median Weighted RMSE | Max Max Abs Error | Warnings |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base_stage4 | unweighted | 595 | 6 | 0.018833 | 0.014615 | 0.018833 | 0.758247 | 14 |
| base_stage4 | spread_weighted | 595 | 6 | 0.018955 | 0.014687 | 0.018754 | 0.758247 | 16 |
| base_stage4 | vega_weighted | 595 | 6 | 0.025419 | 0.015077 | 0.008159 | 0.758247 | 13 |
| recommended | unweighted | 206 | 3 | 0.007316 | 0.006073 | 0.007316 | 0.027229 | 3 |
| recommended | spread_weighted | 206 | 3 | 0.007655 | 0.006613 | 0.007401 | 0.024268 | 3 |
| recommended | vega_weighted | 206 | 3 | 0.009874 | 0.005123 | 0.003528 | 0.047198 | 3 |

## Warning Summary

Base Stage 4 warning counts:

| Warning | Count |
|---|---:|
| alpha_near_bound | 2 |
| rho_near_bound | 1 |
| rho_near_plus_minus_one | 1 |
| nu_near_bound | 5 |
| high_nu_gt_5 | 6 |
| max_abs_error_gt_0_25 | 1 |
| optimizer_failure | 0 |
| too_few_points | 0 |

Recommended universe warning counts:

| Warning | Count |
|---|---:|
| alpha_near_bound | 0 |
| rho_near_bound | 0 |
| rho_near_plus_minus_one | 0 |
| nu_near_bound | 0 |
| high_nu_gt_5 | 3 |
| max_abs_error_gt_0_25 | 0 |
| optimizer_failure | 0 |
| too_few_points | 0 |

## Recommended Calibration Universe

Deterministic rule selected for later Heston/model-comparison work:

- exclude `T < 2 / 252`
- require `relative_spread <= 0.25`
- require `volume > 0`
- require `open_interest > 0`

Recommended universe summary:

| Metric | Value |
|---|---:|
| Base rows | 595 |
| Recommended rows | 206 |
| Rows removed | 389 |
| Expiries retained | 3 |
| Min expiry | 2026-07-16 |
| Max expiry | 2026-07-20 |
| Min time to expiry | 0.008642 |
| Max time to expiry | 0.019601 |
| OTM put count | 140 |
| ATM count | 31 |
| OTM call count | 35 |
| Median relative spread | 0.039050 |
| Median volume | 597.0 |
| Median open interest | 1495.5 |

## Generated Outputs

- `outputs/stage4_5_sabr_robustness/robustness_grid.csv`
- `outputs/stage4_5_sabr_robustness/robustness_summary.csv`
- `outputs/stage4_5_sabr_robustness/filtered_universe_summary.csv`
- `outputs/stage4_5_sabr_robustness/beta_sensitivity.csv`
- `outputs/stage4_5_sabr_robustness/weighting_sensitivity.csv`
- `outputs/stage4_5_sabr_robustness/expiry_filter_sensitivity.csv`
- `outputs/stage4_5_sabr_robustness/liquidity_filter_sensitivity.csv`
- `outputs/stage4_5_sabr_robustness/recommended_calibration_universe.csv`
- `outputs/stage4_5_sabr_robustness/warning_summary.csv`

Generated plots:

- `outputs/stage4_5_sabr_robustness/plots/rmse_by_expiry_across_expiry_filters.png`
- `outputs/stage4_5_sabr_robustness/plots/warning_counts_by_scenario.png`
- `outputs/stage4_5_sabr_robustness/plots/beta_sensitivity_rmse.png`
- `outputs/stage4_5_sabr_robustness/plots/weighting_sensitivity_rmse.png`
- `outputs/stage4_5_sabr_robustness/plots/raw_vs_recommended_residuals.png`

## Limitations

- Single point-in-time yfinance snapshot, not a vendor-certified market dataset.
- Not a historical backtest.
- No Heston.
- No model comparison against Heston.
- No live-data dependency in deterministic regeneration.
- No trading claims.
- The recommended universe is a deterministic calibration hygiene choice, not a
  claim that excluded quotes are wrong.
