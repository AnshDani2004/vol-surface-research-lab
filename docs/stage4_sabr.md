# Stage 4: SABR Calibration

## Scope

Stage 4 calibrates a per-expiry SABR smile approximation to observed OTM IVs
from a single SPY snapshot. The result is a calibration diagnostic, not a
trading signal or historical model validation.

This stage does not add Heston, model comparison, historical backtesting,
arbitrage-free interpolation, or calibration claims beyond the reported fit
diagnostics.

## Input Source

The calibration input is the Stage 3.5 saved SPY yfinance snapshot after the
existing cleaning, forward extraction, IV inversion, and OTM surface selection
pipeline:

- `outputs/stage3_real_snapshot/otm_surface_table.csv`
- Source: `yfinance`
- Retrieval timestamp: `2026-07-12T20:17:34.011027+00:00`
- Underlying: `SPY`
- OTM/ATM surface rows: `595`

The snapshot is a single point-in-time dataset and should not be interpreted as
a historical backtest.

## Formula Summary

`hagan_sabr_iv(F, K, T, alpha, beta, rho, nu)` implements the Hagan lognormal
SABR implied-volatility approximation. Inputs assume:

- positive forward `F`
- positive strike `K`
- positive expiry `T`
- `alpha > 0`
- `beta in [0, 1]`
- `rho in (-1, 1)`
- `nu >= 0`

Rates and dividends are assumed to have already been absorbed into the forward
used by the Stage 2/3 pipeline. The ATM case is handled separately through the
stable limiting form.

## Calibration Setup

- Beta: fixed at `0.5`
- Weighting: `spread_weighted`
- Optimizer: `scipy.optimize.least_squares`
- Alpha bounds: `[1e-4, 5.0]`
- Rho bounds: `[-0.999, 0.999]`
- Nu bounds: `[1e-4, 10.0]`
- Minimum points per expiry: `6`

No beta grid was generated in Stage 4.

## Actual Commands

Generated outputs:

```bash
python scripts/generate_stage4_sabr_outputs.py
```

Use the current top-level validation commands in `README.md` for package-wide
test and lint status.

## Calibration Summary

| Metric | Count |
|---|---:|
| Expiries attempted | 6 |
| Expiries calibrated successfully | 6 |
| Expiries skipped | 0 |
| Fit points | 595 |

Parameter estimates and fit quality:

| Expiry | T | Points | Alpha | Rho | Nu | RMSE | MAE | Weighted RMSE | Max Abs Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-13 | 0.000423 | 69 | 5.000000 | -0.999000 | 10.000000 | 0.364291 | 0.297518 | 0.380991 | 0.758247 |
| 2026-07-14 | 0.003163 | 102 | 5.000000 | -0.530310 | 10.000000 | 0.050445 | 0.038873 | 0.052098 | 0.178460 |
| 2026-07-15 | 0.005903 | 113 | 3.652951 | -0.403378 | 10.000000 | 0.020177 | 0.014932 | 0.021173 | 0.085668 |
| 2026-07-16 | 0.008642 | 28 | 2.768034 | -0.332840 | 10.000000 | 0.011831 | 0.009934 | 0.011478 | 0.023062 |
| 2026-07-17 | 0.011382 | 183 | 2.300818 | -0.359109 | 10.000000 | 0.017733 | 0.014443 | 0.016335 | 0.059228 |
| 2026-07-20 | 0.019601 | 100 | 2.197372 | -0.364128 | 7.174631 | 0.012705 | 0.011025 | 0.011731 | 0.028127 |

## Failure And Warning Summary

No expiries were skipped and no optimizer failures occurred. The failure report
is used as a warning report for suspicious or unstable fits.

| Warning | Count |
|---|---:|
| high nu > 5.0 | 6 |
| nu near calibration bound | 5 |
| alpha near calibration bound | 2 |
| rho near calibration bound | 1 |
| rho near +/-1 | 1 |
| max_abs_error > 0.25 | 1 |

The first expiry, `2026-07-13`, is especially unstable: `alpha`, `rho`, and
`nu` are all near bounds and the maximum absolute IV error is `0.758247`.

## Generated Outputs

- `outputs/stage4_sabr/sabr_params.csv`
- `outputs/stage4_sabr/sabr_fit_errors.csv`
- `outputs/stage4_sabr/sabr_fit_points.csv`
- `outputs/stage4_sabr/sabr_failure_report.csv`

Generated plots:

- `outputs/stage4_sabr/plots/market_vs_sabr_by_expiry.png`
- `outputs/stage4_sabr/plots/sabr_residuals_by_log_moneyness.png`
- `outputs/stage4_sabr/plots/sabr_error_by_expiry.png`

## Limitations

- Single point-in-time yfinance snapshot, not a vendor-certified market dataset.
- Not a historical backtest.
- Very short-dated expiries can be unstable under Hagan SABR calibration.
- Fixed beta at `0.5`; no beta grid comparison in this stage.
- Flat spread-weighted objective; no liquidity model or robust outlier model.
- No Heston.
- No model comparison.
- No arbitrage-free interpolation.
- No trading or profitability claims.
