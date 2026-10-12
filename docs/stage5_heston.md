# Stage 5: Heston Calibration And Same-Universe Comparison

> Note (expiry-clock fix, 2026-10-12): the numbers in this document predate the expiry-clock fix (time to
> expiry now runs to the 16:00 New York close, not 00:00 UTC). Current numbers are in README.md and
> outputs/; old vs new is in outputs/audit/expiry_clock_sabr_summary.csv.


Stage 5 compares Heston and SABR diagnostics on the same filtered
single-snapshot universe. It is not a historical validation, trading signal, or
claim that either model is universally better.

## Input Universe

Stage 5 reads
`outputs/stage4_5_sabr_robustness/recommended_calibration_universe.csv`.
It does not calibrate Heston on the raw Stage 3.5 saved snapshot universe.

The same recommended universe is used for:

- one global Heston calibration;
- fixed-beta SABR recalibration with `beta=0.5`;
- by-expiry and by-moneyness comparison diagnostics.

## Heston Pricing

The implemented Heston process uses the forward as the pricing state:

```text
dF_t / F_t = sqrt(v_t) dW^F_t
dv_t = kappa(theta - v_t) dt + sigma sqrt(v_t) dW^v_t
corr(dW^F_t, dW^v_t) = rho
```

The calibrated parameters are:

- `v0`: initial variance;
- `theta`: long-run variance;
- `kappa`: variance mean-reversion speed;
- `sigma`: volatility of variance;
- `rho`: forward/variance correlation.

European calls are priced with the Heston characteristic-function integral and
deterministic Gauss-Laguerre quadrature. Puts are priced from forward put-call
parity. Heston prices are converted back to Black-Scholes implied volatility by
deterministic bisection.

## Calibration Setup

The Heston objective minimizes weighted implied-volatility residuals over the
same Stage 4.5 recommended universe used for the SABR comparison.

Parameter bounds:

| Parameter | Lower | Upper |
|---|---:|---:|
| `v0` | 0.0001 | 4.0 |
| `theta` | 0.0001 | 4.0 |
| `kappa` | 0.001 | 50.0 |
| `sigma` | 0.0001 | 10.0 |
| `rho` | -0.999 | 0.999 |

Diagnostics include optimizer status, runtime, price/IV conversion failure
counts, parameter-bound hits, the Feller condition, suspicious parameter
patterns, and comparison underperformance warnings.

## Generated Outputs

```text
outputs/stage5_heston/heston_params.csv
outputs/stage5_heston/heston_fit_points.csv
outputs/stage5_heston/heston_fit_errors.csv
outputs/stage5_heston/heston_failure_report.csv
outputs/stage5_heston/sabr_recommended_fit_errors.csv
outputs/stage5_heston/model_comparison_by_expiry.csv
outputs/stage5_heston/model_comparison_by_moneyness.csv
outputs/stage5_heston/runtime_summary.csv
outputs/stage5_heston/plots/
```

## Test And Generation Commands

Generation command:

```bash
python scripts/generate_stage5_heston_outputs.py
```

Use the current top-level validation commands in `README.md` for package-wide
test and lint status.

## Measured Heston Parameters

| `v0` | `theta` | `kappa` | `sigma` | `rho` | Optimizer | Feller |
|---:|---:|---:|---:|---:|---|---|
| 0.028684 | 0.000389 | 17.436712 | 0.380554 | -0.457068 | success, `ftol` | false |

The fitted `theta` is near its lower bound, the Feller condition is not
satisfied, and `v0/theta` is flagged as suspicious. These are diagnostics, not
reasons to discard the table.

## Measured Error Summary

| Scope | Expiry | Points | Heston RMSE IV | Heston MAE IV | Weighted RMSE IV | Max Abs Error |
|---|---|---:|---:|---:|---:|---:|
| global | all | 206 | 0.117433 | 0.074092 | 0.127041 | 0.364102 |
| expiry | 2026-07-16 | 16 | 0.125951 | 0.088676 | 0.135935 | 0.261791 |
| expiry | 2026-07-17 | 106 | 0.145687 | 0.096644 | 0.156281 | 0.364102 |
| expiry | 2026-07-20 | 84 | 0.063358 | 0.042856 | 0.067124 | 0.278257 |

No Heston price failures or implied-volatility conversion failures were
observed in this deterministic run.

## Same-Universe SABR Versus Heston

| Expiry | Points | SABR RMSE IV | Heston RMSE IV | Winner By RMSE |
|---|---:|---:|---:|---|
| 2026-07-16 | 16 | 0.005636 | 0.125951 | SABR |
| 2026-07-17 | 106 | 0.009108 | 0.145687 | SABR |
| 2026-07-20 | 84 | 0.007655 | 0.063358 | SABR |

By moneyness bucket:

| Bucket | Points | SABR RMSE IV | Heston RMSE IV | Winner By RMSE |
|---|---:|---:|---:|---|
| left_wing | 67 | 0.008644 | 0.183797 | SABR |
| left_mid | 65 | 0.006033 | 0.080035 | SABR |
| atm | 48 | 0.010705 | 0.039286 | SABR |
| right_mid | 26 | 0.007214 | 0.057860 | SABR |
| right_wing | 0 | n/a | n/a | not available |

## Limitations

- Results are point-in-time diagnostics from one saved SPY snapshot, not a
  historical validation.
- Heston is calibrated globally while SABR is recalibrated per expiry, so the
  comparison is useful diagnostic context but not a broad model-ranking claim.
- The saved snapshot and recommended universe are deterministic local artifacts;
  regeneration does not download live market data.
- American exercise, discrete dividends, early exercise, and historical
  backtesting are out of scope.
- Arbitrage-aware interpolation remains optional later work and is not
  prioritized ahead of SABR/Heston diagnostics.
