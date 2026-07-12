# Stage 3: Surface Construction and Static-Arbitrage Diagnostics

## Scope

Stage 3 converts cleaned Stage 2 option-chain rows into observed
implied-volatility tables, selects OTM/ATM rows for surface construction input,
summarizes surface coverage by expiry, and checks simple static-arbitrage
violations on observed filtered quotes.

Stage 3 checks simple static-arbitrage violations on observed filtered quotes.
It does not construct a fully arbitrage-free interpolated surface.

No SABR, Heston, model calibration, model comparison, or plotting is included in
Stage 3.

## Inputs

The deterministic command reads `data/samples/sample_option_chain.csv`, runs the
Stage 2 cleaning and forward-extraction pipeline, and then computes Stage 3
tables.

The sample CSV is a deterministic SPY-like fixture for pipeline testing. It is
not live market data and is not a vendor-certified historical snapshot.

## IV Solver Method

`compute_implied_vol_table(df, risk_free_rate, dividend_yield=0.0)` uses the
Stage 1 bisection implied-volatility solver as the reference method. Every row
keeps `iv_status` and `iv_failure_reason`; failed solves are recorded without
crashing the pipeline.

Output columns are deterministic:

- `underlying`
- `quote_datetime`
- `expiry`
- `option_type`
- `strike`
- `mid`
- `bid`
- `ask`
- `spot`
- `forward`
- `time_to_expiry`
- `log_moneyness`
- `option_side_for_surface`
- `implied_vol`
- `iv_status`
- `iv_failure_reason`

## OTM Surface Rule

`build_otm_surface_table(iv_df)` keeps rows where:

- `iv_status == success`
- `option_side_for_surface` is one of `otm_put`, `atm`, or `otm_call`

The table is sorted by expiry, strike, and option type. It is an observed OTM
surface input table, not a smoothed or interpolated surface.

## Static-Arbitrage Diagnostics

Implemented diagnostics:

- `check_call_monotonicity`: call mids should be non-increasing in strike.
- `check_put_monotonicity`: put mids should be non-decreasing in strike.
- `check_call_convexity`: call mids should be convex in strike, using adjacent
  nonuniform strike slopes.
- `check_calendar_monotonicity`: same-type, same-strike mids should generally
  be non-decreasing with expiry under simplified assumptions.

Calendar monotonicity is explicitly a simplified diagnostic because dividends,
rates, American exercise, and sparse strikes can complicate interpretation.

## Actual Commands

Generated outputs:

```bash
python scripts/generate_stage3_outputs.py
```

Validated package:

```bash
python -m pytest
python -m ruff check .
```

Actual validation output:

```text
.....................................                                    [100%]
37 passed in 4.93s
```

```text
All checks passed!
```

## IV Counts

| IV Status | Count |
|---|---:|
| success | 18 |
| failed | 0 |

## OTM Surface Counts

| Surface Side | Count |
|---|---:|
| atm | 4 |
| otm_call | 4 |
| otm_put | 3 |
| total | 11 |

## Surface Summary

| Expiry | Raw Rows | Successful IV | Failed IV | OTM Rows | Min IV | Median IV | Max IV | ATM IV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024-04-19 16:00:00 | 6 | 6 | 0 | 4 | 0.093002 | 0.122533 | 0.147062 | 0.115250 |
| 2024-05-17 16:00:00 | 6 | 6 | 0 | 3 | 0.112305 | 0.142348 | 0.168539 |  |
| 2024-06-21 16:00:00 | 6 | 6 | 0 | 4 | 0.101358 | 0.175650 | 0.218829 | 0.179968 |

## Arbitrage Diagnostics

The filtered fixture produced zero simple static-arbitrage violations.

| Diagnostic | Violations |
|---|---:|
| call_monotonicity | 0 |
| put_monotonicity | 0 |
| call_convexity | 0 |
| calendar_monotonicity | 0 |

## Generated Outputs

- `outputs/stage3_surface/iv_table.csv`
- `outputs/stage3_surface/otm_surface_table.csv`
- `outputs/stage3_surface/arbitrage_report.csv`
- `outputs/stage3_surface/surface_summary.csv`

No optional plots were generated in Stage 3.

## Limitations

- Fixture data only, not a real vendor snapshot.
- Observed quote diagnostics only.
- No arbitrage-free interpolation yet.
- No SABR.
- No Heston.
- No model calibration claims.
