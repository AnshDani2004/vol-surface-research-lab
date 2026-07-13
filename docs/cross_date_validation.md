# Cross-Date Vol Surface Validation

## Scope

The cross-date extension tests whether the original SABR/Heston findings can be
rerun across a manifest of saved SPY option-chain snapshots. It is a
model-validation and robustness layer, not a trading strategy.

The current checked manifest contains one saved SPY yfinance snapshot. That
means the current artifacts validate the multi-snapshot pipeline mechanics but
do not prove that the original one-snapshot conclusion generalizes across
dates.

## Data Layer

Snapshot metadata lives in:

```text
data/raw_snapshots/snapshot_manifest.csv
```

Each manifest row includes:

- `snapshot_id`
- `underlying`
- `snapshot_datetime`
- `spot`
- `source`
- `raw_path`
- `metadata_path`
- `market_regime`
- `notes`

The pipeline runs only from saved local CSV files. It does not download live
data.

## Methodology

For each saved snapshot, the cross-date pipeline applies:

1. Existing option-chain loading.
2. Mid-price and time-to-expiry calculation.
3. Existing quote cleaning filters.
4. Put-call-parity forward extraction.
5. OTM implied-volatility surface construction.
6. Existing deterministic recommended-universe filters.
7. Per-expiry SABR calibration.
8. Global and per-expiry Heston calibration where feasible.
9. Same-universe SABR/Heston comparison.
10. Failure-mode and limitation reporting.

## Current Checked Output

The current manifest processes:

| Metric | Value |
|---|---:|
| Saved snapshots | 1 |
| Total raw option rows | 1169 |
| Recommended calibration rows | 206 |
| SABR expiry wins vs Heston | 3 |
| Heston expiry wins vs SABR | 0 |
| Generalization status | insufficient snapshots |

## Generated Artifacts

```text
artifacts/cross_date/snapshot_summary.csv
artifacts/cross_date/aggregate_summary.csv
artifacts/cross_date/sabr_cross_date_results.csv
artifacts/cross_date/heston_cross_date_results.csv
artifacts/cross_date/model_comparison_summary.csv
artifacts/cross_date/filter_sensitivity_summary.csv
artifacts/cross_date/failure_modes.md
artifacts/cross_date/cross_date_research_note.md
```

Processed per-snapshot tables are written under:

```text
data/processed_snapshots/
```

## Interpretation

The current run preserves the original single-snapshot conclusion: on the saved
SPY snapshot, SABR has lower RMSE than Heston on the same filtered expiries.
The model-comparison artifact uses per-expiry Heston where available and records
global/fallback scopes separately. However, because only one saved snapshot is
currently checked into the manifest, this does not prove a universal model
ranking or cross-date robustness.

## Limitations

- Current checked manifest contains one saved snapshot.
- yfinance data is not vendor-certified.
- Results depend on snapshot quality, maturity coverage, and filtering rules.
- This is not a trading strategy.
- This does not prove a universal model ranking.
- Additional saved snapshots are required for real cross-date validation.
