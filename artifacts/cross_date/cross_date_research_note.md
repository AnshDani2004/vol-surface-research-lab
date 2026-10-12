# Cross-Date Vol Surface Validation

## Motivation

The original project was intentionally honest about its main limitation: most real-data conclusions came from one saved SPY option-chain snapshot. This extension adds a deterministic multi-snapshot validation layer.

## Data Snapshots Used

- Saved snapshots processed: 5
- Total raw option rows processed: 7017
- Total recommended calibration rows: 273
- Snapshots usable for model comparison: 1

## Methodology

For each saved snapshot, the pipeline applies the existing quote cleaning, forward extraction, OTM IV surface construction, recommended expiry/liquidity filtering, SABR calibration, and global and per-expiry Heston calibration where feasible.

## Filtering Rules

The recommended calibration universe keeps expiries with at least two trading days to expiry, relative spread <= 0.25, positive volume, and positive open interest.

## SABR Results

- Median recommended-universe SABR RMSE: 0.006035

## Heston Results

- Median global Heston RMSE: 0.108180
- Median per-expiry Heston RMSE: 0.103543

## Cross-Date Comparison

- SABR expiry wins: 4
- Heston expiry wins: 0
- Generalization status: insufficient_snapshots
- Conclusion stability: insufficient
- Interpretation: The run has 5 saved snapshot(s) and 1 usable snapshot(s), so it remains below the minimum threshold for cross-date evidence.

## Per-Snapshot Model Summary

| snapshot_id | sabr_median_rmse | heston_median_rmse | sabr_expiry_wins | heston_expiry_wins | available_comparison_expiries | winner_summary |
| --- | --- | --- | --- | --- | --- | --- |
| spy_2026_07_12 | 0.006035 | 0.103543 | 4 | 0 | 4 | sabr_only |
| spy_2026_07_13_0452 | n/a | n/a | 0 | 0 | 0 | not_available |
| spy_2026_07_13_0556 | n/a | n/a | 0 | 0 | 0 | not_available |
| spy_2026_07_13_0556_02 | n/a | n/a | 0 | 0 | 0 | not_available |
| spy_2026_07_13_0556_03 | n/a | n/a | 0 | 0 | 0 | not_available |

## Failure Modes

The failure-mode report records non-calibrated SABR rows, Heston optimizer/Feller/parameter warnings, and same-universe Heston underfit cases.

## Limitations

- This is not a trading strategy.
- This does not support a broad model-ranking claim.
- Results depend on snapshot quality, maturity coverage, and filtering rules.
- Additional saved snapshots are required for real cross-date claims.

## Interpretation Boundaries

This extension shows the pipeline can run deterministically over a manifest of saved snapshots. The evidence label is based on saved and usable snapshot counts and should not be read as a universal model-ranking claim.
