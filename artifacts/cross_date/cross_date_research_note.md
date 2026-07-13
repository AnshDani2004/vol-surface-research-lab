# Cross-Date Vol Surface Validation

## Motivation

The original project was intentionally honest about its main limitation: most real-data conclusions came from one saved SPY option-chain snapshot. This extension adds a deterministic multi-snapshot validation layer.

## Data Snapshots Used

- Saved snapshots processed: 1
- Total raw option rows processed: 1169
- Total recommended calibration rows: 206

## Methodology

For each saved snapshot, the pipeline applies the existing quote cleaning, forward extraction, OTM IV surface construction, recommended expiry/liquidity filtering, SABR calibration, and global and per-expiry Heston calibration where feasible.

## Filtering Rules

The recommended calibration universe keeps expiries with at least two trading days to expiry, relative spread <= 0.25, positive volume, and positive open interest.

## SABR Results

- Median recommended-universe SABR RMSE: 0.007655

## Heston Results

- Median global Heston RMSE: 0.117464
- Median per-expiry Heston RMSE: 0.113647

## Cross-Date Comparison

- SABR expiry wins: 3
- Heston expiry wins: 0
- Generalization status: The current repository has only one saved SPY snapshot, so this run does not prove cross-date generalization.

## Failure Modes

The failure-mode report records non-calibrated SABR rows, Heston optimizer/Feller/parameter warnings, and same-universe Heston underfit cases.

## Limitations

- This is not a trading strategy.
- This does not prove a universal model ranking.
- Results depend on snapshot quality, maturity coverage, and filtering rules.
- Additional saved snapshots are required for real cross-date claims.

## What This Does And Does Not Prove

This extension proves the pipeline can run deterministically over a manifest of saved snapshots. With the current one-snapshot manifest, it does not prove that the SABR/Heston conclusion generalizes.

## Resume Bullet Candidates

- Conservative QR bullet: Extended a reproducible SPY vol-surface engine to support multi-snapshot validation; current checked manifest processes 1 saved snapshot and 1169 option rows, so cross-date generalization remains future work.
- Stronger QR bullet if results generalize: After adding more saved snapshots, report the number of snapshots, total rows, SABR/Heston RMSE distributions, and whether the same-universe conclusion persists.
- Honest negative-result bullet if mixed: Built a cross-date SABR/Heston validation framework that preserved failed calibrations and showed model conclusions were snapshot-dependent rather than universal.
