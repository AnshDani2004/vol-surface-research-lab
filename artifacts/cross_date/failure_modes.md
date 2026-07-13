# Cross-Date Failure Modes

## Scope

Processed 2 saved snapshot(s). This report is diagnostic and not a trading signal.

## SABR Calibration Failures Or Warnings

- Non-calibrated SABR rows: 2

## Heston Calibration Failures Or Warnings

- Heston warning/failure rows: 8

## Same-Universe Underfit Cases

- Expiries where Heston underfit SABR: 3

## Interpretation

With the currently saved data, the cross-date layer validates the pipeline mechanics but remains below the threshold for cross-date evidence. Add more saved snapshots to `data/raw_snapshots/snapshot_manifest.csv` to test additional dates and regimes.
