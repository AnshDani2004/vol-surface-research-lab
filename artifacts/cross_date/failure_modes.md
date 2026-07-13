# Cross-Date Failure Modes

## Scope

Processed 1 saved snapshot(s). This report is diagnostic and not a trading signal.

## SABR Calibration Failures Or Warnings

- Non-calibrated SABR rows: 0

## Heston Calibration Failures Or Warnings

- Heston warning/failure rows: 7

## Same-Universe Underfit Cases

- Expiries where Heston underfit SABR: 3

## Interpretation

With the currently saved data, the cross-date layer validates the pipeline mechanics but cannot establish cross-date generalization. Add more saved snapshots to `data/raw_snapshots/snapshot_manifest.csv` to test additional dates and regimes.
