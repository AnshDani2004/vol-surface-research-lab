# Raw Snapshot Manifest

`snapshot_manifest.csv` registers saved option-chain snapshots for cross-date
validation. The pipeline is deterministic: it reads local CSV files listed in
the manifest and does not download live data.

To add another saved SPY snapshot, place the normalized option-chain CSV and
metadata JSON somewhere in the repo, then append a row with:

- `snapshot_id`
- `underlying`
- `snapshot_datetime`
- `spot`
- `source`
- `raw_path`
- `metadata_path`
- `market_regime`
- `notes`

The current checked manifest contains one saved SPY yfinance snapshot, so the
generated cross-date report is a pipeline validation and not yet evidence of
cross-date generalization.
