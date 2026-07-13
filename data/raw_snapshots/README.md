# Raw Snapshot Manifest

`snapshot_manifest.csv` registers saved option-chain snapshots for cross-date
validation. The pipeline is deterministic: it reads local CSV files listed in
the manifest and does not download live data.

To add another saved SPY snapshot, place the normalized option-chain CSV and
metadata JSON somewhere in the repo, then run:

```bash
python scripts/add_cross_date_snapshot.py \
  --snapshot-id spy_YYYY_MM_DD_HHMM \
  --raw-csv path/to/normalized_option_chain.csv \
  --metadata-json path/to/metadata.json \
  --market-regime saved_yfinance_snapshot \
  --notes "Saved SPY snapshot; not vendor-certified."
```

The manifest row contains:

- `snapshot_id`
- `underlying`
- `snapshot_datetime`
- `spot`
- `source`
- `raw_path`
- `metadata_path`
- `market_regime`
- `notes`

The current checked manifest contains two saved SPY yfinance snapshots. One is
usable for model comparison; the newer saved pull is retained as a low-quality
failed snapshot. The generated cross-date report is still below the
five-snapshot threshold for weak cross-date evidence.
