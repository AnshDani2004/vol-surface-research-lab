# Cross-Date Vol Surface Validation

## Scope

The cross-date extension tests whether the original SABR/Heston findings can be
rerun across a manifest of saved SPY option-chain snapshots. It is a
model-validation and robustness layer, not a trading strategy.

The current checked manifest contains two saved SPY yfinance snapshots. One is
usable for model comparison. The newer pull is retained as a failed low-quality
snapshot because deterministic cleaning left no same-strike call/put pairs for
forward extraction.

That means the current artifacts validate multi-snapshot mechanics and failure
preservation, but still do not prove that the original one-snapshot conclusion
generalizes across dates.

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

## Adding Saved Snapshots

Manual CSV drops can be validated and appended with:

```bash
python scripts/add_cross_date_snapshot.py \
  --snapshot-id spy_YYYY_MM_DD_HHMM \
  --raw-csv path/to/normalized_option_chain.csv \
  --metadata-json path/to/metadata.json \
  --market-regime saved_yfinance_snapshot \
  --notes "Saved SPY snapshot; not vendor-certified."
```

The command rejects duplicate `snapshot_id` values, missing files, malformed
metadata, and CSVs that cannot be normalized into the option-chain schema.

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
| Saved snapshots | 2 |
| Usable model-comparison snapshots | 1 |
| Total raw option rows | 2631 |
| Recommended calibration rows | 206 |
| SABR expiry wins vs Heston | 3 |
| Heston expiry wins vs SABR | 0 |
| Generalization status | insufficient snapshots |
| Conclusion stability | insufficient |

## Generated Artifacts

```text
artifacts/cross_date/snapshot_summary.csv
artifacts/cross_date/aggregate_summary.csv
artifacts/cross_date/sabr_cross_date_results.csv
artifacts/cross_date/heston_cross_date_results.csv
artifacts/cross_date/model_comparison_summary.csv
artifacts/cross_date/filter_sensitivity_summary.csv
artifacts/cross_date/snapshot_quality_report.csv
artifacts/cross_date/model_summary_by_snapshot.csv
artifacts/cross_date/cross_snapshot_aggregation.csv
artifacts/cross_date/filter_attrition_by_snapshot.csv
artifacts/cross_date/failed_snapshot_examples.csv
artifacts/cross_date/forward_pairing_audit.csv
artifacts/cross_date/failure_modes.md
artifacts/cross_date/cross_date_research_note.md
```

Processed per-snapshot tables are written under:

```text
data/processed_snapshots/
```

## Interpretation

The current run preserves the original usable-snapshot conclusion: on the saved
SPY snapshot that passes quality checks, SABR has lower RMSE than Heston on the
same filtered expiries. The second saved snapshot is a negative data-quality
result: it has 1462 raw rows but only 4 clean rows and 0 recommended rows after
deterministic filters.

The model-comparison artifact uses per-expiry Heston where available and
records global/fallback/failed scopes separately. Because only two saved
snapshots are checked into the manifest and only one is currently usable for
model comparison, this does not prove a universal model ranking or cross-date
robustness.

## Failed Snapshot Audit

The failed `spy_2026_07_13_0452` snapshot is classified as
`true_low_quality_snapshot`.

The audit did not find evidence of a schema-mapping, timezone, or call/put
pairing bug:

- Calls and puts both exist: 696 calls and 766 puts.
- Six expiries parse consistently.
- Strike values are numeric floats.
- Option labels are normalized to `call` and `put`.
- Raw call/put strike overlap exists by expiry before quote filtering.
- Future expiries remain positive after time-to-expiry parsing.

The failure is quote quality after deterministic filters:

| Stage | Rows Before | Rows After | Main Reason |
|---|---:|---:|---|
| Positive time to expiry | 1462 | 1207 | same-day expiry already expired at snapshot time |
| Valid bid/ask | 1207 | 1187 | missing bid values |
| Positive mid | 1187 | 4 | zero bid/ask quotes |
| Same-strike call/put pairs | 4 | 0 | no overlapping call/put strikes after filtering |
| Recommended universe | 0 | 0 | no surface rows |

Supporting artifacts:

```text
artifacts/cross_date/filter_attrition_by_snapshot.csv
artifacts/cross_date/failed_snapshot_examples.csv
artifacts/cross_date/forward_pairing_audit.csv
```

## Limitations

- Current checked manifest contains two saved snapshots, below the five-snapshot
  threshold for weak cross-date evidence.
- Only one saved snapshot is currently usable for model comparison.
- The failed snapshot is a genuine quote-quality failure under current filters,
  not evidence that the model result generalizes.
- yfinance data is not vendor-certified.
- Results depend on snapshot quality, maturity coverage, and filtering rules.
- This is not a trading strategy.
- This does not prove a universal model ranking.
- Additional usable saved snapshots are required for real cross-date validation.
