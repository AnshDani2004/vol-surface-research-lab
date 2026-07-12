# Stage 3.5: Saved Real Market Snapshot

## Scope

Stage 3.5 introduces a saved market snapshot so later calibration stages are not
based only on the deterministic fixture. The snapshot is still a single
point-in-time dataset and should not be interpreted as a historical backtest.

This stage runs the existing Stage 2 and Stage 3 pipeline on a saved SPY
option-chain snapshot. It does not add SABR, Heston, arbitrage-free
interpolation, optimization routines, model comparison, or calibration claims.

## Data Source

- Underlying: `SPY`
- Source: `yfinance`
- Retrieval method: `python scripts/download_spy_snapshot.py`
- Retrieval timestamp: `2026-07-12T20:17:34.011027+00:00`
- Spot from source: `754.9500122070312`
- Expiries: `2026-07-13`, `2026-07-14`, `2026-07-15`, `2026-07-16`,
  `2026-07-17`, `2026-07-20`
- Market-data timing: not certified live; yfinance data may be delayed

The saved deterministic input is:

- `data/snapshots/spy_option_chain_snapshot.csv`
- `data/snapshots/spy_option_chain_snapshot_metadata.json`

Deterministic regeneration uses the checked CSV snapshot, not a live download.

## Schema Normalization

`src/volsurface/market_data.py` normalizes calls and puts into the existing
Stage 2 required schema:

- `underlying`
- `quote_datetime`
- `expiry`
- `option_type`
- `strike`
- `bid`
- `ask`
- `last`
- `volume`
- `open_interest`
- `spot`

The optional downloader combines yfinance calls and puts across the first six
available SPY expiries. Missing bid, ask, or last prices are represented as
missing values so deterministic filters can reject them. Missing volume and
open interest are represented as zero.

## Actual Commands

Saved snapshot:

```bash
python scripts/download_spy_snapshot.py
```

Generated deterministic outputs:

```bash
python scripts/generate_stage3_real_snapshot_outputs.py
```

Validated package:

```bash
python -m pytest
python -m ruff check .
```

Actual validation output:

```text
.........................................                                [100%]
41 passed in 5.36s
```

```text
All checks passed!
```

## Output Counts

| Metric | Count |
|---|---:|
| Raw contracts | 1169 |
| Cleaned contracts | 957 |
| Forward rows | 6 |
| IV successes | 943 |
| IV failures | 14 |
| OTM surface rows | 595 |

Filter report:

| Reason | Count |
|---|---:|
| kept | 957 |
| missing_required_fields | 5 |
| negative_bid | 0 |
| ask_below_bid | 0 |
| zero_or_negative_ask | 5 |
| zero_or_negative_mid | 0 |
| expired_contract | 0 |
| extremely_wide_relative_spread | 202 |
| nonpositive_strike | 0 |
| nonpositive_spot | 0 |

Forward by expiry:

| Expiry | Time To Expiry | Forward | Paired Strikes |
|---|---:|---:|---:|
| 2026-07-13 | 0.000423 | 755.078169 | 48 |
| 2026-07-14 | 0.003163 | 755.162303 | 46 |
| 2026-07-15 | 0.005903 | 755.232655 | 48 |
| 2026-07-16 | 0.008642 | 755.300745 | 12 |
| 2026-07-17 | 0.011382 | 755.513121 | 158 |
| 2026-07-20 | 0.019601 | 755.612714 | 37 |

IV status:

| Status | Count |
|---|---:|
| success | 943 |
| failed | 14 |

OTM surface rows:

| Surface Side | Count |
|---|---:|
| atm | 75 |
| otm_call | 160 |
| otm_put | 360 |
| total | 595 |

Static-arbitrage diagnostics:

| Diagnostic | Violations |
|---|---:|
| calendar_monotonicity | 10 |
| call_convexity | 125 |
| call_monotonicity | 2 |
| put_monotonicity | 1 |

## Generated Outputs

- `outputs/stage3_real_snapshot/clean_chain.csv`
- `outputs/stage3_real_snapshot/filter_report.csv`
- `outputs/stage3_real_snapshot/forward_by_expiry.csv`
- `outputs/stage3_real_snapshot/iv_table.csv`
- `outputs/stage3_real_snapshot/otm_surface_table.csv`
- `outputs/stage3_real_snapshot/arbitrage_report.csv`
- `outputs/stage3_real_snapshot/surface_summary.csv`
- `outputs/stage3_real_snapshot/snapshot_metadata.json`

## Limitations

- The snapshot is a single point-in-time yfinance pull, not a vendor-certified
  dataset.
- yfinance data may be delayed and may not match exchange-quality historical
  records.
- The snapshot is not a historical backtest.
- Diagnostics are observed-quote checks only and do not prove the surface is
  arbitrage-free.
- The risk-free rate is a flat input of `0.05`, not a bootstrapped curve.
- No arbitrage-free interpolation.
- No SABR.
- No Heston.
- No model calibration claims.
