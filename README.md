# Vol Surface Research Lab

A reproducible Python options research engine for SPY option-chain cleaning, IV
surface construction, static-arbitrage diagnostics, SABR calibration, Heston
calibration, robustness testing, and failure-mode analysis. The repository is a
package-first research artifact with deterministic scripts, tests, saved input
data, and generated outputs.

## Key Result

On the one saved SPY option-chain snapshot that passes quality checks, raw SABR
calibration was unstable due to ultra-short expiries and noisy quotes. After
deterministic expiry/liquidity filtering, per-expiry SABR median RMSE improved
from `0.018955` to `0.007655`. Global and per-expiry Heston variants underfit
SABR on the same filtered snapshot, despite passing synthetic Heston recovery
and implementation sanity checks. This is a reproducible calibration and
diagnostics framework, not a trading signal or broad model-ranking claim.

## What This Project Proves

- Black-Scholes pricing, Greeks, and IV inversion were implemented and tested.
- Real SPY option-chain snapshot ingestion and deterministic regeneration were
  added.
- Quote filtering, forward extraction, and OTM IV surface construction were
  implemented.
- Static-arbitrage diagnostics were run on observed quotes.
- SABR calibration was stress-tested across expiry filters, liquidity filters,
  beta choices, and weighting modes.
- Heston pricing/calibration was implemented and compared on the same filtered
  universe.
- Heston underperformance was investigated through sanity checks, synthetic
  recovery, quadrature sensitivity, multistart, Feller/regularized variants, and
  per-expiry diagnostics.

## Headline Metrics

| Metric | Value |
|---|---:|
| Tests | 121 passed |
| Raw snapshot rows | 1169 |
| Clean rows | 957 |
| OTM surface rows | 595 |
| Recommended universe rows | 206 |
| Recommended universe expiries | 3 |
| SABR median RMSE before filtering | 0.018955 |
| SABR median RMSE after filtering | 0.007655 |
| Heston global RMSE, Stage 5 single-snapshot path | 0.117433 |
| Synthetic Heston recovery RMSE | 0.000043 |
| Quadrature sensitivity 32 vs 96 node mean IV diff | 0.052438 |

## Cross-Date Vol Surface Validation

The cross-date extension adds a manifest-driven pipeline for running the same
cleaning, filtering, SABR calibration, Heston calibration, and same-universe
comparison across multiple saved SPY option-chain snapshots.
The comparison artifact uses per-expiry Heston fits where available and records
global, per-expiry, fallback, and failed calibration scopes explicitly.

The current checked manifest contains five saved SPY yfinance pulls. The first
snapshot is usable for model comparison; four later pulls are preserved as
low-quality failed snapshots because quote cleaning left no same-strike call/put
pairs for forward extraction. The fingerprint audit shows the failed pulls are
near-duplicate stale quote structures: they have different raw file hashes and
timestamps, but the same spot, expiries, strike set, row counts, positive-mid
count, and post-filter call/put overlap. They are retained as data-quality
diagnostics, not independent evidence.

The failed pulls are true low-quality quote snapshots under the current filters,
not schema, timezone, or call/put pairing bugs. Each contains both calls and
puts across six expiries, but only four rows have positive mids after bid/ask
checks, leaving zero overlapping call/put strikes after filtering.

This is not a trading strategy, does not support a broad model-ranking claim, and
depends on snapshot quality, maturity coverage, and filtering rules. All
cross-date outputs are reproducible from saved data. Existing normalized CSV
snapshots can be added with:

```bash
python scripts/add_cross_date_snapshot.py \
  --snapshot-id spy_YYYY_MM_DD_HHMM \
  --raw-csv path/to/normalized_option_chain.csv \
  --metadata-json path/to/metadata.json \
  --market-regime saved_yfinance_snapshot \
  --notes "Saved SPY snapshot; not vendor-certified."
```

Live SPY yfinance snapshots can be collected and quality-gated with:

```bash
python scripts/collect_spy_snapshot.py \
  --market-regime active_market_hours \
  --notes "Saved SPY snapshot; yfinance is not vendor-certified."
```

The collector saves the raw CSV and metadata locally without overwriting. It
appends the manifest only if pre-ingestion checks pass unless
`--allow-low-quality` is explicit. Tests never download live data.

Current cross-date artifact summary:

| Metric | Value |
|---|---:|
| Saved snapshots processed | 5 |
| Usable model-comparison snapshots | 1 |
| Total raw rows processed | 7,017 |
| Total recommended calibration rows | 206 |
| SABR expiry wins vs Heston | 3 |
| Heston expiry wins vs SABR | 0 |
| Median global Heston RMSE, cross-date path | 0.117464 |
| Generalization status | insufficient snapshots |
| Conclusion stability | insufficient |

The Stage 5 single-snapshot path reports Heston global RMSE `0.117433`; the
cross-date regeneration path reports median global Heston RMSE `0.117464` under
its saved-manifest execution settings. Both are single-usable-snapshot
diagnostics and should not be read as cross-date model evidence.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python scripts/generate_stage3_real_snapshot_outputs.py
python scripts/generate_stage4_sabr_outputs.py
python scripts/generate_stage4_5_sabr_robustness_outputs.py
python scripts/generate_stage5_heston_outputs.py
python scripts/generate_stage5_5_heston_improvement_outputs.py
python scripts/generate_cross_date_outputs.py
python scripts/generate_final_report_outputs.py
python -m pytest
python -m ruff check .
```

## Repository Map

```text
src/volsurface/        Python package
tests/                 pytest suite
data/snapshots/        saved SPY yfinance snapshot
data/raw_snapshots/    cross-date snapshot manifest
data/processed_snapshots/ generated per-snapshot processed tables
outputs/               deterministic generated reports
artifacts/cross_date/  cross-date validation artifacts
docs/                  research notes and technical documentation
scripts/               output regeneration scripts
notebooks/             optional exploration area
```

## Final Report

- [Research note](docs/research_note.md)
- [Project summary](docs/project_summary.md)
- [Technical case study](docs/technical_case_study.md)
- [Technical FAQ](docs/technical_faq.md)
- [Cross-date validation note](docs/cross_date_validation.md)
- [Final report outputs](outputs/final_report/)

## Limitations

- Five saved yfinance SPY snapshots; one currently usable for model comparison.
- Not vendor-certified data.
- No historical validation.
- No trading signal.
- American exercise and dividends are not fully modeled.
- Heston result is diagnostic, not a deployment-ready calibration.
- Quadrature sensitivity requires caution.
- SABR and Heston comparison is not universal.

Remaining optional work: historical validation and arbitrage-aware interpolation.
