# Vol Surface Research Lab

A reproducible Python options research engine for SPY option-chain cleaning, IV
surface construction, static-arbitrage diagnostics, SABR calibration, Heston
calibration, robustness testing, and failure-mode analysis.

## Key Result

On a saved SPY option-chain snapshot, raw SABR calibration was unstable due to
ultra-short expiries and noisy quotes. After deterministic expiry/liquidity
filtering, per-expiry SABR fit the recommended universe well. Global and
per-expiry Heston variants underfit SABR on the same filtered snapshot, despite
passing synthetic Heston recovery and implementation sanity checks. The project
is a reproducible calibration and diagnostics framework, not a trading signal or
universal model ranking.

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
| Tests | 101 passed |
| Raw snapshot rows | 1169 |
| Clean rows | 957 |
| OTM surface rows | 595 |
| Recommended universe rows | 206 |
| Recommended universe expiries | 3 |
| SABR median RMSE before filtering | 0.018955 |
| SABR median RMSE after filtering | 0.007655 |
| Heston global RMSE | 0.117433 |
| Synthetic Heston recovery RMSE | 0.000043 |
| Quadrature sensitivity 32 vs 96 node mean IV diff | 0.052438 |

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
python scripts/generate_final_report_outputs.py
python -m pytest
python -m ruff check .
```

## Repository Map

```text
src/volsurface/        Python package
tests/                 pytest suite
data/snapshots/        saved SPY yfinance snapshot
outputs/               deterministic generated reports
docs/                  research note and portfolio docs
scripts/               output regeneration scripts
notebooks/             optional exploration area
```

## Final Report

- [Research note](docs/research_note.md)
- [Project summary](docs/project_summary.md)
- [Website case study](docs/website_case_study.md)
- [Technical FAQ](docs/technical_faq.md)
- [Final report outputs](outputs/final_report/)

## Limitations

- Single saved yfinance SPY snapshot.
- Not vendor-certified data.
- No historical validation.
- No trading signal.
- American exercise and dividends are not fully modeled.
- Heston result is diagnostic, not a deployment-ready calibration.
- Quadrature sensitivity requires caution.
- SABR and Heston comparison is not universal.

Remaining optional work: historical validation and arbitrage-aware interpolation.
