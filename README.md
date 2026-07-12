# vol-surface-research-lab

A reproducible Python research engine for options volatility work.

The project is intentionally staged. Stage 0 creates the package, test, docs,
data, output, and notebook layout. Stage 1 implements Black-Scholes European
option pricing, Greeks, put-call parity checks, and implied-volatility inversion.
Stage 2 adds deterministic option-chain loading, quote cleaning, time-to-expiry,
put-call-parity forward extraction, and forward-based moneyness.
Stage 2 uses a deterministic SPY-like fixture to test the data pipeline.
Stage 3.5 adds a saved SPY yfinance snapshot for point-in-time market-data
diagnostics; calibration claims are based on the saved snapshot, not the fixture.
Stage 3 adds observed implied-volatility tables, OTM surface input selection,
and simple static-arbitrage diagnostics on filtered quotes.
Stage 3.5 adds a saved SPY yfinance option-chain snapshot so the same pipeline
can run on point-in-time market data without live downloads during regeneration.
Stage 4 calibrates fixed-beta per-expiry SABR smiles to the saved SPY snapshot
as a calibration diagnostic.
Stage 4.5 adds SABR robustness diagnostics across expiry filters, liquidity
filters, beta choices, and weighting modes, and writes a recommended
calibration universe for later Heston comparison.
Stage 5 adds global Heston pricing and calibration diagnostics on the Stage 4.5
recommended universe, then recalibrates SABR on that same universe for
point-in-time model comparison.

Planned later stages, not implemented yet, are production reporting polish,
historical validation, and optional arbitrage-aware interpolation.

## Layout

```text
src/volsurface/        Python package
tests/                 pytest suite
data/                  raw or local research data, ignored by default
outputs/               generated reference outputs and reports
notebooks/             exploratory notebooks
docs/                  design notes and stage documentation
scripts/               deterministic utility scripts
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Test

```bash
pytest
ruff check .
```

## Regenerate Stage 1 Outputs

```bash
python scripts/generate_stage1_outputs.py
```

The generated Stage 1 CSV files live in
`outputs/stage1_black_scholes/reference_values.csv` and
`outputs/stage1_black_scholes/iv_recovery_tests.csv`.

## Regenerate Stage 2 Outputs

```bash
python scripts/generate_stage2_outputs.py
```

The generated Stage 2 CSV files live under
`outputs/stage2_chain_cleaning/`.

## Regenerate Stage 3 Outputs

```bash
python scripts/generate_stage3_outputs.py
```

The generated Stage 3 CSV files live under `outputs/stage3_surface/`.

## Regenerate Stage 3.5 Outputs

```bash
python scripts/generate_stage3_real_snapshot_outputs.py
```

The optional live download script is separate:

```bash
python scripts/download_spy_snapshot.py
```

## Regenerate Stage 4 Outputs

```bash
python scripts/generate_stage4_sabr_outputs.py
```

## Regenerate Stage 4.5 Outputs

```bash
python scripts/generate_stage4_5_sabr_robustness_outputs.py
```

## Regenerate Stage 5 Outputs

```bash
python scripts/generate_stage5_heston_outputs.py
```
