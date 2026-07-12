# vol-surface-research-lab

A reproducible Python research engine for options volatility work.

The project is intentionally staged. Stage 0 creates the package, test, docs,
data, output, and notebook layout. Stage 1 implements Black-Scholes European
option pricing, Greeks, put-call parity checks, and implied-volatility inversion.
Stage 2 adds deterministic option-chain loading, quote cleaning, time-to-expiry,
put-call-parity forward extraction, and forward-based moneyness.
Stage 2 uses a deterministic SPY-like fixture to test the data pipeline. Real
market-chain ingestion is planned before final model comparison and no
calibration claims are made from the fixture.

Planned later stages, not implemented yet, are implied-volatility surface
construction, static-arbitrage diagnostics, and SABR/Heston calibration.

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
