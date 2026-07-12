# Design

## Goal

`vol-surface-research-lab` is a reproducible options volatility research engine
for real option-chain cleaning, implied-volatility surface construction,
static-arbitrage diagnostics, and SABR/Heston calibration.

The project is built as a normal Python package rather than a notebook-only
experiment. Notebooks may be used for exploration, but core logic belongs under
`src/volsurface`, with deterministic tests and generated outputs that can be
recreated from source.

## Stages

### Stage 0: Research Package Foundation

- Package source under `src/volsurface`.
- Tests under `tests`.
- Generated artifacts under `outputs`.
- Documentation under `docs`.
- Optional raw/local data under `data`.
- Optional exploratory notebooks under `notebooks`.
- Tooling through `pyproject.toml`, pytest, and ruff.

### Stage 1: Black-Scholes Baseline

Implemented scope:

- European call and put pricing.
- Greeks for delta, gamma, vega, theta, and rho.
- Put-call parity residuals and tolerance checks.
- Implied-volatility inversion with bisection and Newton methods.
- Deterministic reference CSV outputs.
- Finite-difference Greek validation tests.
- Implied-volatility recovery tests.

Explicitly out of scope for Stage 1:

- SABR calibration.
- Heston calibration.
- Real option-chain ingestion or cleaning.
- Volatility-surface construction.
- Static-arbitrage diagnostics.
- Plotting.

### Stage 2: Chain Ingestion, Cleaning, and Forwards

Implemented scope:

- CSV option-chain loading with schema normalization.
- Deterministic mid-price, bid-ask spread, and relative-spread fields.
- ACT/365 time-to-expiry calculation.
- Quote filters for missing fields, crossed markets, invalid prices, expired
  contracts, wide spreads, bad strikes, and bad spots.
- Put-call-parity forward extraction by expiry.
- Forward-based log-moneyness and OTM surface-side labels.

Explicitly out of scope for Stage 2:

- Implied-volatility surface construction.
- Static-arbitrage diagnostics.
- SABR calibration.
- Heston calibration.
- Plotting.

### Stage 3: Observed Surface Tables and Diagnostics

Implemented scope:

- Implied-volatility inversion on cleaned option-chain rows.
- OTM/ATM observed surface input table selection.
- Per-expiry surface coverage summaries.
- Simple observed-quote static-arbitrage diagnostics for call monotonicity, put
  monotonicity, call convexity, and simplified calendar monotonicity.

Explicitly out of scope for Stage 3:

- Fully arbitrage-free interpolated surface construction.
- SABR calibration.
- Heston calibration.
- Model calibration or comparison claims.
- Real vendor-certified market-chain ingestion.

### Stage 4: SABR Calibration Diagnostics

Implemented scope:

- Hagan lognormal SABR implied-volatility approximation.
- Fixed-beta per-expiry SABR calibration on the Stage 3.5 saved SPY snapshot.
- Spread-weighted residuals, fit error tables, parameter tables, warning reports,
  and deterministic plots.

Explicitly out of scope for Stage 4:

- Heston calibration.
- Model comparison.
- Historical backtesting.
- Trading signals.
- Arbitrage-free interpolation.

## Determinism

Stage 1 uses closed-form Black-Scholes formulas and fixed parameter grids.
There is no randomness in the package functions, tests, or generated reference
CSV files.
