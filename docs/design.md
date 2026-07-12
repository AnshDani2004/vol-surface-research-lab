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

### Stage 4.5: SABR Robustness And Calibration Hygiene

Implemented scope:

- Expiry-filter sensitivity for ultra-short maturities.
- Liquidity-filter sensitivity for relative spread, volume, and open interest.
- Fixed-beta sensitivity for beta values `0.0`, `0.5`, and `1.0`.
- Weighting sensitivity for unweighted, spread-weighted, and vega-weighted
  residuals.
- Warning classification and a deterministic recommended calibration universe.

Explicitly out of scope for Stage 4.5:

- Heston calibration.
- SABR versus Heston model comparison.
- Historical backtesting.
- Trading claims.

### Stage 5: Heston Calibration And Same-Universe Comparison

Implemented scope:

- Heston characteristic function and European call/put pricing.
- Global five-parameter Heston calibration on
  `outputs/stage4_5_sabr_robustness/recommended_calibration_universe.csv`.
- Heston price-to-IV conversion and point-level fit diagnostics.
- Optimizer status, runtime, parameter-bound, Feller-condition, suspicious
  parameter, and failed point diagnostics.
- SABR recalibration on the exact same recommended universe.
- Same-universe comparison tables by expiry and log-moneyness bucket.
- Deterministic CSV outputs and diagnostic plots under `outputs/stage5_heston`.

Explicitly out of scope for Stage 5:

- Historical validation.
- Trading signals.
- Live market data dependency.
- Heston claims on raw Stage 3.5 rows outside the recommended universe.
- Arbitrage-aware interpolation.

### Stage 5.5: Heston Improvement And Fairness Diagnostics

Implemented scope:

- Heston-to-Black-Scholes limit sanity checks.
- Synthetic Heston surface generation and recovery.
- Gauss-Laguerre quadrature sensitivity diagnostics.
- Deterministic multi-start global Heston calibration subset.
- Soft Feller-penalized and mild regularized Heston variants.
- Per-expiry Heston calibration as a fairness diagnostic against per-expiry
  SABR.
- Variant comparison tables by expiry and log-moneyness bucket.
- Failure reports, improvement summary, and deterministic plots under
  `outputs/stage5_5_heston_improvement`.

Explicitly out of scope for Stage 5.5:

- Rough volatility.
- Local volatility.
- Neural networks.
- Historical backtesting.
- Trading signals.
- Live data dependency.
- New option-chain downloads.
- Forcing Heston to beat SABR.

## Determinism

Stage 1 uses closed-form Black-Scholes formulas and fixed parameter grids.
There is no randomness in the package functions, tests, or generated reference
CSV files.
