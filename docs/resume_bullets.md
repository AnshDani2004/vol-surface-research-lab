# Resume Bullets

## A. One-Bullet Version

- Built a reproducible Python volatility-surface research engine for SPY
  option-chain ingestion, quote filtering, forward extraction, IV surface
  construction, static-arbitrage diagnostics, SABR/Heston calibration, and
  same-universe model comparison.

## B. Two-Bullet Version

- Built a reproducible Python options research engine for Black-Scholes pricing,
  Greeks, implied-vol inversion, SPY option-chain ingestion, quote filtering,
  forward extraction, OTM IV surface construction, static-arbitrage diagnostics,
  and SABR/Heston calibration.
- Diagnosed model instability across ultra-short expiries, liquidity filters,
  beta choices, weighting modes, and Heston variants; selected a deterministic
  filtered universe that reduced SABR median RMSE from 0.0190 to 0.0077, then
  showed global/per-expiry Heston underfit SABR despite synthetic Heston
  recovery and implementation sanity checks.

## C. Quant-Dev / Numerical-Methods Version

- Engineered a package-first Python volatility research lab with deterministic
  scripts, pytest coverage, ruff checks, generated CSV outputs, Black-Scholes
  Greeks/IV solvers, Heston characteristic-function pricing, Gauss-Laguerre
  quadrature diagnostics, and reproducible calibration reports.
- Implemented numerical diagnostics for implied-vol inversion, finite-difference
  Greeks, Heston-to-Black-Scholes limit behavior, synthetic Heston recovery,
  quadrature sensitivity, optimizer multistart behavior, and soft
  Feller/regularization penalties.

## D. Quant-Research Version

- Built an end-to-end volatility-surface research pipeline for a saved SPY
  yfinance snapshot, including quote cleaning, forward extraction, OTM surface
  construction, static-arbitrage diagnostics, SABR robustness testing, Heston
  calibration, and same-universe model comparison.
- Reduced SABR median RMSE from 0.0190 to 0.0077 through deterministic
  expiry/liquidity filtering, then documented why Heston variants underfit the
  same recommended calibration universe without claiming a universal model
  ranking or trading signal.
