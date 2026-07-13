# Vol Surface Research Lab: Technical Case Study

## Subtitle

A reproducible Python research engine for option-chain cleaning,
implied-volatility surfaces, SABR/Heston calibration, and model failure
analysis.

## Problem

Options volatility models often look clean on generated data but become fragile
on real option chains. This project asks what happens when a full calibration
workflow is forced to handle a saved SPY yfinance snapshot with noisy quotes,
ultra-short expiries, and model diagnostics recorded at every step.

## Approach

The project is a package-first Python research engine with deterministic
scripts, pytest coverage, ruff checks, generated CSV outputs, and documentation.
The pipeline cleans option-chain rows, extracts forwards, builds OTM IV surface
inputs, runs observed-quote static-arbitrage diagnostics, calibrates SABR and
Heston, and compares models only on the same recommended calibration universe.

## What Failed

Raw SABR calibration was unstable on ultra-short expiries. Global Heston
underfit the filtered snapshot and produced suspicious parameter diagnostics.
Feller penalties did not fully stabilize Heston inside the deterministic
runtime cap, and the Stage 5 fitted Heston parameters were quadrature-sensitive.

## What Improved

Deterministic expiry and liquidity filtering reduced SABR median RMSE from
`0.018955` to `0.007655`. Heston synthetic recovery worked with RMSE IV
`0.000043`, and per-expiry Heston improved Heston fit versus global Heston,
though it still underfit SABR.

## Final Finding

On a single point-in-time saved SPY yfinance snapshot, per-expiry SABR fit the
recommended calibration universe better than global or per-expiry Heston
variants. This is a diagnostic finding, not a trading signal or universal model
ranking. The cross-date extension processes five saved yfinance pulls, but only
one is usable for model comparison; four later pulls have near-duplicate
low-quality quote structure. The broader generalization claim remains
unsupported.

## Technical Stack

Python, NumPy, SciPy, pandas, matplotlib, pytest, ruff, package source under
`src/volsurface`, deterministic scripts, CSV outputs, and markdown research
documentation.

## Key Metrics

| Metric | Value |
|---|---:|
| Tests | 121 passed |
| Raw snapshot rows | 1169 |
| Clean rows | 957 |
| OTM surface rows | 595 |
| Recommended universe rows | 206 |
| SABR median RMSE after filtering | 0.007655 |
| Heston global RMSE | 0.117433 |
| Synthetic Heston recovery RMSE | 0.000043 |

## Links

- `README.md`
- `docs/research_note.md`
- `outputs/final_report/`
- `outputs/stage5_5_heston_improvement/`
