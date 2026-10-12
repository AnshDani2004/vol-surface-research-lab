# Technical FAQ

> Note (expiry-clock fix, 2026-10-12): the numbers in this document predate the expiry-clock fix (time to
> expiry now runs to the 16:00 New York close, not 00:00 UTC). Current numbers are in README.md and
> outputs/; old vs new is in outputs/audit/expiry_clock_sabr_summary.csv.


## Project Overview

`vol-surface-research-lab` is a reproducible Python research engine for a saved
SPY yfinance option-chain snapshot. It covers quote cleaning, forward
extraction, observed implied-volatility surface construction, static-arbitrage
diagnostics, SABR calibration, Heston calibration, robustness checks, and final
reporting. The project is designed as a deterministic research artifact, not a
trading signal.

## What Is The Main Finding?

On a saved short-dated SPY yfinance snapshot, per-expiry SABR fit the filtered
local smile better than global or per-expiry Heston variants. Heston sanity
checks and synthetic recovery passed, so the underfit appears to come from
model/setup limitations and numerical sensitivity rather than a basic
implementation failure.

## Why SABR?

SABR is a standard smile model that can be calibrated per expiry. It is useful
for local smile fitting and for testing how sensitive calibration is to expiry
selection, liquidity filters, beta choice, and weighting.

## Why Heston?

Heston is a classic stochastic-volatility model with a characteristic-function
pricing formula. It is a natural comparison point because it has richer dynamics
than Black-Scholes and a different parameter structure from per-expiry SABR.

## Why Did Heston Underperform?

The filtered snapshot is short-dated and local-smile dominated. Global Heston
has to fit multiple expiries with one parameter set, while SABR is calibrated
per expiry. Per-expiry Heston improved Heston fit, which suggests global-model
rigidity mattered, but it still underfit SABR on this snapshot.

## How Do The Heston Sanity Checks Help?

Stage 5.5 added Heston-to-Black-Scholes limit checks and synthetic Heston
recovery. Price errors in the Black-Scholes limit were small, and synthetic
Heston recovery reached RMSE IV `0.000043` with recovered parameters close to
the generating parameters. These checks do not prove every numerical detail is
perfect, but they argue against a basic broken implementation.

## Why Is The SABR-vs-Heston Comparison Not Universal?

The current manifest contains five saved SPY yfinance pulls, but only one is
usable for same-universe model comparison. Four failed pulls have near-duplicate
low-quality quote structure under the fingerprint audit, so they are
data-quality diagnostics rather than independent model evidence. A universal
comparison would require many usable dates, underlyings, maturities, data
vendors, and market regimes.

## What Were The Biggest Numerical Risks?

The main numerical risks were implied-vol inversion near zero extrinsic value,
Heston quadrature sensitivity, optimizer local minima, parameter-bound behavior,
and Feller-condition instability.

## Why Filter Ultra-Short Expiries?

Ultra-short expiries produced noisy calibration behavior and warnings. Excluding
the shortest maturities and applying liquidity filters reduced SABR median RMSE
from `0.018955` to `0.007655` and reduced warning count from `16` to `3`.

## Why Use yfinance?

The goal was to build a reproducible research engine on accessible data, not to
make institutional data claims. Saved yfinance snapshots are explicitly
documented as point-in-time pulls with quality gates and limitations.

## What Would Improve This With Better Data?

Better data would allow validation across many dates, underlyings, maturities,
and regimes. It would also support vendor-certified quote fields, improved
dividend and American-exercise handling, and richer calibration stability
analysis.

## What Should Not Be Overclaimed?

This is not a trading signal, not a broad model-ranking claim, not evidence that
SABR is always better than Heston, and not a replacement for vendor-certified
historical data validation.

## What Does The Cross-Date Extension Add?

The cross-date extension adds a saved-snapshot manifest and deterministic
pipeline for running the same cleaning, filtering, SABR calibration, Heston
calibration, and same-universe comparison across multiple local SPY snapshots.
It records global and per-expiry Heston scopes separately and preserves failed
calibrations in the output tables.
The current manifest has five saved pulls, but four later pulls are retained as
near-duplicate low-quality failed snapshots. The generated cross-date artifacts
validate the workflow, collection gate, and failure preservation, but do not
prove cross-date generalization.
