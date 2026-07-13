# Interview Defense

## 30-Second Explanation

I built a reproducible Python options research engine that takes a saved SPY
yfinance option-chain snapshot through quote cleaning, forward extraction,
observed IV surface construction, static-arbitrage diagnostics, SABR
calibration, Heston calibration, robustness testing, and final reporting. The
main finding is diagnostic: after filtering, per-expiry SABR fit this
short-dated snapshot much better than Heston variants, while Heston sanity
checks passed.

## 2-Minute Explanation

The project is staged like a small research platform. Stage 1 implements and
tests Black-Scholes, Greeks, parity, and IV inversion. Stages 2 and 3 clean
option-chain rows, estimate forwards, build OTM IV surface inputs, and run
static-arbitrage diagnostics. Stage 3.5 freezes a saved SPY yfinance snapshot so
regeneration is deterministic. Stages 4 and 4.5 calibrate SABR and stress-test
it across expiry filters, liquidity filters, beta choices, and weighting modes.
Stages 5 and 5.5 implement Heston pricing/calibration and investigate why Heston
underfit SABR on the same recommended calibration universe.

The important part is not a model ranking. The important part is that the repo
records data cleaning, model diagnostics, robustness checks, numerical failure
modes, and limitations.

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

## How Do You Know The Heston Implementation Was Not Broken?

Stage 5.5 added Heston-to-Black-Scholes limit checks and synthetic Heston
recovery. Price errors in the Black-Scholes limit were small, and synthetic
Heston recovery reached RMSE IV `0.000043` with recovered parameters close to
the generating parameters. That does not prove every numerical detail is
perfect, but it argues against a basic broken implementation.

## Why Is The SABR-vs-Heston Comparison Not Universal?

It uses one saved SPY yfinance snapshot, a short-dated filtered universe, and a
specific calibration setup. A universal comparison would require many dates,
underlyings, maturities, data vendors, and market regimes.

## What Were The Biggest Numerical Risks?

Implied-vol inversion near zero extrinsic value, Heston quadrature sensitivity,
optimizer local minima, parameter-bound behavior, and Feller-condition
instability.

## Why Did You Filter Ultra-Short Expiries?

Ultra-short expiries produced noisy calibration behavior and warnings. Excluding
the shortest maturities and applying liquidity filters reduced SABR median RMSE
from `0.018955` to `0.007655` and reduced warning count from `16` to `3`.

## Why Use yfinance If It Is Not Vendor-Certified?

The goal was to build a reproducible research engine on accessible data, not to
make institutional data claims. The saved yfinance snapshot is explicitly
documented as a single point-in-time snapshot with limitations.

## What Would You Do With Better Data?

I would validate across many dates, underlyings, maturities, and regimes; add
vendor-certified quote fields; improve dividend and American-exercise handling;
and compare calibration stability across richer datasets.

## What Would You Improve Next?

Historical validation, independent Heston pricing checks, stronger numerical
controls for quadrature, and optional arbitrage-aware interpolation.

## What Should Not Be Overclaimed?

This is not a trading signal, not a universal model ranking, not evidence that
SABR is always better than Heston, and not a replacement for vendor-certified
historical data validation.
