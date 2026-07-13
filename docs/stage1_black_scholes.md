# Stage 1: Black-Scholes

## Scope

Stage 1 implements deterministic European Black-Scholes analytics:

- Call and put pricing.
- Delta, gamma, vega, theta, and rho.
- Put-call parity residuals and tolerance checks.
- Implied-volatility inversion by bisection.
- Implied-volatility inversion by safeguarded Newton steps.

No SABR, Heston, real option-chain ingestion, volatility-surface construction,
static-arbitrage diagnostics, or plotting is included in Stage 1.

## Assumptions

- European exercise.
- Continuously compounded risk-free rate `r`.
- Continuous dividend yield `q`.
- Constant volatility `sigma`.
- No transaction costs, funding frictions, or early exercise.
- Annualized time-to-expiry `T`.

## Pricing Formulas

For spot `S`, strike `K`, expiry `T`, rate `r`, dividend yield `q`, and volatility
`sigma`:

```text
d1 = [ln(S / K) + (r - q + 0.5 sigma^2) T] / [sigma sqrt(T)]
d2 = d1 - sigma sqrt(T)
```

European call:

```text
C = S exp(-qT) N(d1) - K exp(-rT) N(d2)
```

European put:

```text
P = K exp(-rT) N(-d2) - S exp(-qT) N(-d1)
```

Put-call parity residual:

```text
residual = C - P - [S exp(-qT) - K exp(-rT)]
```

## Greeks

The package returns:

```text
call_delta = exp(-qT) N(d1)
put_delta  = exp(-qT) [N(d1) - 1]
gamma      = exp(-qT) n(d1) / [S sigma sqrt(T)]
vega       = S exp(-qT) n(d1) sqrt(T)
call_rho   = K T exp(-rT) N(d2)
put_rho    = -K T exp(-rT) N(-d2)
```

Theta follows the market convention `dV/dcalendar_time`, equal to the negative
derivative with respect to time-to-expiry.

## Validation

Use the current top-level validation commands in `README.md` for package-wide
test and lint status. Stage 1 numerical reference outputs were generated with:

```bash
python scripts/generate_stage1_black_scholes_outputs.py
```

## Measured Numerical Errors

The implied-volatility recovery grid was generated into
`outputs/stage1_black_scholes/iv_recovery_tests.csv` with 216 deterministic
cases.

| Check | Max Absolute Error |
|---|---:|
| IV recovery, bisection | 1.0343115253164115e-12 |
| IV recovery, safeguarded Newton | 3.7581049383561549e-14 |
| Finite-difference Greek validation | 5.3546101241863653e-07 |

Finite-difference Greek validation details:

| Option Type | Greek | Absolute Error |
|---|---|---:|
| call | delta | 5.3546101241863653e-07 |
| call | gamma | 4.9412542391102932e-09 |
| call | vega | 1.2806474813942259e-07 |
| call | theta | 9.7122043740682784e-10 |
| call | rho | 2.1551613826886751e-09 |
| put | delta | 5.3546098199852565e-07 |
| put | gamma | 4.9405844831929002e-09 |
| put | vega | 1.281002752762106e-07 |
| put | theta | 3.2386626713787336e-10 |
| put | rho | 2.9055371442154865e-09 |

## Known numerical limitations

- IV inversion can become ill-conditioned for options with near-zero extrinsic
  value.
- Newton steps are safeguarded, but bisection remains the fallback reference
  method.
- Stage 1 tests use deterministic generated prices, not market quotes.
- American exercise, discrete dividends, and early exercise are out of scope.

## Regenerate Outputs

```bash
python scripts/generate_stage1_outputs.py
```
