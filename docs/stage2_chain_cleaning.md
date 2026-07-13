# Stage 2: Option-Chain Cleaning and Forward Extraction

## Scope

Stage 2 builds deterministic package infrastructure for loading an option-chain
snapshot, normalizing fields, filtering bad quotes, computing mid prices and
ACT/365 time-to-expiry, extracting forwards by expiry from put-call parity, and
adding forward-based moneyness.

No implied-volatility surface construction, SABR, Heston, static-arbitrage
diagnostics, or plotting is included in Stage 2.

## Data Source

The checked input file is
`data/samples/sample_option_chain.csv`.

It is a realistic SPY-like sample snapshot designed for deterministic package
tests and generated outputs. It is not live market data and should not be
treated as a historical quote from an exchange or vendor. A live option-chain
download is intentionally not part of Stage 2 output regeneration because that
would make the stage non-reproducible.

## Normalized Schema

Required input columns:

- `underlying`
- `quote_datetime`
- `expiry`
- `option_type`
- `strike`
- `bid`
- `ask`
- `last`
- `volume`
- `open_interest`
- `spot`

`load_option_chain(path)` normalizes common aliases into this schema, parses
`quote_datetime` and `expiry`, maps option types to `call` or `put`, converts
numeric quote fields, and returns a deterministic column order.

## Derived Fields

- `mid = (bid + ask) / 2`
- `bid_ask_spread = ask - bid`
- `relative_spread = bid_ask_spread / mid` when `mid > 0`
- `time_to_expiry = (expiry - quote_datetime).total_seconds() / 365 days`
- `forward` is merged by expiry after put-call-parity extraction
- `log_moneyness = log(strike / forward)`
- `option_side_for_surface` labels `atm`, `otm_put`, `otm_call`, or
  `not_surface_otm`

Forward-based moneyness is used instead of spot moneyness so later calibration
stages work in the natural expiry-specific forward measure.

## Filter Rules

Rows are assigned the first failing reason in this deterministic order:

1. `missing_required_fields`
2. `negative_bid`
3. `ask_below_bid`
4. `zero_or_negative_ask`
5. `zero_or_negative_mid`
6. `expired_contract`
7. `extremely_wide_relative_spread`
8. `nonpositive_strike`
9. `nonpositive_spot`

The default maximum relative spread threshold is `1.5`. A zero bid is allowed
when the ask is positive and the resulting mid is positive, unless the row is
filtered by the relative-spread threshold.

## Forward Extraction

Calls and puts are paired by expiry and strike. For each pair:

```text
C - P = exp(-rT) * (F - K)
F = K + exp(rT) * (C - P)
```

The Stage 2 output script uses mid prices, `risk_free_rate = 0.05`, and
spread-weighted averaging across same-expiry pairs.

## Actual Commands

Generated outputs:

```bash
python scripts/generate_stage2_outputs.py
```

Use the current top-level validation commands in `README.md` for package-wide
test and lint status.

## Output Counts

| Metric | Count |
|---|---:|
| Raw contracts | 26 |
| Cleaned contracts | 18 |

Filter report:

| Reason | Count |
|---|---:|
| kept | 18 |
| missing_required_fields | 1 |
| negative_bid | 1 |
| ask_below_bid | 1 |
| zero_or_negative_ask | 1 |
| zero_or_negative_mid | 0 |
| expired_contract | 1 |
| extremely_wide_relative_spread | 1 |
| nonpositive_strike | 1 |
| nonpositive_spot | 1 |

Forward by expiry:

| Expiry | Time To Expiry | Forward | Paired Strikes |
|---|---:|---:|---:|
| 2024-04-19 16:00:00 | 0.095919 | 516.008157 | 3 |
| 2024-05-17 16:00:00 | 0.172631 | 517.099941 | 3 |
| 2024-06-21 16:00:00 | 0.268522 | 518.504331 | 3 |

## Generated Outputs

- `outputs/stage2_chain_cleaning/clean_chain.csv`
- `outputs/stage2_chain_cleaning/filter_report.csv`
- `outputs/stage2_chain_cleaning/forward_by_expiry.csv`

## Limitations

- The sample CSV is deterministic fixture data, not a vendor-certified market
  snapshot.
- The risk-free rate is a single user-supplied continuously compounded rate,
  not a bootstrapped curve.
- Dividends are not separately modeled; forwards are inferred from put-call
  parity.
- American exercise and early exercise are out of scope.
- Surface construction and arbitrage diagnostics start after Stage 2.
