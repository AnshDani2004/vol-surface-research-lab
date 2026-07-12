"""Generate deterministic Stage 1 Black-Scholes reference outputs."""

from __future__ import annotations

import csv
from pathlib import Path

from volsurface.black_scholes import (
    black_scholes_greeks,
    black_scholes_price,
    implied_vol_bisection,
    implied_vol_newton,
    put_call_parity_residual,
)

OUTPUT_DIR = Path("outputs/stage1_black_scholes")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_reference_values()
    write_iv_recovery_tests()


def write_reference_values() -> None:
    rows = []
    for strike in (80.0, 90.0, 100.0, 110.0, 120.0):
        for time_to_expiry in (0.25, 0.5, 1.0):
            for volatility in (0.15, 0.25, 0.4):
                call = black_scholes_price(
                    100.0, strike, time_to_expiry, 0.03, volatility, "call", 0.01
                )
                put = black_scholes_price(
                    100.0, strike, time_to_expiry, 0.03, volatility, "put", 0.01
                )
                call_greeks = black_scholes_greeks(
                    100.0, strike, time_to_expiry, 0.03, volatility, "call", 0.01
                )
                put_greeks = black_scholes_greeks(
                    100.0, strike, time_to_expiry, 0.03, volatility, "put", 0.01
                )
                rows.append(
                    {
                        "spot": 100.0,
                        "strike": strike,
                        "time_to_expiry": time_to_expiry,
                        "rate": 0.03,
                        "dividend_yield": 0.01,
                        "volatility": volatility,
                        "call_price": call,
                        "put_price": put,
                        "call_delta": call_greeks.delta,
                        "put_delta": put_greeks.delta,
                        "gamma": call_greeks.gamma,
                        "vega": call_greeks.vega,
                        "call_theta": call_greeks.theta,
                        "put_theta": put_greeks.theta,
                        "call_rho": call_greeks.rho,
                        "put_rho": put_greeks.rho,
                        "parity_residual": put_call_parity_residual(
                            call, put, 100.0, strike, time_to_expiry, 0.03, 0.01
                        ),
                    }
                )
    _write_csv(OUTPUT_DIR / "reference_values.csv", rows)


def write_iv_recovery_tests() -> None:
    rows = []
    for option_type in ("call", "put"):
        for spot in (90.0, 100.0, 110.0):
            for strike in (90.0, 100.0, 110.0):
                for time_to_expiry in (0.25, 0.75, 1.5):
                    for true_volatility in (0.15, 0.25, 0.4, 0.6):
                        price = black_scholes_price(
                            spot,
                            strike,
                            time_to_expiry,
                            0.025,
                            true_volatility,
                            option_type,
                            0.005,
                        )
                        bisection_iv = implied_vol_bisection(
                            price,
                            spot,
                            strike,
                            time_to_expiry,
                            0.025,
                            option_type,
                            0.005,
                        )
                        newton_iv = implied_vol_newton(
                            price,
                            spot,
                            strike,
                            time_to_expiry,
                            0.025,
                            option_type,
                            0.005,
                            initial_volatility=0.3,
                        )
                        rows.append(
                            {
                                "option_type": option_type,
                                "spot": spot,
                                "strike": strike,
                                "time_to_expiry": time_to_expiry,
                                "rate": 0.025,
                                "dividend_yield": 0.005,
                                "true_volatility": true_volatility,
                                "price": price,
                                "bisection_iv": bisection_iv,
                                "newton_iv": newton_iv,
                                "abs_error_bisection": abs(
                                    bisection_iv - true_volatility
                                ),
                                "abs_error_newton": abs(newton_iv - true_volatility),
                            }
                        )
    _write_csv(OUTPUT_DIR / "iv_recovery_tests.csv", rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
