from __future__ import annotations

import math

import pytest

from volsurface.black_scholes import (
    black_scholes_greeks,
    black_scholes_price,
    check_put_call_parity,
    implied_vol_bisection,
    implied_vol_newton,
    put_call_parity_residual,
)


def test_atm_reference_values_without_dividends() -> None:
    call = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, "call")
    put = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, "put")
    call_greeks = black_scholes_greeks(100.0, 100.0, 1.0, 0.05, 0.2, "call")
    put_greeks = black_scholes_greeks(100.0, 100.0, 1.0, 0.05, 0.2, "put")

    assert call == pytest.approx(10.450583572185565)
    assert put == pytest.approx(5.573526022256971)
    assert call_greeks.delta == pytest.approx(0.6368306511756191)
    assert put_greeks.delta == pytest.approx(-0.3631693488243809)
    assert call_greeks.gamma == pytest.approx(0.018762017345846895)
    assert call_greeks.vega == pytest.approx(37.52403469169379)
    assert call_greeks.theta == pytest.approx(-6.414027546438197)
    assert put_greeks.theta == pytest.approx(-1.657880423934626)
    assert call_greeks.rho == pytest.approx(53.232481545376345)
    assert put_greeks.rho == pytest.approx(-41.89046090469506)


def test_put_call_parity() -> None:
    call = black_scholes_price(100.0, 95.0, 0.75, 0.03, 0.22, "call", 0.01)
    put = black_scholes_price(100.0, 95.0, 0.75, 0.03, 0.22, "put", 0.01)

    residual = put_call_parity_residual(call, put, 100.0, 95.0, 0.75, 0.03, 0.01)

    assert residual == pytest.approx(0.0, abs=1e-12)
    assert check_put_call_parity(call, put, 100.0, 95.0, 0.75, 0.03, 0.01)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_greeks_match_finite_differences(option_type: str) -> None:
    spot = 103.0
    strike = 97.0
    time_to_expiry = 0.8
    rate = 0.025
    volatility = 0.31
    dividend_yield = 0.01
    greeks = black_scholes_greeks(
        spot, strike, time_to_expiry, rate, volatility, option_type, dividend_yield
    )

    ds = 1e-3 * spot
    dv = 1e-4
    dr = 1e-5
    dt = 1e-5
    price_up = black_scholes_price(
        spot + ds, strike, time_to_expiry, rate, volatility, option_type, dividend_yield
    )
    price_mid = black_scholes_price(
        spot, strike, time_to_expiry, rate, volatility, option_type, dividend_yield
    )
    price_down = black_scholes_price(
        spot - ds, strike, time_to_expiry, rate, volatility, option_type, dividend_yield
    )
    delta_fd = (price_up - price_down) / (2.0 * ds)
    gamma_fd = (price_up - 2.0 * price_mid + price_down) / (ds**2)

    vega_fd = (
        black_scholes_price(
            spot,
            strike,
            time_to_expiry,
            rate,
            volatility + dv,
            option_type,
            dividend_yield,
        )
        - black_scholes_price(
            spot,
            strike,
            time_to_expiry,
            rate,
            volatility - dv,
            option_type,
            dividend_yield,
        )
    ) / (2.0 * dv)
    rho_fd = (
        black_scholes_price(
            spot,
            strike,
            time_to_expiry,
            rate + dr,
            volatility,
            option_type,
            dividend_yield,
        )
        - black_scholes_price(
            spot,
            strike,
            time_to_expiry,
            rate - dr,
            volatility,
            option_type,
            dividend_yield,
        )
    ) / (2.0 * dr)
    theta_fd = -(
        black_scholes_price(
            spot,
            strike,
            time_to_expiry + dt,
            rate,
            volatility,
            option_type,
            dividend_yield,
        )
        - black_scholes_price(
            spot,
            strike,
            time_to_expiry - dt,
            rate,
            volatility,
            option_type,
            dividend_yield,
        )
    ) / (2.0 * dt)

    assert greeks.delta == pytest.approx(delta_fd, rel=2e-5, abs=2e-5)
    assert greeks.gamma == pytest.approx(gamma_fd, rel=5e-4, abs=5e-6)
    assert greeks.vega == pytest.approx(vega_fd, rel=2e-7, abs=2e-7)
    assert greeks.theta == pytest.approx(theta_fd, rel=2e-7, abs=2e-7)
    assert greeks.rho == pytest.approx(rho_fd, rel=2e-7, abs=2e-7)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_implied_volatility_recovery(option_type: str) -> None:
    max_bisection_error = 0.0
    max_newton_error = 0.0
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
                    max_bisection_error = max(
                        max_bisection_error, abs(bisection_iv - true_volatility)
                    )
                    max_newton_error = max(
                        max_newton_error, abs(newton_iv - true_volatility)
                    )

    assert max_bisection_error < 1e-8
    assert max_newton_error < 1e-8


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((0.0, 100.0, 1.0, 0.05, 0.2, "call"), "spot must be positive"),
        ((100.0, -1.0, 1.0, 0.05, 0.2, "call"), "strike must be positive"),
        ((100.0, 100.0, -0.1, 0.05, 0.2, "call"), "time_to_expiry"),
        ((100.0, 100.0, 1.0, math.nan, 0.2, "call"), "rate must be finite"),
        ((100.0, 100.0, 1.0, 0.05, -0.2, "call"), "volatility"),
        ((100.0, 100.0, 1.0, 0.05, 0.2, "straddle"), "option_type"),
    ],
)
def test_invalid_pricing_inputs_raise_value_error(
    args: tuple[float, float, float, float, float, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        black_scholes_price(*args)


def test_implied_volatility_rejects_arbitrage_violating_price() -> None:
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_vol_bisection(101.0, 100.0, 100.0, 1.0, 0.02, "call")


def test_expiry_price_is_intrinsic() -> None:
    assert black_scholes_price(100.0, 90.0, 0.0, 0.05, 0.2, "call") == pytest.approx(
        10.0
    )
    assert black_scholes_price(100.0, 90.0, 0.0, 0.05, 0.2, "put") == pytest.approx(0.0)


def test_zero_volatility_price_uses_discounted_forward_intrinsic() -> None:
    call = black_scholes_price(100.0, 95.0, 1.0, 0.04, 0.0, "call", 0.01)
    expected = max(100.0 * math.exp(-0.01) - 95.0 * math.exp(-0.04), 0.0)

    assert call == pytest.approx(expected)
