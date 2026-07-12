"""Black-Scholes pricing, Greeks, parity checks, and implied volatility."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class BlackScholesGreeks:
    """First-order Black-Scholes sensitivities for a European option."""

    delta: float | np.ndarray
    gamma: float | np.ndarray
    vega: float | np.ndarray
    theta: float | np.ndarray
    rho: float | np.ndarray


def black_scholes_price(
    spot: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    option_type: OptionType,
    dividend_yield: ArrayLike = 0.0,
) -> float | np.ndarray:
    """Price a European call or put with the Black-Scholes formula.

    `theta` and discounting use annualized continuously compounded rates.
    Inputs may be scalars or broadcast-compatible NumPy arrays.
    """

    _validate_option_type(option_type)
    s, k, t, r, sigma, q = _broadcast_inputs(
        spot, strike, time_to_expiry, rate, volatility, dividend_yield
    )
    _validate_price_inputs(s, k, t, sigma)
    _validate_finite("rate", r)
    _validate_finite("dividend_yield", q)

    forward_intrinsic = _discounted_intrinsic(s, k, t, r, q, option_type)
    regular = (t > 0.0) & (sigma > 0.0)
    price = np.array(forward_intrinsic, dtype=float)

    if np.any(regular):
        d1, d2 = _d1_d2(
            s[regular],
            k[regular],
            t[regular],
            r[regular],
            sigma[regular],
            q[regular],
        )
        discounted_spot = s[regular] * np.exp(-q[regular] * t[regular])
        discounted_strike = k[regular] * np.exp(-r[regular] * t[regular])
        if option_type == "call":
            price[regular] = discounted_spot * norm.cdf(d1) - (
                discounted_strike * norm.cdf(d2)
            )
        else:
            price[regular] = discounted_strike * norm.cdf(-d2) - (
                discounted_spot * norm.cdf(-d1)
            )

    return _as_scalar_if_scalar(price)


def black_scholes_greeks(
    spot: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    option_type: OptionType,
    dividend_yield: ArrayLike = 0.0,
) -> BlackScholesGreeks:
    """Return delta, gamma, vega, theta, and rho for a European option.

    Theta is the common market convention `dV/dcalendar_time`, i.e. the negative
    derivative with respect to time-to-expiry.
    """

    _validate_option_type(option_type)
    s, k, t, r, sigma, q = _broadcast_inputs(
        spot, strike, time_to_expiry, rate, volatility, dividend_yield
    )
    _validate_price_inputs(s, k, t, sigma)
    _validate_finite("rate", r)
    _validate_finite("dividend_yield", q)
    if np.any(t <= 0.0):
        raise ValueError("time_to_expiry must be positive for Greeks")
    if np.any(sigma <= 0.0):
        raise ValueError("volatility must be positive for Greeks")

    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    pdf_d1 = norm.pdf(d1)
    discounted_spot_factor = np.exp(-q * t)
    discounted_strike_factor = np.exp(-r * t)
    sqrt_t = np.sqrt(t)

    gamma = discounted_spot_factor * pdf_d1 / (s * sigma * sqrt_t)
    vega = s * discounted_spot_factor * pdf_d1 * sqrt_t

    if option_type == "call":
        delta = discounted_spot_factor * norm.cdf(d1)
        theta = (
            -(s * discounted_spot_factor * pdf_d1 * sigma) / (2.0 * sqrt_t)
            - r * k * discounted_strike_factor * norm.cdf(d2)
            + q * s * discounted_spot_factor * norm.cdf(d1)
        )
        rho = k * t * discounted_strike_factor * norm.cdf(d2)
    else:
        delta = discounted_spot_factor * (norm.cdf(d1) - 1.0)
        theta = (
            -(s * discounted_spot_factor * pdf_d1 * sigma) / (2.0 * sqrt_t)
            + r * k * discounted_strike_factor * norm.cdf(-d2)
            - q * s * discounted_spot_factor * norm.cdf(-d1)
        )
        rho = -k * t * discounted_strike_factor * norm.cdf(-d2)

    return BlackScholesGreeks(
        delta=_as_scalar_if_scalar(delta),
        gamma=_as_scalar_if_scalar(gamma),
        vega=_as_scalar_if_scalar(vega),
        theta=_as_scalar_if_scalar(theta),
        rho=_as_scalar_if_scalar(rho),
    )


def put_call_parity_residual(
    call_price: ArrayLike,
    put_price: ArrayLike,
    spot: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    rate: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
) -> float | np.ndarray:
    """Return `call - put - (S exp(-qT) - K exp(-rT))`."""

    call, put, s, k, t, r, q = np.broadcast_arrays(
        np.asarray(call_price, dtype=float),
        np.asarray(put_price, dtype=float),
        np.asarray(spot, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(time_to_expiry, dtype=float),
        np.asarray(rate, dtype=float),
        np.asarray(dividend_yield, dtype=float),
    )
    _validate_finite("call_price", call)
    _validate_finite("put_price", put)
    _validate_price_inputs(s, k, t, np.ones_like(s))
    _validate_finite("rate", r)
    _validate_finite("dividend_yield", q)
    residual = call - put - (s * np.exp(-q * t) - k * np.exp(-r * t))
    return _as_scalar_if_scalar(residual)


def check_put_call_parity(
    call_price: ArrayLike,
    put_price: ArrayLike,
    spot: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    rate: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
    tolerance: float = 1e-10,
) -> bool:
    """Return whether call and put prices satisfy put-call parity."""

    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be a finite non-negative number")
    residual = put_call_parity_residual(
        call_price, put_price, spot, strike, time_to_expiry, rate, dividend_yield
    )
    return bool(np.all(np.abs(residual) <= tolerance))


def implied_vol_bisection(
    price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    lower: float = 1e-8,
    upper: float = 5.0,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> float:
    """Invert Black-Scholes implied volatility with bisection."""

    _validate_iv_request(
        price,
        spot,
        strike,
        time_to_expiry,
        rate,
        option_type,
        dividend_yield,
        tolerance,
        max_iterations,
    )
    if not (isfinite(lower) and isfinite(upper)) or lower <= 0.0 or upper <= lower:
        raise ValueError("bisection bounds must satisfy 0 < lower < upper")

    target = float(price)
    low = float(lower)
    high = float(upper)
    low_value = (
        _scalar_price(
            spot, strike, time_to_expiry, rate, low, option_type, dividend_yield
        )
        - target
    )
    high_value = (
        _scalar_price(
            spot, strike, time_to_expiry, rate, high, option_type, dividend_yield
        )
        - target
    )

    if abs(low_value) <= tolerance:
        return low
    if abs(high_value) <= tolerance:
        return high
    if low_value * high_value > 0.0:
        raise ValueError("price is not bracketed by the supplied volatility bounds")

    mid = 0.5 * (low + high)
    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        mid_value = (
            _scalar_price(
                spot,
                strike,
                time_to_expiry,
                rate,
                mid,
                option_type,
                dividend_yield,
            )
            - target
        )
        if abs(mid_value) <= tolerance or (high - low) <= tolerance:
            return mid
        if low_value * mid_value <= 0.0:
            high = mid
            high_value = mid_value
        else:
            low = mid
            low_value = mid_value

    raise RuntimeError("bisection implied-vol solver did not converge")


def implied_vol_newton(
    price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    initial_volatility: float = 0.2,
    tolerance: float = 1e-12,
    max_iterations: int = 100,
) -> float:
    """Invert Black-Scholes implied volatility with safeguarded Newton steps."""

    _validate_iv_request(
        price,
        spot,
        strike,
        time_to_expiry,
        rate,
        option_type,
        dividend_yield,
        tolerance,
        max_iterations,
    )
    if not isfinite(initial_volatility) or initial_volatility <= 0.0:
        raise ValueError("initial_volatility must be a finite positive number")

    target = float(price)
    low = 1e-8
    high = 5.0
    low_value = (
        _scalar_price(
            spot, strike, time_to_expiry, rate, low, option_type, dividend_yield
        )
        - target
    )
    high_value = (
        _scalar_price(
            spot, strike, time_to_expiry, rate, high, option_type, dividend_yield
        )
        - target
    )
    if abs(low_value) <= tolerance:
        return low
    if abs(high_value) <= tolerance:
        return high
    if low_value * high_value > 0.0:
        raise ValueError("price is not bracketed by the default volatility bounds")

    sigma = min(max(float(initial_volatility), low), high)
    for _ in range(max_iterations):
        model_price = _scalar_price(
            spot, strike, time_to_expiry, rate, sigma, option_type, dividend_yield
        )
        diff = model_price - target
        if abs(diff) <= tolerance:
            return sigma

        if low_value * diff <= 0.0:
            high = sigma
            high_value = diff
        else:
            low = sigma
            low_value = diff

        vega = black_scholes_greeks(
            spot, strike, time_to_expiry, rate, sigma, option_type, dividend_yield
        ).vega
        if vega > np.finfo(float).eps:
            candidate = sigma - diff / float(vega)
        else:
            candidate = float("nan")
        if not isfinite(candidate) or candidate <= low or candidate >= high:
            candidate = 0.5 * (low + high)
        sigma = candidate
        if (high - low) <= tolerance:
            return sigma

    raise RuntimeError("Newton implied-vol solver did not converge")


def no_arbitrage_price_bounds(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Return European option no-arbitrage lower and upper price bounds."""

    _validate_option_type(option_type)
    s, k, t, r, q = map(
        float, (spot, strike, time_to_expiry, rate, dividend_yield)
    )
    _validate_price_inputs(
        np.asarray(s), np.asarray(k), np.asarray(t), np.asarray(1.0)
    )
    _validate_finite("rate", np.asarray(r))
    _validate_finite("dividend_yield", np.asarray(q))
    discounted_spot = s * np.exp(-q * t)
    discounted_strike = k * np.exp(-r * t)
    if option_type == "call":
        return max(discounted_spot - discounted_strike, 0.0), discounted_spot
    return max(discounted_strike - discounted_spot, 0.0), discounted_strike


def _validate_iv_request(
    price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: OptionType,
    dividend_yield: float,
    tolerance: float,
    max_iterations: int,
) -> None:
    _validate_option_type(option_type)
    for name, value in {
        "price": price,
        "spot": spot,
        "strike": strike,
        "time_to_expiry": time_to_expiry,
        "rate": rate,
        "dividend_yield": dividend_yield,
        "tolerance": tolerance,
    }.items():
        if not isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if spot <= 0.0:
        raise ValueError("spot must be positive")
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if time_to_expiry <= 0.0:
        raise ValueError("time_to_expiry must be positive for implied volatility")
    if price < 0.0:
        raise ValueError("price must be non-negative")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    lower_bound, upper_bound = no_arbitrage_price_bounds(
        spot, strike, time_to_expiry, rate, option_type, dividend_yield
    )
    slack = 1e-12
    if price < lower_bound - slack or price > upper_bound + slack:
        raise ValueError(
            "price violates European option no-arbitrage bounds: "
            f"{lower_bound:.12g} <= price <= {upper_bound:.12g}"
        )


def _broadcast_inputs(*values: ArrayLike) -> tuple[np.ndarray, ...]:
    arrays = (np.asarray(value, dtype=float) for value in values)
    return tuple(np.broadcast_arrays(*arrays))


def _validate_price_inputs(
    spot: np.ndarray,
    strike: np.ndarray,
    time_to_expiry: np.ndarray,
    volatility: np.ndarray,
) -> None:
    _validate_finite("spot", spot)
    _validate_finite("strike", strike)
    _validate_finite("time_to_expiry", time_to_expiry)
    _validate_finite("volatility", volatility)
    if np.any(spot <= 0.0):
        raise ValueError("spot must be positive")
    if np.any(strike <= 0.0):
        raise ValueError("strike must be positive")
    if np.any(time_to_expiry < 0.0):
        raise ValueError("time_to_expiry must be non-negative")
    if np.any(volatility < 0.0):
        raise ValueError("volatility must be non-negative")


def _validate_finite(name: str, values: np.ndarray) -> None:
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must be finite")


def _validate_option_type(option_type: str) -> None:
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")


def _d1_d2(
    spot: np.ndarray,
    strike: np.ndarray,
    time_to_expiry: np.ndarray,
    rate: np.ndarray,
    volatility: np.ndarray,
    dividend_yield: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sqrt_t = np.sqrt(time_to_expiry)
    d1 = (
        np.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    return d1, d2


def _discounted_intrinsic(
    spot: np.ndarray,
    strike: np.ndarray,
    time_to_expiry: np.ndarray,
    rate: np.ndarray,
    dividend_yield: np.ndarray,
    option_type: OptionType,
) -> np.ndarray:
    discounted_spot = spot * np.exp(-dividend_yield * time_to_expiry)
    discounted_strike = strike * np.exp(-rate * time_to_expiry)
    if option_type == "call":
        return np.maximum(discounted_spot - discounted_strike, 0.0)
    return np.maximum(discounted_strike - discounted_spot, 0.0)


def _scalar_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float,
) -> float:
    return float(
        black_scholes_price(
            spot,
            strike,
            time_to_expiry,
            rate,
            volatility,
            option_type,
            dividend_yield,
        )
    )


def _as_scalar_if_scalar(value: np.ndarray) -> float | np.ndarray:
    if value.shape == ():
        return float(value)
    return value
