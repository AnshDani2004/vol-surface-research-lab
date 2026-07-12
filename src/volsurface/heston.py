"""Heston characteristic function and European option pricing."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.polynomial.laguerre import laggauss
from numpy.typing import ArrayLike


def heston_characteristic_function(
    u: ArrayLike,
    time_to_expiry: ArrayLike,
    forward: ArrayLike,
    v0: float,
    theta: float,
    kappa: float,
    sigma: float,
    rho: float,
) -> complex | np.ndarray:
    """Return the Heston characteristic function of log forward price.

    The convention is forward-measure style: the current forward `F` is used as
    the state variable and discounting is handled outside the characteristic
    function. Parameters are initial variance `v0`, long-run variance `theta`,
    mean-reversion speed `kappa`, variance volatility `sigma`, and spot/variance
    correlation `rho`.
    """

    validate_heston_parameters(v0, theta, kappa, sigma, rho)
    u_array, t_array, f_array = np.broadcast_arrays(
        np.asarray(u, dtype=complex),
        np.asarray(time_to_expiry, dtype=float),
        np.asarray(forward, dtype=float),
    )
    _validate_heston_inputs(t_array, f_array)

    iu = 1j * u_array
    b = kappa - rho * sigma * iu
    d = np.sqrt(b**2 + sigma**2 * (u_array**2 + iu))
    g = (b - d) / (b + d)
    exp_neg_dt = np.exp(-d * t_array)
    log_term = np.log((1.0 - g * exp_neg_dt) / (1.0 - g))
    c = (kappa * theta / sigma**2) * ((b - d) * t_array - 2.0 * log_term)
    d_term = ((b - d) / sigma**2) * ((1.0 - exp_neg_dt) / (1.0 - g * exp_neg_dt))
    value = np.exp(iu * np.log(f_array) + c + d_term * v0)
    if value.shape == ():
        return complex(value)
    return value


def heston_call_price(
    forward: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    v0: float,
    theta: float,
    kappa: float,
    sigma: float,
    rho: float,
    discount: ArrayLike = 1.0,
    quadrature_order: int = 32,
) -> float | np.ndarray:
    """Price European calls with Heston P1/P2 integrals."""

    call = _heston_call_array(
        forward,
        strike,
        time_to_expiry,
        v0,
        theta,
        kappa,
        sigma,
        rho,
        discount,
        quadrature_order,
    )
    if call.shape == ():
        return float(call)
    return call


def heston_put_price(
    forward: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    v0: float,
    theta: float,
    kappa: float,
    sigma: float,
    rho: float,
    discount: ArrayLike = 1.0,
    quadrature_order: int = 32,
) -> float | np.ndarray:
    """Price European puts from call price and forward put-call parity."""

    f, k, _, df = _broadcast_price_inputs(forward, strike, time_to_expiry, discount)
    call = _heston_call_array(
        f,
        k,
        time_to_expiry,
        v0,
        theta,
        kappa,
        sigma,
        rho,
        df,
        quadrature_order,
    )
    put = call - df * (f - k)
    put = np.maximum(put, 0.0)
    if put.shape == ():
        return float(put)
    return put


def heston_option_price(
    forward: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    option_type: ArrayLike,
    v0: float,
    theta: float,
    kappa: float,
    sigma: float,
    rho: float,
    discount: ArrayLike = 1.0,
    quadrature_order: int = 32,
) -> float | np.ndarray:
    """Price calls or puts using Heston call pricing plus parity for puts."""

    f, k, _, df = _broadcast_price_inputs(forward, strike, time_to_expiry, discount)
    option_types = np.asarray(option_type)
    option_types = np.broadcast_to(option_types, f.shape)
    if np.any((option_types != "call") & (option_types != "put")):
        raise ValueError("option_type must be 'call' or 'put'")
    call = _heston_call_array(
        f,
        k,
        time_to_expiry,
        v0,
        theta,
        kappa,
        sigma,
        rho,
        df,
        quadrature_order,
    )
    put = np.maximum(call - df * (f - k), 0.0)
    prices = np.where(option_types == "put", put, call)
    if prices.shape == ():
        return float(prices)
    return prices


def validate_heston_parameters(
    v0: float,
    theta: float,
    kappa: float,
    sigma: float,
    rho: float,
) -> None:
    """Validate Heston model parameters."""

    for name, value in {
        "v0": v0,
        "theta": theta,
        "kappa": kappa,
        "sigma": sigma,
        "rho": rho,
    }.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if v0 <= 0.0:
        raise ValueError("v0 must be positive")
    if theta <= 0.0:
        raise ValueError("theta must be positive")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if rho <= -1.0 or rho >= 1.0:
        raise ValueError("rho must be in (-1, 1)")


def feller_condition(kappa: float, theta: float, sigma: float) -> bool:
    """Return whether the Heston Feller condition is satisfied."""

    return bool(2.0 * kappa * theta >= sigma**2)


def _heston_call_array(
    forward: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    v0: float,
    theta: float,
    kappa: float,
    sigma: float,
    rho: float,
    discount: ArrayLike,
    quadrature_order: int,
) -> np.ndarray:
    validate_heston_parameters(v0, theta, kappa, sigma, rho)
    f, k, t, df = _broadcast_price_inputs(forward, strike, time_to_expiry, discount)
    nodes, weights = _laguerre_nodes(quadrature_order)
    u = nodes[:, None]
    k_row = k.reshape(1, -1)
    f_row = f.reshape(1, -1)
    t_row = t.reshape(1, -1)

    phi_minus_i = heston_characteristic_function(
        -1j,
        t_row,
        f_row,
        v0,
        theta,
        kappa,
        sigma,
        rho,
    )
    phi_u_minus_i = heston_characteristic_function(
        u - 1j,
        t_row,
        f_row,
        v0,
        theta,
        kappa,
        sigma,
        rho,
    )
    phi_u = heston_characteristic_function(
        u,
        t_row,
        f_row,
        v0,
        theta,
        kappa,
        sigma,
        rho,
    )

    exp_term = np.exp(-1j * u * np.log(k_row))
    p1_integrand = np.real(exp_term * phi_u_minus_i / (1j * u * phi_minus_i))
    p2_integrand = np.real(exp_term * phi_u / (1j * u))
    scale = weights[:, None] * np.exp(nodes[:, None])
    p1 = 0.5 + (1.0 / np.pi) * np.sum(scale * p1_integrand, axis=0)
    p2 = 0.5 + (1.0 / np.pi) * np.sum(scale * p2_integrand, axis=0)

    call = df.reshape(-1) * (f.reshape(-1) * p1 - k.reshape(-1) * p2)
    intrinsic = df.reshape(-1) * np.maximum(f.reshape(-1) - k.reshape(-1), 0.0)
    call = np.maximum(call, intrinsic)
    call = np.where(np.isfinite(call), call, np.nan)
    if np.any(~np.isfinite(call)):
        raise ValueError("Heston call price integration produced non-finite values")
    return call.reshape(f.shape)


def _broadcast_price_inputs(
    forward: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    discount: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f, k, t, df = np.broadcast_arrays(
        np.asarray(forward, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(time_to_expiry, dtype=float),
        np.asarray(discount, dtype=float),
    )
    _validate_heston_inputs(t, f)
    if np.any(~np.isfinite(k)) or np.any(k <= 0.0):
        raise ValueError("strike must be finite and positive")
    if np.any(~np.isfinite(df)) or np.any(df <= 0.0):
        raise ValueError("discount must be finite and positive")
    return f, k, t, df


def _validate_heston_inputs(time_to_expiry: np.ndarray, forward: np.ndarray) -> None:
    if np.any(~np.isfinite(forward)) or np.any(forward <= 0.0):
        raise ValueError("forward must be finite and positive")
    if np.any(~np.isfinite(time_to_expiry)) or np.any(time_to_expiry <= 0.0):
        raise ValueError("time_to_expiry must be finite and positive")


@lru_cache(maxsize=8)
def _laguerre_nodes(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 8:
        raise ValueError("quadrature_order must be at least 8")
    nodes, weights = laggauss(order)
    return nodes.astype(float), weights.astype(float)
