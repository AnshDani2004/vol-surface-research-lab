"""Hagan SABR implied-volatility approximation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def hagan_sabr_iv(
    forward: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float | np.ndarray:
    """Return Hagan 2002 lognormal SABR implied volatility.

    The approximation assumes a lognormal Black implied-volatility convention,
    positive forward and strike, positive expiry, alpha > 0, beta in [0, 1],
    rho in (-1, 1), and nu >= 0. Dividend and rate effects are assumed to have
    already been absorbed into the forward input.
    """

    _validate_sabr_parameters(alpha, beta, rho, nu)
    f, k, t = np.broadcast_arrays(
        np.asarray(forward, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(time_to_expiry, dtype=float),
    )
    _validate_inputs(f, k, t)

    one_minus_beta = 1.0 - beta
    log_fk = np.log(f / k)
    fk_beta = (f * k) ** (0.5 * one_minus_beta)
    alpha_over_fk = alpha / fk_beta
    log_correction = (
        1.0
        + (one_minus_beta**2 / 24.0) * log_fk**2
        + (one_minus_beta**4 / 1920.0) * log_fk**4
    )
    time_correction = 1.0 + t * (
        (one_minus_beta**2 / 24.0) * (alpha**2 / ((f * k) ** one_minus_beta))
        + (rho * beta * nu * alpha) / (4.0 * fk_beta)
        + ((2.0 - 3.0 * rho**2) * nu**2) / 24.0
    )

    if nu == 0.0:
        z_over_x = np.ones_like(f)
    else:
        z = (nu / alpha) * fk_beta * log_fk
        z_over_x = _z_over_x(z, rho)

    iv = (alpha_over_fk / log_correction) * z_over_x * time_correction
    if np.any(~np.isfinite(iv)) or np.any(iv <= 0.0):
        raise ValueError("Hagan SABR implied volatility is not finite and positive")
    if iv.shape == ():
        return float(iv)
    return iv


def _validate_sabr_parameters(alpha: float, beta: float, rho: float, nu: float) -> None:
    for name, value in {
        "alpha": alpha,
        "beta": beta,
        "rho": rho,
        "nu": nu,
    }.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if beta < 0.0 or beta > 1.0:
        raise ValueError("beta must be in [0, 1]")
    if rho <= -1.0 or rho >= 1.0:
        raise ValueError("rho must be in (-1, 1)")
    if nu < 0.0:
        raise ValueError("nu must be non-negative")


def _validate_inputs(forward: np.ndarray, strike: np.ndarray, time: np.ndarray) -> None:
    if np.any(~np.isfinite(forward)) or np.any(forward <= 0.0):
        raise ValueError("forward must be finite and positive")
    if np.any(~np.isfinite(strike)) or np.any(strike <= 0.0):
        raise ValueError("strike must be finite and positive")
    if np.any(~np.isfinite(time)) or np.any(time <= 0.0):
        raise ValueError("time_to_expiry must be finite and positive")


def _z_over_x(z: np.ndarray, rho: float) -> np.ndarray:
    small = np.abs(z) < 1e-7
    ratio = np.empty_like(z, dtype=float)
    ratio[small] = 1.0 - 0.5 * rho * z[small] + (
        (2.0 - 3.0 * rho**2) / 12.0
    ) * z[small] ** 2
    z_large = z[~small]
    numerator = np.sqrt(1.0 - 2.0 * rho * z_large + z_large**2) + z_large - rho
    x_z = np.log(numerator / (1.0 - rho))
    ratio[~small] = z_large / x_z
    return ratio

