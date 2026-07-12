from __future__ import annotations

import math

import pytest

from volsurface.sabr import hagan_sabr_iv


def test_atm_sabr_formula_returns_finite_positive_iv() -> None:
    iv = hagan_sabr_iv(100.0, 100.0, 1.0, alpha=2.0, beta=0.5, rho=-0.2, nu=0.7)

    assert math.isfinite(iv)
    assert iv > 0.0


def test_non_atm_sabr_formula_returns_finite_positive_iv() -> None:
    iv = hagan_sabr_iv(100.0, 110.0, 0.75, alpha=2.0, beta=0.5, rho=-0.2, nu=0.7)

    assert math.isfinite(iv)
    assert iv > 0.0


@pytest.mark.parametrize(
    ("alpha", "beta", "rho", "nu", "message"),
    [
        (0.0, 0.5, 0.0, 0.5, "alpha"),
        (2.0, -0.1, 0.0, 0.5, "beta"),
        (2.0, 1.1, 0.0, 0.5, "beta"),
        (2.0, 0.5, -1.0, 0.5, "rho"),
        (2.0, 0.5, 1.0, 0.5, "rho"),
        (2.0, 0.5, 0.0, -0.1, "nu"),
    ],
)
def test_invalid_sabr_parameters_are_rejected(
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        hagan_sabr_iv(100.0, 100.0, 1.0, alpha, beta, rho, nu)


def test_sabr_near_zero_nu_returns_sensible_finite_result() -> None:
    iv = hagan_sabr_iv(100.0, 95.0, 1.0, alpha=2.0, beta=0.5, rho=0.0, nu=1e-12)

    assert math.isfinite(iv)
    assert iv == pytest.approx(0.2, rel=0.05)

