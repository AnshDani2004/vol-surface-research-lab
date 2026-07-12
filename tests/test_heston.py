from __future__ import annotations

import math

import numpy as np
import pytest

from volsurface.heston import (
    feller_condition,
    heston_call_price,
    heston_characteristic_function,
    heston_option_price,
    heston_put_price,
)

HESTON_PARAMS = {
    "v0": 0.04,
    "theta": 0.04,
    "kappa": 2.0,
    "sigma": 0.5,
    "rho": -0.5,
}


def test_characteristic_function_is_martingale_at_minus_i() -> None:
    value = heston_characteristic_function(-1j, 1.0, 100.0, **HESTON_PARAMS)

    assert value.real == pytest.approx(100.0)
    assert value.imag == pytest.approx(0.0, abs=1e-12)


def test_heston_call_price_is_deterministic_reference_value() -> None:
    price = heston_call_price(100.0, 100.0, 1.0, **HESTON_PARAMS)

    assert price == pytest.approx(7.2970292125505125)


def test_heston_array_pricing_returns_nonnegative_prices() -> None:
    prices = heston_call_price(
        np.asarray([100.0, 100.0]),
        np.asarray([90.0, 110.0]),
        np.asarray([1.0, 1.0]),
        **HESTON_PARAMS,
    )

    assert prices == pytest.approx([13.64826725, 3.20517029])
    assert np.all(prices >= 0.0)


def test_heston_put_price_satisfies_forward_put_call_parity() -> None:
    call = heston_call_price(100.0, 105.0, 1.0, **HESTON_PARAMS)
    put = heston_put_price(100.0, 105.0, 1.0, **HESTON_PARAMS)

    assert call - put == pytest.approx(-5.0)


def test_heston_option_price_selects_call_and_put() -> None:
    prices = heston_option_price(
        forward=[100.0, 100.0],
        strike=[100.0, 100.0],
        time_to_expiry=[1.0, 1.0],
        option_type=["call", "put"],
        **HESTON_PARAMS,
    )

    assert prices == pytest.approx([7.2970292125505125, 7.297029212550498])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"v0": 0.0}, "v0"),
        ({"theta": -0.01}, "theta"),
        ({"kappa": 0.0}, "kappa"),
        ({"sigma": 0.0}, "sigma"),
        ({"rho": 1.0}, "rho"),
    ],
)
def test_invalid_heston_parameters_are_rejected(
    overrides: dict[str, float],
    message: str,
) -> None:
    params = HESTON_PARAMS | overrides

    with pytest.raises(ValueError, match=message):
        heston_call_price(100.0, 100.0, 1.0, **params)


def test_invalid_heston_option_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="option_type"):
        heston_option_price(100.0, 100.0, 1.0, "straddle", **HESTON_PARAMS)

    with pytest.raises(ValueError, match="forward"):
        heston_call_price(math.nan, 100.0, 1.0, **HESTON_PARAMS)


def test_feller_condition_diagnostic() -> None:
    assert feller_condition(kappa=2.0, theta=0.04, sigma=0.3)
    assert not feller_condition(kappa=1.0, theta=0.02, sigma=0.5)
