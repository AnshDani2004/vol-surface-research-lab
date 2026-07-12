"""Global Heston calibration on an observed IV universe."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from volsurface.black_scholes import black_scholes_price
from volsurface.calibration import compute_sabr_weights
from volsurface.heston import (
    feller_condition,
    heston_option_price,
    validate_heston_parameters,
)

HESTON_PARAM_BOUNDS = {
    "v0": (1e-4, 4.0),
    "theta": (1e-4, 4.0),
    "kappa": (1e-3, 50.0),
    "sigma": (1e-4, 10.0),
    "rho": (-0.999, 0.999),
}

HESTON_PARAMS_COLUMNS = [
    "v0",
    "theta",
    "kappa",
    "sigma",
    "rho",
    "objective_value",
    "optimizer_success",
    "optimizer_message",
    "nfev",
    "runtime_seconds",
    "selected_initial_guess",
    "v0_at_bound",
    "theta_at_bound",
    "kappa_at_bound",
    "sigma_at_bound",
    "rho_at_bound",
    "feller_condition",
    "status",
    "failure_reason",
]

HESTON_FIT_POINT_COLUMNS = [
    "expiry",
    "option_type",
    "strike",
    "forward",
    "time_to_expiry",
    "log_moneyness",
    "option_side_for_surface",
    "market_iv",
    "heston_iv",
    "heston_price",
    "market_mid",
    "iv_error",
    "abs_iv_error",
    "weight",
    "pricing_status",
    "iv_status",
    "failure_reason",
]

HESTON_ERROR_COLUMNS = [
    "scope",
    "expiry",
    "total_points",
    "successful_points",
    "failed_points",
    "rmse_iv",
    "mae_iv",
    "weighted_rmse_iv",
    "max_abs_error",
    "median_abs_error",
    "p95_abs_error",
]

HESTON_FAILURE_COLUMNS = [
    "category",
    "severity",
    "expiry",
    "detail",
]


@dataclass(frozen=True)
class HestonCalibrationResult:
    """Container for Stage 5 Heston calibration outputs."""

    params: pd.DataFrame
    fit_points: pd.DataFrame
    fit_errors: pd.DataFrame
    failure_report: pd.DataFrame
    runtime_summary: pd.DataFrame


def calibrate_heston_global(
    surface_df: pd.DataFrame,
    weighting: str = "spread_weighted",
    discount: float = 1.0,
    quadrature_order: int = 32,
    max_nfev: int = 60,
) -> HestonCalibrationResult:
    """Calibrate one global Heston parameter set to market implied vols."""

    _validate_surface(surface_df)
    validate_heston_parameters(0.04, 0.04, 1.0, 1.0, -0.3)
    start_time = perf_counter()
    surface = surface_df.sort_values(["expiry", "strike", "option_type"]).reset_index(
        drop=True
    )
    weights = compute_sabr_weights(surface, weighting).to_numpy(dtype=float)
    residual_scale = np.sqrt(weights)
    market_iv = surface["implied_vol"].to_numpy(dtype=float)
    starts = _initial_guesses(surface)
    parameter_order = ("v0", "theta", "kappa", "sigma", "rho")
    bounds = (
        [HESTON_PARAM_BOUNDS[name][0] for name in parameter_order],
        [HESTON_PARAM_BOUNDS[name][1] for name in parameter_order],
    )
    counters = {"price_eval_count": 0, "iv_conversion_count": 0}
    best = None
    best_start_label = ""

    def residuals(params: np.ndarray) -> np.ndarray:
        ivs, statuses = _heston_ivs_for_surface(
            surface,
            params,
            discount,
            quadrature_order,
            counters,
        )
        residual = np.where(statuses, ivs - market_iv, 10.0)
        return residual_scale * residual

    for label, guess in starts:
        result = least_squares(
            residuals,
            x0=guess,
            bounds=bounds,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
            max_nfev=max_nfev,
        )
        objective = float(np.sum(result.fun**2))
        if best is None or objective < best[0]:
            best = (objective, result)
            best_start_label = label

    assert best is not None
    objective, optimizer_result = best
    params = optimizer_result.x.astype(float)
    fit_points = build_heston_fit_points(
        surface,
        params,
        weights,
        discount=discount,
        quadrature_order=quadrature_order,
        counters=counters,
    )
    fit_errors = summarize_heston_fit_errors(fit_points)
    runtime_seconds = perf_counter() - start_time
    params_df = _params_frame(
        params,
        objective,
        optimizer_result,
        runtime_seconds,
        best_start_label,
    )
    failure_report = build_heston_failure_report(params_df, fit_points)
    runtime_summary = pd.DataFrame(
        [
            {
                "heston_calibration_runtime_seconds": runtime_seconds,
                "heston_price_eval_count": counters["price_eval_count"],
                "heston_iv_conversion_count": counters["iv_conversion_count"],
                "heston_failed_price_count": int(
                    (fit_points["pricing_status"] != "success").sum()
                ),
                "heston_failed_iv_count": int(
                    (fit_points["iv_status"] != "success").sum()
                ),
            }
        ]
    )
    return HestonCalibrationResult(
        params=params_df,
        fit_points=fit_points,
        fit_errors=fit_errors,
        failure_report=failure_report,
        runtime_summary=runtime_summary,
    )


def build_heston_fit_points(
    surface: pd.DataFrame,
    params: np.ndarray,
    weights: np.ndarray,
    discount: float = 1.0,
    quadrature_order: int = 32,
    counters: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Build point-level Heston price and IV errors for a parameter vector."""

    prices, price_statuses, price_failures = _heston_prices_for_surface(
        surface,
        params,
        discount,
        quadrature_order,
        counters,
    )
    heston_ivs, iv_success, iv_failures = _implied_vols_from_forward_prices(
        prices,
        surface,
    )
    if counters is not None:
        counters["iv_conversion_count"] += len(surface)
    iv_statuses = np.where(iv_success, "success", "failed")
    failure_reasons = [
        "" if price_status == "success" and success else price_failure or iv_failure
        for price_status, success, price_failure, iv_failure in zip(
            price_statuses,
            iv_success,
            price_failures,
            iv_failures,
            strict=True,
        )
    ]

    result = pd.DataFrame(
        {
            "expiry": surface["expiry"],
            "option_type": surface["option_type"],
            "strike": surface["strike"],
            "forward": surface["forward"],
            "time_to_expiry": surface["time_to_expiry"],
            "log_moneyness": surface["log_moneyness"],
            "option_side_for_surface": surface["option_side_for_surface"],
            "market_iv": surface["implied_vol"],
            "heston_iv": heston_ivs,
            "heston_price": prices,
            "market_mid": surface["mid"],
            "weight": weights,
            "pricing_status": price_statuses,
            "iv_status": iv_statuses,
            "failure_reason": failure_reasons,
        }
    )
    result["iv_error"] = result["heston_iv"] - result["market_iv"]
    result["abs_iv_error"] = result["iv_error"].abs()
    return result.loc[:, HESTON_FIT_POINT_COLUMNS]


def summarize_heston_fit_errors(fit_points: pd.DataFrame) -> pd.DataFrame:
    """Summarize Heston IV errors globally and by expiry."""

    rows = [_error_row("global", "", fit_points)]
    for expiry, group in fit_points.groupby("expiry", sort=True):
        rows.append(_error_row("expiry", expiry, group))
    return pd.DataFrame(rows, columns=HESTON_ERROR_COLUMNS)


def build_heston_failure_report(
    params: pd.DataFrame,
    fit_points: pd.DataFrame,
) -> pd.DataFrame:
    """Build Heston optimizer, parameter, Feller, and point-failure warnings."""

    row = params.iloc[0]
    rows = []
    if not bool(row["optimizer_success"]):
        rows.append(
            _failure("optimizer_failure", "error", "", row["optimizer_message"])
        )
    for parameter in ("v0", "theta", "kappa", "sigma", "rho"):
        if bool(row[f"{parameter}_at_bound"]):
            rows.append(_failure("parameter_bound_hit", "warning", "", parameter))
    if not bool(row["feller_condition"]):
        rows.append(
            _failure("feller_violation", "warning", "", "2*kappa*theta < sigma^2")
        )
    if row["sigma"] > 5.0:
        rows.append(_failure("suspicious_parameter", "warning", "", "sigma > 5.0"))
    if abs(row["rho"]) > 0.95:
        rows.append(_failure("suspicious_parameter", "warning", "", "rho near +/-1"))
    if row["kappa"] > 25.0:
        rows.append(_failure("suspicious_parameter", "warning", "", "kappa > 25"))
    ratio = row["v0"] / row["theta"]
    if ratio > 4.0 or ratio < 0.25:
        rows.append(
            _failure("suspicious_parameter", "warning", "", "v0/theta mismatch")
        )

    failed = fit_points[fit_points["iv_status"] != "success"]
    for expiry, count in failed.groupby("expiry").size().items():
        rows.append(_failure("iv_conversion_failure", "error", expiry, f"{count} rows"))
    return pd.DataFrame(rows, columns=HESTON_FAILURE_COLUMNS)


def heston_bound_hits(params: dict[str, float]) -> dict[str, bool]:
    """Return parameter bound-hit diagnostics for a Heston parameter dict."""

    return {
        f"{name}_at_bound": _near_bound(value, HESTON_PARAM_BOUNDS[name])
        for name, value in params.items()
    }


def _heston_ivs_for_surface(
    surface: pd.DataFrame,
    params: np.ndarray,
    discount: float,
    quadrature_order: int,
    counters: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    prices, price_statuses, _ = _heston_prices_for_surface(
        surface,
        params,
        discount,
        quadrature_order,
        counters,
    )
    ivs, statuses, _ = _implied_vols_from_forward_prices(prices, surface)
    price_success = np.asarray([status == "success" for status in price_statuses])
    statuses = statuses & price_success
    counters["iv_conversion_count"] += len(surface)
    return ivs, statuses


def _heston_prices_for_surface(
    surface: pd.DataFrame,
    params: np.ndarray,
    discount: float,
    quadrature_order: int,
    counters: dict[str, int] | None,
) -> tuple[np.ndarray, list[str], list[str]]:
    v0, theta, kappa, sigma, rho = params
    try:
        prices = heston_option_price(
            forward=surface["forward"].to_numpy(dtype=float),
            strike=surface["strike"].to_numpy(dtype=float),
            time_to_expiry=surface["time_to_expiry"].to_numpy(dtype=float),
            option_type=surface["option_type"].to_numpy(),
            v0=float(v0),
            theta=float(theta),
            kappa=float(kappa),
            sigma=float(sigma),
            rho=float(rho),
            discount=discount,
            quadrature_order=quadrature_order,
        )
    except (RuntimeError, ValueError, OverflowError, FloatingPointError) as error:
        return (
            np.full(len(surface), np.nan),
            ["failed"] * len(surface),
            [str(error)] * len(surface),
        )
    if counters is not None:
        counters["price_eval_count"] += len(surface)
    statuses = ["success"] * len(surface)
    failures = [""] * len(surface)
    return np.asarray(prices, dtype=float), statuses, failures


def _validate_surface(surface_df: pd.DataFrame) -> None:
    required = {
        "expiry",
        "option_type",
        "strike",
        "forward",
        "time_to_expiry",
        "log_moneyness",
        "option_side_for_surface",
        "implied_vol",
        "mid",
    }
    missing = required.difference(surface_df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"missing columns for Heston calibration: {missing_text}")


def _initial_guesses(surface: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    atm = surface[surface["log_moneyness"].abs() <= 0.01]
    base_iv = float(atm["implied_vol"].median())
    if not np.isfinite(base_iv):
        base_iv = float(surface["implied_vol"].median())
    base_var = float(np.clip(base_iv**2, 1e-4, 1.0))
    guesses = [
        ("kappa_1_sigma_0.5_rho_-0.5", [base_var, base_var, 1.0, 0.5, -0.5]),
        ("kappa_5_sigma_1.5_rho_-0.3", [base_var, base_var, 5.0, 1.5, -0.3]),
    ]
    return [(label, np.asarray(values, dtype=float)) for label, values in guesses]


def _params_frame(
    params: np.ndarray,
    objective: float,
    optimizer_result: object,
    runtime_seconds: float,
    selected_initial_guess: str,
) -> pd.DataFrame:
    v0, theta, kappa, sigma, rho = (float(value) for value in params)
    bound_hits = heston_bound_hits(
        {"v0": v0, "theta": theta, "kappa": kappa, "sigma": sigma, "rho": rho}
    )
    success = bool(optimizer_result.success)
    row = {
        "v0": v0,
        "theta": theta,
        "kappa": kappa,
        "sigma": sigma,
        "rho": rho,
        "objective_value": float(objective),
        "optimizer_success": success,
        "optimizer_message": str(optimizer_result.message),
        "nfev": int(optimizer_result.nfev),
        "runtime_seconds": runtime_seconds,
        "selected_initial_guess": selected_initial_guess,
        **bound_hits,
        "feller_condition": feller_condition(kappa, theta, sigma),
        "status": "calibrated" if success else "optimizer_failed",
        "failure_reason": "" if success else str(optimizer_result.message),
    }
    return pd.DataFrame([row], columns=HESTON_PARAMS_COLUMNS)


def _error_row(
    scope: str,
    expiry: object,
    fit_points: pd.DataFrame,
) -> dict[str, object]:
    successful = fit_points[fit_points["iv_status"] == "success"]
    errors = successful["iv_error"].to_numpy(dtype=float)
    abs_errors = np.abs(errors)
    weights = successful["weight"].to_numpy(dtype=float)
    if len(successful) == 0:
        rmse = mae = weighted_rmse = max_abs = median_abs = p95_abs = np.nan
    else:
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(abs_errors))
        weighted_rmse = float(np.sqrt(np.average(errors**2, weights=weights)))
        max_abs = float(np.max(abs_errors))
        median_abs = float(np.median(abs_errors))
        p95_abs = float(np.quantile(abs_errors, 0.95))
    return {
        "scope": scope,
        "expiry": expiry,
        "total_points": int(len(fit_points)),
        "successful_points": int(len(successful)),
        "failed_points": int(len(fit_points) - len(successful)),
        "rmse_iv": rmse,
        "mae_iv": mae,
        "weighted_rmse_iv": weighted_rmse,
        "max_abs_error": max_abs,
        "median_abs_error": median_abs,
        "p95_abs_error": p95_abs,
    }


def _implied_vols_from_forward_prices(
    prices: np.ndarray,
    surface: pd.DataFrame,
    lower: float = 1e-8,
    upper: float = 5.0,
    iterations: int = 48,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    forwards = surface["forward"].to_numpy(dtype=float)
    strikes = surface["strike"].to_numpy(dtype=float)
    times = surface["time_to_expiry"].to_numpy(dtype=float)
    option_types = surface["option_type"].to_numpy()
    prices = np.asarray(prices, dtype=float)
    lower_bounds = np.where(
        option_types == "call",
        np.maximum(forwards - strikes, 0.0),
        np.maximum(strikes - forwards, 0.0),
    )
    upper_bounds = np.where(option_types == "call", forwards, strikes)
    success = (
        np.isfinite(prices)
        & (prices >= lower_bounds - 1e-10)
        & (prices <= upper_bounds + 1e-10)
    )
    lows = np.full(len(prices), lower, dtype=float)
    highs = np.full(len(prices), upper, dtype=float)
    for _ in range(iterations):
        mids = 0.5 * (lows + highs)
        model_prices = _black_scholes_forward_prices(
            forwards,
            strikes,
            times,
            mids,
            option_types,
        )
        too_high = model_prices > prices
        highs = np.where(too_high, mids, highs)
        lows = np.where(too_high, lows, mids)
    ivs = 0.5 * (lows + highs)
    ivs = np.where(success, ivs, np.nan)
    failures = [
        "" if ok else "price violates forward no-arbitrage bounds" for ok in success
    ]
    return ivs, success, failures


def _black_scholes_forward_prices(
    forwards: np.ndarray,
    strikes: np.ndarray,
    times: np.ndarray,
    vols: np.ndarray,
    option_types: np.ndarray,
) -> np.ndarray:
    prices = np.empty_like(forwards, dtype=float)
    call_mask = option_types == "call"
    if np.any(call_mask):
        prices[call_mask] = black_scholes_price(
            forwards[call_mask],
            strikes[call_mask],
            times[call_mask],
            0.0,
            vols[call_mask],
            "call",
        )
    if np.any(~call_mask):
        prices[~call_mask] = black_scholes_price(
            forwards[~call_mask],
            strikes[~call_mask],
            times[~call_mask],
            0.0,
            vols[~call_mask],
            "put",
        )
    return prices


def _near_bound(
    value: float,
    bounds: tuple[float, float],
    tolerance: float = 1e-3,
) -> bool:
    lower, upper = bounds
    return bool(abs(value - lower) <= tolerance or abs(value - upper) <= tolerance)


def _failure(
    category: str,
    severity: str,
    expiry: object,
    detail: str,
) -> dict[str, object]:
    return {
        "category": category,
        "severity": severity,
        "expiry": expiry,
        "detail": detail,
    }
