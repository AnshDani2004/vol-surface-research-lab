"""SABR calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from volsurface.sabr import hagan_sabr_iv

PARAMETER_BOUNDS = {
    "alpha": (1e-4, 5.0),
    "rho": (-0.999, 0.999),
    "nu": (1e-4, 10.0),
}

SABR_PARAMS_COLUMNS = [
    "expiry",
    "T",
    "n_points",
    "beta",
    "alpha",
    "rho",
    "nu",
    "rmse_iv",
    "mae_iv",
    "weighted_rmse_iv",
    "max_abs_error",
    "optimizer_success",
    "optimizer_message",
    "alpha_at_bound",
    "rho_at_bound",
    "nu_at_bound",
    "status",
    "failure_reason",
]

SABR_FIT_POINT_COLUMNS = [
    "expiry",
    "option_type",
    "strike",
    "forward",
    "time_to_expiry",
    "log_moneyness",
    "market_iv",
    "sabr_iv",
    "iv_error",
    "abs_iv_error",
    "weight",
    "option_side_for_surface",
]

SABR_FAILURE_COLUMNS = [
    "expiry",
    "status",
    "reason",
    "severity",
    "detail",
]


@dataclass(frozen=True)
class SabrCalibrationResult:
    """Container for Stage 4 SABR calibration outputs."""

    params: pd.DataFrame
    fit_points: pd.DataFrame
    failure_report: pd.DataFrame


def calibrate_sabr_per_expiry(
    surface_df: pd.DataFrame,
    beta: float = 0.5,
    weighting: str = "spread_weighted",
    min_points: int = 6,
    max_abs_error_threshold: float = 0.25,
) -> SabrCalibrationResult:
    """Calibrate fixed-beta SABR parameters independently by expiry."""

    _validate_calibration_inputs(surface_df, beta, weighting, min_points)
    param_rows: list[dict[str, object]] = []
    fit_point_frames: list[pd.DataFrame] = []
    failure_rows: list[dict[str, object]] = []

    for expiry, expiry_rows in surface_df.groupby("expiry", sort=True):
        points = _valid_surface_points(expiry_rows)
        if len(points) < min_points:
            reason = f"too few valid points: {len(points)} < {min_points}"
            param_rows.append(_failed_param_row(expiry, beta, len(points), reason))
            failure_rows.append(_failure_row(expiry, "skipped", reason, "error"))
            continue

        forward = float(points["forward"].median())
        time_to_expiry = float(points["time_to_expiry"].median())
        market_iv = points["implied_vol"].to_numpy(dtype=float)
        strikes = points["strike"].to_numpy(dtype=float)
        weights = compute_sabr_weights(points, weighting).to_numpy(dtype=float)
        residual_scale = np.sqrt(weights)
        initial = _initial_guess(forward, beta, market_iv)

        def residuals(
            params: np.ndarray,
            forward_value: float = forward,
            strikes_value: np.ndarray = strikes,
            time_value: float = time_to_expiry,
            market_iv_value: np.ndarray = market_iv,
            residual_scale_value: np.ndarray = residual_scale,
        ) -> np.ndarray:
            alpha, rho, nu = params
            model_iv = hagan_sabr_iv(
                forward_value,
                strikes_value,
                time_value,
                alpha,
                beta,
                rho,
                nu,
            )
            return residual_scale_value * (model_iv - market_iv_value)

        result = least_squares(
            residuals,
            x0=initial,
            bounds=(
                [
                    PARAMETER_BOUNDS["alpha"][0],
                    PARAMETER_BOUNDS["rho"][0],
                    PARAMETER_BOUNDS["nu"][0],
                ],
                [
                    PARAMETER_BOUNDS["alpha"][1],
                    PARAMETER_BOUNDS["rho"][1],
                    PARAMETER_BOUNDS["nu"][1],
                ],
            ),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=2000,
        )
        alpha, rho, nu = (float(value) for value in result.x)
        model_iv = hagan_sabr_iv(
            forward,
            strikes,
            time_to_expiry,
            alpha,
            beta,
            rho,
            nu,
        )
        fit_points = _build_fit_points(points, model_iv, weights)
        fit_point_frames.append(fit_points)

        metrics = _fit_metrics(fit_points)
        bound_hits = _bound_hits(alpha, rho, nu)
        status = "calibrated" if result.success else "optimizer_failed"
        failure_reason = "" if result.success else str(result.message)
        param_rows.append(
            {
                "expiry": expiry,
                "T": time_to_expiry,
                "n_points": int(len(points)),
                "beta": float(beta),
                "alpha": alpha,
                "rho": rho,
                "nu": nu,
                **metrics,
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                **bound_hits,
                "status": status,
                "failure_reason": failure_reason,
            }
        )
        failure_rows.extend(
            _diagnose_fit(
                expiry=expiry,
                status=status,
                optimizer_success=bool(result.success),
                optimizer_message=str(result.message),
                alpha=alpha,
                rho=rho,
                nu=nu,
                metrics=metrics,
                bound_hits=bound_hits,
                max_abs_error_threshold=max_abs_error_threshold,
            )
        )

    params = pd.DataFrame(param_rows, columns=SABR_PARAMS_COLUMNS)
    fit_points = (
        pd.concat(fit_point_frames, ignore_index=True)
        if fit_point_frames
        else pd.DataFrame(columns=SABR_FIT_POINT_COLUMNS)
    )
    failure_report = pd.DataFrame(failure_rows, columns=SABR_FAILURE_COLUMNS)
    return SabrCalibrationResult(params, fit_points, failure_report)


def compute_sabr_weights(points: pd.DataFrame, weighting: str) -> pd.Series:
    """Compute calibration weights for market-IV residuals."""

    if weighting == "unweighted":
        return pd.Series(1.0, index=points.index, dtype=float)
    if weighting != "spread_weighted":
        raise ValueError("weighting must be 'unweighted' or 'spread_weighted'")
    if {"bid", "ask"}.issubset(points.columns):
        spread = (points["ask"] - points["bid"]).astype(float).abs()
    elif "bid_ask_spread" in points.columns:
        spread = points["bid_ask_spread"].astype(float).abs()
    else:
        return pd.Series(1.0, index=points.index, dtype=float)
    floor = max(float(spread[spread > 0.0].median()) * 0.1, 1e-4)
    return 1.0 / np.maximum(spread, floor)


def _validate_calibration_inputs(
    surface_df: pd.DataFrame,
    beta: float,
    weighting: str,
    min_points: int,
) -> None:
    required = {
        "expiry",
        "option_type",
        "strike",
        "forward",
        "time_to_expiry",
        "log_moneyness",
        "implied_vol",
        "option_side_for_surface",
    }
    missing = required.difference(surface_df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"missing columns for SABR calibration: {missing_text}")
    if beta < 0.0 or beta > 1.0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and in [0, 1]")
    if weighting not in {"unweighted", "spread_weighted"}:
        raise ValueError("weighting must be 'unweighted' or 'spread_weighted'")
    if min_points <= 0:
        raise ValueError("min_points must be positive")


def _valid_surface_points(expiry_rows: pd.DataFrame) -> pd.DataFrame:
    points = expiry_rows.copy()
    mask = (
        np.isfinite(points["strike"])
        & np.isfinite(points["forward"])
        & np.isfinite(points["time_to_expiry"])
        & np.isfinite(points["implied_vol"])
        & (points["strike"] > 0.0)
        & (points["forward"] > 0.0)
        & (points["time_to_expiry"] > 0.0)
        & (points["implied_vol"] > 0.0)
    )
    return points[mask].sort_values(["strike", "option_type"]).reset_index(drop=True)


def _initial_guess(forward: float, beta: float, market_iv: np.ndarray) -> np.ndarray:
    alpha = float(np.clip(np.median(market_iv) * forward ** (1.0 - beta), 1e-4, 5.0))
    return np.asarray([alpha, 0.0, 0.75], dtype=float)


def _build_fit_points(
    points: pd.DataFrame,
    model_iv: np.ndarray,
    weights: np.ndarray,
) -> pd.DataFrame:
    result = points.copy()
    result["market_iv"] = result["implied_vol"].astype(float)
    result["sabr_iv"] = model_iv
    result["iv_error"] = result["sabr_iv"] - result["market_iv"]
    result["abs_iv_error"] = result["iv_error"].abs()
    result["weight"] = weights
    return result.loc[:, SABR_FIT_POINT_COLUMNS]


def _fit_metrics(fit_points: pd.DataFrame) -> dict[str, float]:
    errors = fit_points["iv_error"].to_numpy(dtype=float)
    abs_errors = np.abs(errors)
    weights = fit_points["weight"].to_numpy(dtype=float)
    return {
        "rmse_iv": float(np.sqrt(np.mean(errors**2))),
        "mae_iv": float(np.mean(abs_errors)),
        "weighted_rmse_iv": float(np.sqrt(np.average(errors**2, weights=weights))),
        "max_abs_error": float(np.max(abs_errors)),
    }


def _bound_hits(alpha: float, rho: float, nu: float) -> dict[str, bool]:
    return {
        "alpha_at_bound": _near_bound(alpha, PARAMETER_BOUNDS["alpha"]),
        "rho_at_bound": _near_bound(rho, PARAMETER_BOUNDS["rho"]),
        "nu_at_bound": _near_bound(nu, PARAMETER_BOUNDS["nu"]),
    }


def _near_bound(
    value: float,
    bounds: tuple[float, float],
    tolerance: float = 1e-3,
) -> bool:
    lower, upper = bounds
    return bool(abs(value - lower) <= tolerance or abs(value - upper) <= tolerance)


def _failed_param_row(
    expiry: object,
    beta: float,
    n_points: int,
    reason: str,
) -> dict[str, object]:
    return {
        "expiry": expiry,
        "T": np.nan,
        "n_points": int(n_points),
        "beta": float(beta),
        "alpha": np.nan,
        "rho": np.nan,
        "nu": np.nan,
        "rmse_iv": np.nan,
        "mae_iv": np.nan,
        "weighted_rmse_iv": np.nan,
        "max_abs_error": np.nan,
        "optimizer_success": False,
        "optimizer_message": "",
        "alpha_at_bound": False,
        "rho_at_bound": False,
        "nu_at_bound": False,
        "status": "skipped",
        "failure_reason": reason,
    }


def _diagnose_fit(
    expiry: object,
    status: str,
    optimizer_success: bool,
    optimizer_message: str,
    alpha: float,
    rho: float,
    nu: float,
    metrics: dict[str, float],
    bound_hits: dict[str, bool],
    max_abs_error_threshold: float,
) -> list[dict[str, object]]:
    rows = []
    if not optimizer_success:
        rows.append(
            _failure_row(
                expiry,
                status,
                f"optimizer failure: {optimizer_message}",
                "error",
            )
        )
    for parameter in ("alpha", "rho", "nu"):
        if bound_hits[f"{parameter}_at_bound"]:
            rows.append(
                _failure_row(
                    expiry,
                    status,
                    f"{parameter} near calibration bound",
                    "warning",
                )
            )
    if nu > 5.0:
        rows.append(_failure_row(expiry, status, "high nu > 5.0", "warning"))
    if abs(rho) > 0.95:
        rows.append(_failure_row(expiry, status, "rho near +/-1", "warning"))
    if metrics["max_abs_error"] > max_abs_error_threshold:
        rows.append(
            _failure_row(
                expiry,
                status,
                f"max_abs_error > {max_abs_error_threshold:g}",
                "warning",
                detail=f"max_abs_error={metrics['max_abs_error']:.12g}",
            )
        )
    return rows


def _failure_row(
    expiry: object,
    status: str,
    reason: str,
    severity: str,
    detail: str = "",
) -> dict[str, object]:
    return {
        "expiry": expiry,
        "status": status,
        "reason": reason,
        "severity": severity,
        "detail": detail,
    }
