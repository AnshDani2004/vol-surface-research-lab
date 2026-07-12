"""Stage 5.5 Heston improvement and fairness diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from volsurface.black_scholes import (
    black_scholes_price,
    implied_vol_bisection,
)
from volsurface.calibration import compute_sabr_weights
from volsurface.heston import (
    feller_condition,
    heston_option_price,
)
from volsurface.heston_calibration import (
    HESTON_PARAM_BOUNDS,
    _implied_vols_from_forward_prices,
    build_heston_fit_points,
    heston_bound_hits,
    summarize_heston_fit_errors,
)
from volsurface.model_comparison import assign_moneyness_bucket

PARAMETER_ORDER = ("v0", "theta", "kappa", "sigma", "rho")
BUCKET_ORDER = ["left_wing", "left_mid", "atm", "right_mid", "right_wing"]
FELLER_GAP_SCALE = 10.0

MULTISTART_COLUMNS = [
    "start_id",
    "initial_v0",
    "initial_theta",
    "initial_kappa",
    "initial_sigma",
    "initial_rho",
    "final_v0",
    "final_theta",
    "final_kappa",
    "final_sigma",
    "final_rho",
    "objective",
    "rmse_iv",
    "mae_iv",
    "weighted_rmse_iv",
    "max_abs_error",
    "optimizer_success",
    "nfev",
    "runtime_seconds",
    "feller_satisfied",
    "parameter_warnings",
]

VARIANT_PARAMS_COLUMNS = [
    "variant_name",
    "expiry",
    "v0",
    "theta",
    "kappa",
    "sigma",
    "rho",
    "objective",
    "rmse_iv",
    "mae_iv",
    "weighted_rmse_iv",
    "max_abs_error",
    "feller_satisfied",
    "parameter_warnings",
    "runtime_seconds",
    "optimizer_success",
    "optimizer_message",
    "nfev",
]

VARIANT_COMPARISON_EXPIRY_COLUMNS = [
    "expiry",
    "n_points",
    "sabr_rmse_iv",
    "stage5_global_heston_rmse_iv",
    "improved_global_heston_rmse_iv",
    "regularized_heston_rmse_iv",
    "feller_penalized_heston_rmse_iv",
    "per_expiry_heston_rmse_iv",
    "best_heston_variant",
    "best_heston_rmse_iv",
    "winner_by_rmse",
    "notes",
]

VARIANT_COMPARISON_BUCKET_COLUMNS = [
    "bucket",
    "n_points",
    "sabr_rmse_iv",
    "stage5_global_heston_rmse_iv",
    "best_heston_rmse_iv",
    "per_expiry_heston_rmse_iv",
    "sabr_bias",
    "best_heston_bias",
    "winner_by_rmse",
    "notes",
]

IMPROVEMENT_SUMMARY_COLUMNS = [
    "diagnostic",
    "result",
    "key_metric_before",
    "key_metric_after",
    "interpretation",
]

FAILURE_REPORT_COLUMNS = ["category", "severity", "variant_name", "expiry", "detail"]


@dataclass(frozen=True)
class HestonVariantResult:
    """Container for one Heston variant fit."""

    params: pd.DataFrame
    fit_points: pd.DataFrame
    fit_errors: pd.DataFrame
    failure_report: pd.DataFrame
    attempts: pd.DataFrame


def run_black_scholes_limit_diagnostic(
    forwards: list[float] | None = None,
    strikes: list[float] | None = None,
    maturities: list[float] | None = None,
    variance_levels: list[float] | None = None,
    sigma: float = 1e-4,
    kappa: float = 2.0,
    rho: float = -0.5,
    quadrature_order: int = 64,
) -> pd.DataFrame:
    """Compare Heston prices to Black-Scholes in the low vol-of-vol limit."""

    forwards = [90.0, 100.0, 110.0] if forwards is None else forwards
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0] if strikes is None else strikes
    maturities = [0.05, 0.25, 1.0] if maturities is None else maturities
    variance_levels = [0.01, 0.04, 0.09] if variance_levels is None else variance_levels
    rows = []
    for forward in forwards:
        for strike in strikes:
            for maturity in maturities:
                for variance in variance_levels:
                    option_type = "put" if strike < forward else "call"
                    bs_vol = float(np.sqrt(variance))
                    bs_price = black_scholes_price(
                        forward,
                        strike,
                        maturity,
                        0.0,
                        bs_vol,
                        option_type,
                    )
                    heston_price = heston_option_price(
                        forward,
                        strike,
                        maturity,
                        option_type,
                        variance,
                        variance,
                        kappa,
                        sigma,
                        rho,
                        quadrature_order=quadrature_order,
                    )
                    heston_iv = _safe_implied_vol(
                        heston_price,
                        forward,
                        strike,
                        maturity,
                        option_type,
                    )
                    rows.append(
                        {
                            "forward": forward,
                            "strike": strike,
                            "time_to_expiry": maturity,
                            "option_type": option_type,
                            "variance": variance,
                            "sigma": sigma,
                            "kappa": kappa,
                            "rho": rho,
                            "black_scholes_price": bs_price,
                            "heston_price": heston_price,
                            "abs_price_error": abs(float(heston_price) - bs_price),
                            "heston_iv": heston_iv,
                            "black_scholes_vol": bs_vol,
                            "abs_iv_error": (
                                abs(heston_iv - bs_vol)
                                if np.isfinite(heston_iv)
                                else np.nan
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def black_scholes_limit_summary(limit_rows: pd.DataFrame) -> dict[str, float]:
    """Return summary metrics for the Heston-to-Black-Scholes diagnostic."""

    return {
        "max_price_error": float(limit_rows["abs_price_error"].max()),
        "median_price_error": float(limit_rows["abs_price_error"].median()),
        "max_iv_error": float(limit_rows["abs_iv_error"].max()),
        "median_iv_error": float(limit_rows["abs_iv_error"].median()),
    }


def generate_synthetic_heston_surface(
    true_params: dict[str, float] | None = None,
    maturities: list[float] | None = None,
    log_moneyness: list[float] | None = None,
    forward: float = 100.0,
    quadrature_order: int = 64,
) -> pd.DataFrame:
    """Generate deterministic synthetic IVs from known Heston parameters."""

    true_params = (
        {"v0": 0.04, "theta": 0.04, "kappa": 2.0, "sigma": 0.6, "rho": -0.5}
        if true_params is None
        else true_params
    )
    maturities = [0.05, 0.10, 0.25] if maturities is None else maturities
    log_moneyness = (
        [-0.08, -0.04, 0.0, 0.04, 0.08] if log_moneyness is None else log_moneyness
    )
    rows = []
    for maturity in maturities:
        expiry = f"synthetic_T_{maturity:.2f}"
        for moneyness in log_moneyness:
            strike = forward * float(np.exp(moneyness))
            option_type = "put" if strike < forward else "call"
            price = heston_option_price(
                forward,
                strike,
                maturity,
                option_type,
                **true_params,
                quadrature_order=quadrature_order,
            )
            implied_vol = _safe_implied_vol(
                price,
                forward,
                strike,
                maturity,
                option_type,
            )
            rows.append(
                {
                    "expiry": expiry,
                    "option_type": option_type,
                    "strike": strike,
                    "forward": forward,
                    "time_to_expiry": maturity,
                    "log_moneyness": moneyness,
                    "option_side_for_surface": (
                        "atm" if abs(moneyness) < 1e-12 else f"otm_{option_type}"
                    ),
                    "implied_vol": implied_vol,
                    "mid": price,
                    "bid": max(float(price) - 0.05, 0.0),
                    "ask": float(price) + 0.05,
                }
            )
    return pd.DataFrame(rows)


def run_synthetic_heston_recovery(
    quadrature_order: int = 32,
    max_nfev: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic Heston IVs and calibrate Heston back to them."""

    true_params = {
        "v0": 0.04,
        "theta": 0.04,
        "kappa": 2.0,
        "sigma": 0.6,
        "rho": -0.5,
    }
    surface = generate_synthetic_heston_surface(true_params=true_params)
    initial = [
        ("synthetic_near_true", _params_to_array(true_params)),
        ("synthetic_lower_sigma", np.asarray([0.035, 0.04, 1.5, 0.4, -0.4])),
        ("synthetic_higher_kappa", np.asarray([0.045, 0.035, 4.0, 0.8, -0.6])),
    ]
    result = calibrate_heston_variant(
        surface,
        "synthetic_heston_recovery",
        initial,
        weighting="unweighted",
        quadrature_order=quadrature_order,
        max_nfev=max_nfev,
    )
    row = result.params.iloc[0]
    recovery = {
        "true_v0": true_params["v0"],
        "true_theta": true_params["theta"],
        "true_kappa": true_params["kappa"],
        "true_sigma": true_params["sigma"],
        "true_rho": true_params["rho"],
        "recovered_v0": row["v0"],
        "recovered_theta": row["theta"],
        "recovered_kappa": row["kappa"],
        "recovered_sigma": row["sigma"],
        "recovered_rho": row["rho"],
        "abs_error_v0": abs(row["v0"] - true_params["v0"]),
        "abs_error_theta": abs(row["theta"] - true_params["theta"]),
        "abs_error_kappa": abs(row["kappa"] - true_params["kappa"]),
        "abs_error_sigma": abs(row["sigma"] - true_params["sigma"]),
        "abs_error_rho": abs(row["rho"] - true_params["rho"]),
        "synthetic_rmse_iv": row["rmse_iv"],
        "synthetic_mae_iv": row["mae_iv"],
        "optimizer_success": row["optimizer_success"],
        "optimizer_message": row["optimizer_message"],
        "runtime_seconds": row["runtime_seconds"],
    }
    fit_points = result.fit_points.copy()
    fit_points["true_heston_iv"] = fit_points["market_iv"]
    return pd.DataFrame([recovery]), fit_points


def run_quadrature_sensitivity(
    surface: pd.DataFrame,
    params: dict[str, float],
    node_counts: list[int] | None = None,
) -> pd.DataFrame:
    """Evaluate IV sensitivity to Gauss-Laguerre node count."""

    node_counts = [32, 64, 96] if node_counts is None else node_counts
    weights = compute_sabr_weights(surface, "spread_weighted").to_numpy(dtype=float)
    frames = {}
    rows = []
    for node_count in node_counts:
        start = perf_counter()
        fit_points = build_heston_fit_points(
            surface,
            _params_to_array(params),
            weights,
            quadrature_order=node_count,
        )
        runtime = perf_counter() - start
        frames[node_count] = fit_points
        rows.append(
            {
                "node_count": node_count,
                "runtime_seconds": runtime,
                "pricing_failures": int(
                    (fit_points["pricing_status"] != "success").sum()
                ),
                "iv_failures": int((fit_points["iv_status"] != "success").sum()),
            }
        )
    reference_nodes = max(node_counts)
    reference = frames[reference_nodes].set_index(_point_keys())["heston_iv"]
    result_rows = []
    for row in rows:
        node_count = int(row["node_count"])
        candidate = frames[node_count].set_index(_point_keys())["heston_iv"]
        diff = (candidate - reference).abs()
        result_rows.append(
            {
                **row,
                "reference_node_count": reference_nodes,
                "mean_abs_iv_difference_from_reference": float(diff.mean()),
                "max_abs_iv_difference_from_reference": float(diff.max()),
            }
        )
    columns = [
        "node_count",
        "reference_node_count",
        "mean_abs_iv_difference_from_reference",
        "max_abs_iv_difference_from_reference",
        "runtime_seconds",
        "pricing_failures",
        "iv_failures",
    ]
    return pd.DataFrame(result_rows, columns=columns)


def default_multistart_guesses(surface: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    """Return a documented deterministic subset of the proposed start grid."""

    base_var = _base_variance(surface)
    grid = [
        (0.5, 0.2, -0.8),
        (0.5, 0.5, -0.5),
        (2.0, 0.2, -0.2),
        (2.0, 1.0, -0.8),
        (5.0, 0.5, -0.5),
        (5.0, 2.0, -0.2),
        (10.0, 0.2, 0.0),
        (10.0, 1.0, -0.5),
        (20.0, 0.5, -0.2),
        (20.0, 2.0, 0.0),
        (2.0, 2.0, 0.0),
        (5.0, 1.0, -0.8),
    ]
    starts = []
    for index, (kappa, sigma, rho) in enumerate(grid, start=1):
        starts.append(
            (
                f"subset_{index:02d}",
                np.asarray([base_var, base_var, kappa, sigma, rho], dtype=float),
            )
        )
    return starts


def compact_variant_guesses(surface: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    """Return a smaller deterministic start set for constrained variants."""

    base_var = _base_variance(surface)
    return [
        ("compact_01", np.asarray([base_var, base_var, 2.0, 0.5, -0.5])),
        ("compact_02", np.asarray([base_var, base_var, 5.0, 1.0, -0.5])),
        ("compact_03", np.asarray([base_var, base_var, 10.0, 0.5, -0.2])),
        ("compact_04", np.asarray([base_var, base_var, 20.0, 0.2, 0.0])),
    ]


def calibrate_heston_variant(
    surface_df: pd.DataFrame,
    variant_name: str,
    initial_guesses: list[tuple[str, np.ndarray]],
    weighting: str = "spread_weighted",
    quadrature_order: int = 32,
    max_nfev: int = 30,
    feller_lambda: float = 0.0,
    regularization_lambda: float = 0.0,
    min_points: int = 6,
) -> HestonVariantResult:
    """Calibrate a Heston variant from explicit deterministic starts."""

    surface = surface_df.sort_values(["expiry", "strike", "option_type"]).reset_index(
        drop=True
    )
    if len(surface) < min_points:
        failure = pd.DataFrame(
            [
                {
                    "category": "too_few_points",
                    "severity": "error",
                    "variant_name": variant_name,
                    "expiry": _expiry_label(surface),
                    "detail": f"{len(surface)} < {min_points}",
                }
            ],
            columns=FAILURE_REPORT_COLUMNS,
        )
        empty_params = pd.DataFrame(columns=VARIANT_PARAMS_COLUMNS)
        fit_points = pd.DataFrame()
        fit_errors = pd.DataFrame()
        attempts = pd.DataFrame(columns=MULTISTART_COLUMNS)
        return HestonVariantResult(
            empty_params, fit_points, fit_errors, failure, attempts
        )

    weights = compute_sabr_weights(surface, weighting).to_numpy(dtype=float)
    residual_scale = np.sqrt(weights)
    market_iv = surface["implied_vol"].to_numpy(dtype=float)
    bounds = (
        [HESTON_PARAM_BOUNDS[name][0] for name in PARAMETER_ORDER],
        [HESTON_PARAM_BOUNDS[name][1] for name in PARAMETER_ORDER],
    )
    attempt_rows = []
    best: tuple[float, object, np.ndarray, str, float] | None = None

    for start_id, initial in initial_guesses:
        start_time = perf_counter()

        def residuals(params: np.ndarray) -> np.ndarray:
            prices = heston_option_price(
                surface["forward"].to_numpy(dtype=float),
                surface["strike"].to_numpy(dtype=float),
                surface["time_to_expiry"].to_numpy(dtype=float),
                surface["option_type"].to_numpy(),
                *params,
                quadrature_order=quadrature_order,
            )
            ivs = _implied_vols_for_surface(prices, surface)
            residual = np.where(np.isfinite(ivs), ivs - market_iv, 10.0)
            penalties = _penalty_residuals(
                params,
                feller_lambda=feller_lambda,
                regularization_lambda=regularization_lambda,
            )
            return np.concatenate([residual_scale * residual, penalties])

        try:
            optimizer_result = least_squares(
                residuals,
                x0=initial,
                bounds=bounds,
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8,
                max_nfev=max_nfev,
            )
            final_params = optimizer_result.x.astype(float)
            objective = float(np.sum(optimizer_result.fun**2))
            success = bool(optimizer_result.success)
            message = str(optimizer_result.message)
            nfev = int(optimizer_result.nfev)
        except (RuntimeError, ValueError, OverflowError, FloatingPointError) as error:
            final_params = np.full(5, np.nan)
            objective = np.inf
            success = False
            message = str(error)
            nfev = 0
        runtime = perf_counter() - start_time
        fit_points = (
            _fit_points_for_params(surface, final_params, weights, quadrature_order)
            if np.all(np.isfinite(final_params))
            else pd.DataFrame()
        )
        metrics = _global_metrics(fit_points)
        warnings = parameter_warnings(final_params)
        attempt_rows.append(
            {
                "start_id": start_id,
                "initial_v0": initial[0],
                "initial_theta": initial[1],
                "initial_kappa": initial[2],
                "initial_sigma": initial[3],
                "initial_rho": initial[4],
                "final_v0": final_params[0],
                "final_theta": final_params[1],
                "final_kappa": final_params[2],
                "final_sigma": final_params[3],
                "final_rho": final_params[4],
                "objective": objective,
                **metrics,
                "optimizer_success": success,
                "nfev": nfev,
                "runtime_seconds": runtime,
                "feller_satisfied": _safe_feller(final_params),
                "parameter_warnings": warnings,
            }
        )
        selection_metric = metrics["weighted_rmse_iv"]
        if np.isfinite(selection_metric) and (
            best is None or selection_metric < best[0]
        ):
            best = (selection_metric, optimizer_result, final_params, message, runtime)

    attempts = pd.DataFrame(attempt_rows, columns=MULTISTART_COLUMNS)
    if best is None:
        failure = pd.DataFrame(
            [
                {
                    "category": "optimizer_failure",
                    "severity": "error",
                    "variant_name": variant_name,
                    "expiry": _expiry_label(surface),
                    "detail": "all starts failed",
                }
            ],
            columns=FAILURE_REPORT_COLUMNS,
        )
        return HestonVariantResult(
            pd.DataFrame(columns=VARIANT_PARAMS_COLUMNS),
            pd.DataFrame(),
            pd.DataFrame(),
            failure,
            attempts,
        )

    _, optimizer_result, params, message, runtime = best
    fit_points = _fit_points_for_params(surface, params, weights, quadrature_order)
    fit_points.insert(0, "variant_name", variant_name)
    fit_errors = summarize_heston_fit_errors(
        fit_points.drop(columns=["variant_name"])
    ).copy()
    fit_errors.insert(0, "variant_name", variant_name)
    metrics = _global_metrics(fit_points)
    param_row = _variant_param_row(
        variant_name,
        _expiry_label(surface),
        params,
        float(np.sum(optimizer_result.fun**2)),
        metrics,
        runtime,
        bool(optimizer_result.success),
        message,
        int(optimizer_result.nfev),
    )
    params_df = pd.DataFrame([param_row], columns=VARIANT_PARAMS_COLUMNS)
    failure_report = build_variant_failure_report(params_df, fit_points)
    return HestonVariantResult(
        params_df, fit_points, fit_errors, failure_report, attempts
    )


def calibrate_per_expiry_heston(
    surface: pd.DataFrame,
    max_nfev: int = 25,
    quadrature_order: int = 32,
    min_points: int = 6,
) -> HestonVariantResult:
    """Fit Heston separately by expiry as a fairness diagnostic."""

    params_frames = []
    fit_point_frames = []
    fit_error_frames = []
    failure_frames = []
    attempt_frames = []
    for expiry, group in surface.groupby("expiry", sort=True):
        result = calibrate_heston_variant(
            group,
            "per_expiry_heston",
            compact_variant_guesses(group),
            quadrature_order=quadrature_order,
            max_nfev=max_nfev,
            min_points=min_points,
        )
        if not result.params.empty:
            params = result.params.copy()
            params["expiry"] = expiry
            params_frames.append(params)
        if not result.fit_points.empty:
            fit_points = result.fit_points.copy()
            fit_points["expiry"] = expiry
            fit_point_frames.append(fit_points)
        if not result.fit_errors.empty:
            errors = result.fit_errors.copy()
            errors["expiry"] = errors["expiry"].replace("", expiry)
            fit_error_frames.append(errors[errors["scope"] == "expiry"])
        failure_frames.append(result.failure_report)
        attempts = result.attempts.copy()
        if not attempts.empty:
            attempts.insert(0, "expiry", expiry)
            attempt_frames.append(attempts)
    params_df = (
        pd.concat(params_frames, ignore_index=True)
        if params_frames
        else pd.DataFrame(columns=VARIANT_PARAMS_COLUMNS)
    )
    fit_points_df = (
        pd.concat(fit_point_frames, ignore_index=True)
        if fit_point_frames
        else pd.DataFrame()
    )
    fit_errors_df = (
        pd.concat(fit_error_frames, ignore_index=True)
        if fit_error_frames
        else pd.DataFrame()
    )
    failure_report = (
        pd.concat(failure_frames, ignore_index=True)
        if failure_frames
        else pd.DataFrame(columns=FAILURE_REPORT_COLUMNS)
    )
    attempts_df = (
        pd.concat(attempt_frames, ignore_index=True)
        if attempt_frames
        else pd.DataFrame(columns=["expiry", *MULTISTART_COLUMNS])
    )
    return HestonVariantResult(
        params_df,
        fit_points_df,
        fit_errors_df,
        failure_report,
        attempts_df,
    )


def feller_penalty(params: np.ndarray, penalty_lambda: float) -> float:
    """Return the scaled soft Feller penalty value."""

    if penalty_lambda <= 0.0:
        return 0.0
    _, theta, kappa, sigma, _ = params
    gap = max(float(sigma**2 - 2.0 * kappa * theta), 0.0)
    return float(penalty_lambda * (FELLER_GAP_SCALE * gap) ** 2)


def regularization_penalty(params: np.ndarray, penalty_lambda: float = 1.0) -> float:
    """Return a mild deterministic parameter-stability penalty."""

    if penalty_lambda <= 0.0:
        return 0.0
    residuals = _regularization_residuals(params)
    return float(penalty_lambda * np.sum(residuals**2))


def parameter_warnings(params: np.ndarray) -> str:
    """Return semicolon-separated parameter diagnostics."""

    if not np.all(np.isfinite(params)):
        return "non-finite parameters"
    v0, theta, kappa, sigma, rho = (float(value) for value in params)
    warnings = []
    hits = heston_bound_hits(
        {"v0": v0, "theta": theta, "kappa": kappa, "sigma": sigma, "rho": rho}
    )
    warnings.extend(
        name.removesuffix("_at_bound") + "_at_bound"
        for name, hit in hits.items()
        if hit
    )
    if not feller_condition(kappa, theta, sigma):
        warnings.append("feller_violation")
    if sigma > 5.0:
        warnings.append("sigma_high")
    if abs(rho) > 0.95:
        warnings.append("rho_edge")
    if kappa > 25.0:
        warnings.append("kappa_high")
    ratio = v0 / theta
    if ratio > 4.0 or ratio < 0.25:
        warnings.append("v0_theta_mismatch")
    return ";".join(warnings) if warnings else "none"


def build_variant_comparison_by_expiry(
    sabr_points: pd.DataFrame,
    variant_fit_points: pd.DataFrame,
) -> pd.DataFrame:
    """Compare Stage 5.5 Heston variants with same-universe SABR by expiry."""

    sabr = sabr_points[sabr_points["row_type"] == "point"].copy()
    rows = []
    for expiry, group in sabr.groupby("expiry", sort=True):
        variant_metrics = _variant_rmse_by_group(
            variant_fit_points[variant_fit_points["expiry"] == expiry],
            group,
        )
        stage5 = variant_metrics.get("global_unconstrained_stage5", np.nan)
        improved = variant_metrics.get("global_multistart_unconstrained", np.nan)
        regularized = variant_metrics.get("global_regularized", np.nan)
        feller = _best_feller_metric(variant_metrics)
        per_expiry = variant_metrics.get("per_expiry_heston", np.nan)
        best_name, best_value = _best_metric(variant_metrics)
        sabr_rmse = _rmse(group["iv_error"])
        rows.append(
            {
                "expiry": expiry,
                "n_points": int(len(group)),
                "sabr_rmse_iv": sabr_rmse,
                "stage5_global_heston_rmse_iv": stage5,
                "improved_global_heston_rmse_iv": improved,
                "regularized_heston_rmse_iv": regularized,
                "feller_penalized_heston_rmse_iv": feller,
                "per_expiry_heston_rmse_iv": per_expiry,
                "best_heston_variant": best_name,
                "best_heston_rmse_iv": best_value,
                "winner_by_rmse": "heston" if best_value < sabr_rmse else "sabr",
                "notes": "same filtered universe",
            }
        )
    return pd.DataFrame(rows, columns=VARIANT_COMPARISON_EXPIRY_COLUMNS)


def build_variant_comparison_by_moneyness(
    sabr_points: pd.DataFrame,
    variant_fit_points: pd.DataFrame,
) -> pd.DataFrame:
    """Compare Heston variants with SABR by Stage 5 moneyness bucket."""

    sabr = sabr_points[sabr_points["row_type"] == "point"].copy()
    sabr["bucket"] = assign_moneyness_bucket(sabr["log_moneyness"])
    variants = variant_fit_points.copy()
    variants["bucket"] = assign_moneyness_bucket(variants["log_moneyness"])
    rows = []
    for bucket in BUCKET_ORDER:
        sabr_group = sabr[sabr["bucket"] == bucket]
        variant_group = variants[variants["bucket"] == bucket]
        variant_metrics = _variant_rmse_by_group(variant_group, sabr_group)
        variant_bias = _variant_bias_by_group(variant_group, sabr_group)
        best_name, best_value = _best_metric(variant_metrics)
        sabr_rmse = _rmse(sabr_group["iv_error"])
        rows.append(
            {
                "bucket": bucket,
                "n_points": int(len(sabr_group)),
                "sabr_rmse_iv": sabr_rmse,
                "stage5_global_heston_rmse_iv": variant_metrics.get(
                    "global_unconstrained_stage5",
                    np.nan,
                ),
                "best_heston_rmse_iv": best_value,
                "per_expiry_heston_rmse_iv": variant_metrics.get(
                    "per_expiry_heston",
                    np.nan,
                ),
                "sabr_bias": _mean(sabr_group["iv_error"]),
                "best_heston_bias": variant_bias.get(best_name, np.nan),
                "winner_by_rmse": "heston" if best_value < sabr_rmse else "sabr",
                "notes": (
                    "same filtered universe" if len(sabr_group) else "empty bucket"
                ),
            }
        )
    return pd.DataFrame(rows, columns=VARIANT_COMPARISON_BUCKET_COLUMNS)


def build_variant_failure_report(
    variant_params: pd.DataFrame,
    variant_fit_points: pd.DataFrame,
    sabr_points: pd.DataFrame | None = None,
    comparison_by_expiry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build failure and warning report for Stage 5.5 variants."""

    rows = []
    for row in variant_params.itertuples(index=False):
        variant = row.variant_name
        expiry = row.expiry
        if not bool(row.optimizer_success):
            rows.append(
                _failure(
                    "optimizer_failure", "error", variant, expiry, row.optimizer_message
                )
            )
        warnings = str(row.parameter_warnings)
        for warning in [] if warnings == "none" else warnings.split(";"):
            severity = "warning"
            category = "parameter_warning"
            if warning == "feller_violation":
                category = "feller_violation"
            elif warning.endswith("_at_bound"):
                category = "parameter_bound_hit"
            rows.append(_failure(category, severity, variant, expiry, warning))
    if not variant_fit_points.empty:
        failed_price = variant_fit_points[
            variant_fit_points["pricing_status"] != "success"
        ]
        failed_iv = variant_fit_points[variant_fit_points["iv_status"] != "success"]
        for (variant, expiry), group in failed_price.groupby(
            ["variant_name", "expiry"]
        ):
            rows.append(
                _failure(
                    "price_failure", "error", variant, expiry, f"{len(group)} rows"
                )
            )
        for (variant, expiry), group in failed_iv.groupby(["variant_name", "expiry"]):
            rows.append(
                _failure("iv_failure", "error", variant, expiry, f"{len(group)} rows")
            )
    if comparison_by_expiry is not None:
        for row in comparison_by_expiry.itertuples(index=False):
            if row.winner_by_rmse == "sabr":
                detail = (
                    f"best Heston RMSE {row.best_heston_rmse_iv:.6g} "
                    f"> SABR {row.sabr_rmse_iv:.6g}"
                )
                rows.append(
                    _failure(
                        "heston_underperforms_sabr",
                        "warning",
                        row.best_heston_variant,
                        row.expiry,
                        detail,
                    )
                )
    if sabr_points is not None and not variant_params.empty:
        rows.extend(_stability_tradeoff_warnings(variant_params))
    return pd.DataFrame(rows, columns=FAILURE_REPORT_COLUMNS)


def build_improvement_summary(
    bs_limit: pd.DataFrame,
    synthetic_recovery: pd.DataFrame,
    quadrature: pd.DataFrame,
    variant_params: pd.DataFrame,
    comparison_by_expiry: pd.DataFrame,
) -> pd.DataFrame:
    """Build the Stage 5.5 plain-language diagnostic summary table."""

    bs_summary = black_scholes_limit_summary(bs_limit)
    stage5 = _variant_row(variant_params, "global_unconstrained_stage5")
    improved = _variant_row(variant_params, "global_multistart_unconstrained")
    regularized = _variant_row(variant_params, "global_regularized")
    per_expiry = variant_params[variant_params["variant_name"] == "per_expiry_heston"]
    feller_variants = variant_params[
        variant_params["variant_name"].str.startswith("global_feller_penalty")
    ]
    best_feller = feller_variants.sort_values("weighted_rmse_iv").head(1)
    max_quad_diff = float(quadrature["max_abs_iv_difference_from_reference"].max())
    any_heston_beats_sabr = bool(
        (comparison_by_expiry["winner_by_rmse"] == "heston").any()
    )
    per_expiry_rmse = (
        float(per_expiry["rmse_iv"].mean()) if not per_expiry.empty else np.nan
    )
    synthetic_rmse = float(synthetic_recovery.loc[0, "synthetic_rmse_iv"])
    stage5_weighted = float(stage5["weighted_rmse_iv"])
    improved_weighted = float(improved["weighted_rmse_iv"])
    regularized_warnings = regularized["parameter_warnings"]
    rows = [
        {
            "diagnostic": "Black-Scholes limit",
            "result": "pass" if bs_summary["max_price_error"] < 1e-3 else "review",
            "key_metric_before": "",
            "key_metric_after": f"max_price_error={bs_summary['max_price_error']:.6g}; "
            f"median_iv_error={bs_summary['median_iv_error']:.6g}",
            "interpretation": (
                "Heston prices approach Black-Scholes; max IV error is affected "
                "by near-zero extrinsic rows."
            ),
        },
        {
            "diagnostic": "Synthetic Heston recovery",
            "result": "pass" if synthetic_rmse < 1e-3 else "directional",
            "key_metric_before": "generated from known Heston parameters",
            "key_metric_after": f"rmse_iv={synthetic_rmse:.6g}",
            "interpretation": (
                "The pricing-to-calibration loop recovers a synthetic Heston "
                "smile directionally."
            ),
        },
        {
            "diagnostic": "Quadrature sensitivity",
            "result": "stable" if max_quad_diff < 1e-4 else "sensitive",
            "key_metric_before": "32 and 64 nodes",
            "key_metric_after": f"max_diff_vs_96={max_quad_diff:.6g}",
            "interpretation": (
                "Node-count sensitivity is measured against the 96-node reference."
            ),
        },
        {
            "diagnostic": "Improved multi-start global Heston",
            "result": (
                "improved" if improved_weighted < stage5_weighted else "not improved"
            ),
            "key_metric_before": f"stage5_weighted_rmse={stage5_weighted:.6g}",
            "key_metric_after": f"multistart_weighted_rmse={improved_weighted:.6g}",
            "interpretation": (
                "Tests whether Stage 5 was materially initialization-limited."
            ),
        },
        {
            "diagnostic": "Feller-penalized Heston",
            "result": (
                "stabilized"
                if not best_feller.empty
                and bool(best_feller.iloc[0]["feller_satisfied"])
                else "still violates Feller"
            ),
            "key_metric_before": f"stage5_feller={stage5['feller_satisfied']}",
            "key_metric_after": _feller_summary(best_feller),
            "interpretation": (
                "Soft Feller penalties test parameter stability versus fit quality."
            ),
        },
        {
            "diagnostic": "Regularized Heston",
            "result": (
                "stable"
                if regularized["parameter_warnings"] == "none"
                else "warnings remain"
            ),
            "key_metric_before": f"stage5_warnings={stage5['parameter_warnings']}",
            "key_metric_after": f"regularized_warnings={regularized_warnings}",
            "interpretation": (
                "Mild regularization tests whether plausible parameters can fit "
                "nearly as well."
            ),
        },
        {
            "diagnostic": "Per-expiry Heston",
            "result": (
                "improved" if per_expiry_rmse < stage5["rmse_iv"] else "not improved"
            ),
            "key_metric_before": f"stage5_global_rmse={stage5['rmse_iv']:.6g}",
            "key_metric_after": f"mean_per_expiry_rmse={per_expiry_rmse:.6g}",
            "interpretation": (
                "Per-expiry Heston tests global-model rigidity against SABR's "
                "per-expiry flexibility."
            ),
        },
        {
            "diagnostic": "Heston versus SABR",
            "result": (
                "heston beats sabr"
                if any_heston_beats_sabr
                else "sabr remains lower RMSE"
            ),
            "key_metric_before": "Stage 5 SABR dominated global Heston",
            "key_metric_after": f"heston_wins={int(any_heston_beats_sabr)} expiries",
            "interpretation": (
                "This is a single-snapshot diagnostic, not a universal model ranking."
            ),
        },
        {
            "diagnostic": "Surviving conclusion",
            "result": "diagnostic",
            "key_metric_before": "",
            "key_metric_after": "",
            "interpretation": (
                "The underfit is not explained by basic pricing failure; global "
                "rigidity and parameter stability are the main diagnostics to inspect."
            ),
        },
    ]
    return pd.DataFrame(rows, columns=IMPROVEMENT_SUMMARY_COLUMNS)


def _fit_points_for_params(
    surface: pd.DataFrame,
    params: np.ndarray,
    weights: np.ndarray,
    quadrature_order: int,
) -> pd.DataFrame:
    return build_heston_fit_points(
        surface,
        params,
        weights,
        quadrature_order=quadrature_order,
    )


def _global_metrics(fit_points: pd.DataFrame) -> dict[str, float]:
    if fit_points.empty:
        return {
            "rmse_iv": np.nan,
            "mae_iv": np.nan,
            "weighted_rmse_iv": np.nan,
            "max_abs_error": np.nan,
        }
    successful = fit_points[fit_points["iv_status"] == "success"]
    if successful.empty:
        return {
            "rmse_iv": np.nan,
            "mae_iv": np.nan,
            "weighted_rmse_iv": np.nan,
            "max_abs_error": np.nan,
        }
    errors = successful["iv_error"].to_numpy(dtype=float)
    abs_errors = np.abs(errors)
    weights = successful["weight"].to_numpy(dtype=float)
    return {
        "rmse_iv": float(np.sqrt(np.mean(errors**2))),
        "mae_iv": float(np.mean(abs_errors)),
        "weighted_rmse_iv": float(np.sqrt(np.average(errors**2, weights=weights))),
        "max_abs_error": float(np.max(abs_errors)),
    }


def _variant_param_row(
    variant_name: str,
    expiry: object,
    params: np.ndarray,
    objective: float,
    metrics: dict[str, float],
    runtime_seconds: float,
    optimizer_success: bool,
    optimizer_message: str,
    nfev: int,
) -> dict[str, object]:
    return {
        "variant_name": variant_name,
        "expiry": expiry,
        "v0": float(params[0]),
        "theta": float(params[1]),
        "kappa": float(params[2]),
        "sigma": float(params[3]),
        "rho": float(params[4]),
        "objective": objective,
        **metrics,
        "feller_satisfied": _safe_feller(params),
        "parameter_warnings": parameter_warnings(params),
        "runtime_seconds": runtime_seconds,
        "optimizer_success": optimizer_success,
        "optimizer_message": optimizer_message,
        "nfev": nfev,
    }


def _penalty_residuals(
    params: np.ndarray,
    feller_lambda: float,
    regularization_lambda: float,
) -> np.ndarray:
    residuals = []
    if feller_lambda > 0.0:
        _, theta, kappa, sigma, _ = params
        gap = max(float(sigma**2 - 2.0 * kappa * theta), 0.0)
        residuals.append(np.sqrt(feller_lambda) * FELLER_GAP_SCALE * gap)
    if regularization_lambda > 0.0:
        residuals.extend(
            np.sqrt(regularization_lambda) * _regularization_residuals(params)
        )
    return np.asarray(residuals, dtype=float)


def _regularization_residuals(params: np.ndarray) -> np.ndarray:
    v0, theta, kappa, sigma, rho = params
    return np.asarray(
        [
            max((0.005 - theta) / 0.005, 0.0),
            max((abs(rho) - 0.9) / 0.099, 0.0),
            max((sigma - 2.0) / 8.0, 0.0),
            max((kappa - 20.0) / 30.0, 0.0),
            abs(float(np.log(v0 / theta))) / 2.0,
        ],
        dtype=float,
    )


def _implied_vols_for_surface(prices: np.ndarray, surface: pd.DataFrame) -> np.ndarray:
    ivs, success, _ = _implied_vols_from_forward_prices(np.asarray(prices), surface)
    return np.where(success, ivs, np.nan)


def _safe_implied_vol(
    price: float,
    forward: float,
    strike: float,
    time_to_expiry: float,
    option_type: str,
) -> float:
    try:
        return float(
            implied_vol_bisection(
                price,
                forward,
                strike,
                time_to_expiry,
                0.0,
                option_type,
            )
        )
    except (RuntimeError, ValueError, OverflowError, FloatingPointError):
        return np.nan


def _base_variance(surface: pd.DataFrame) -> float:
    atm = surface[surface["log_moneyness"].abs() <= 0.01]
    base_iv = float(atm["implied_vol"].median())
    if not np.isfinite(base_iv):
        base_iv = float(surface["implied_vol"].median())
    return float(np.clip(base_iv**2, 1e-4, 1.0))


def _params_to_array(params: dict[str, float]) -> np.ndarray:
    return np.asarray([params[name] for name in PARAMETER_ORDER], dtype=float)


def _safe_feller(params: np.ndarray) -> bool:
    if not np.all(np.isfinite(params)):
        return False
    return feller_condition(float(params[2]), float(params[1]), float(params[3]))


def _point_keys() -> list[str]:
    return ["expiry", "option_type", "strike"]


def _expiry_label(surface: pd.DataFrame) -> object:
    if "expiry" not in surface or surface.empty:
        return ""
    unique = surface["expiry"].unique()
    return unique[0] if len(unique) == 1 else "all"


def _variant_rmse_by_group(
    variant_fit_points: pd.DataFrame,
    sabr_group: pd.DataFrame,
) -> dict[str, float]:
    if len(sabr_group) == 0 or variant_fit_points.empty:
        return {}
    metrics = {}
    keys = _point_keys()
    target_keys = sabr_group.loc[:, keys]
    for variant, group in variant_fit_points.groupby("variant_name", sort=True):
        merged = target_keys.merge(group, on=keys, how="inner")
        if len(merged):
            metrics[variant] = _rmse(merged["iv_error"])
    return metrics


def _variant_bias_by_group(
    variant_fit_points: pd.DataFrame,
    sabr_group: pd.DataFrame,
) -> dict[str, float]:
    if len(sabr_group) == 0 or variant_fit_points.empty:
        return {}
    metrics = {}
    keys = _point_keys()
    target_keys = sabr_group.loc[:, keys]
    for variant, group in variant_fit_points.groupby("variant_name", sort=True):
        merged = target_keys.merge(group, on=keys, how="inner")
        if len(merged):
            metrics[variant] = _mean(merged["iv_error"])
    return metrics


def _best_feller_metric(variant_metrics: dict[str, float]) -> float:
    values = {
        name: value
        for name, value in variant_metrics.items()
        if name.startswith("global_feller_penalty")
    }
    return _best_metric(values)[1]


def _best_metric(metrics: dict[str, float]) -> tuple[str, float]:
    finite = {name: value for name, value in metrics.items() if np.isfinite(value)}
    if not finite:
        return "not_available", np.nan
    name = min(finite, key=finite.get)
    return name, finite[name]


def _rmse(values: pd.Series) -> float:
    if len(values) == 0:
        return np.nan
    errors = values.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(errors**2)))


def _mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else np.nan


def _variant_row(variant_params: pd.DataFrame, variant_name: str) -> pd.Series:
    rows = variant_params[variant_params["variant_name"] == variant_name]
    if rows.empty:
        return pd.Series(dtype=object)
    return rows.sort_values("weighted_rmse_iv").iloc[0]


def _feller_summary(best_feller: pd.DataFrame) -> str:
    if best_feller.empty:
        return "no feller variant"
    row = best_feller.iloc[0]
    return (
        f"{row['variant_name']} weighted_rmse={row['weighted_rmse_iv']:.6g}; "
        f"feller={row['feller_satisfied']}"
    )


def _failure(
    category: str,
    severity: str,
    variant_name: str,
    expiry: object,
    detail: str,
) -> dict[str, object]:
    return {
        "category": category,
        "severity": severity,
        "variant_name": variant_name,
        "expiry": expiry,
        "detail": detail,
    }


def _stability_tradeoff_warnings(
    variant_params: pd.DataFrame,
) -> list[dict[str, object]]:
    rows = []
    stage5 = _variant_row(variant_params, "global_unconstrained_stage5")
    if stage5.empty:
        return rows
    stage5_rmse = float(stage5["weighted_rmse_iv"])
    stage5_warning_count = _warning_count(stage5["parameter_warnings"])
    for row in variant_params.itertuples(index=False):
        warning_count = _warning_count(row.parameter_warnings)
        if row.weighted_rmse_iv < stage5_rmse and warning_count > stage5_warning_count:
            rows.append(
                _failure(
                    "fit_improved_stability_worse",
                    "warning",
                    row.variant_name,
                    row.expiry,
                    "weighted RMSE improved but warning count increased",
                )
            )
        if row.weighted_rmse_iv > stage5_rmse and warning_count < stage5_warning_count:
            rows.append(
                _failure(
                    "stability_improved_fit_worse",
                    "warning",
                    row.variant_name,
                    row.expiry,
                    "warning count improved but weighted RMSE worsened",
                )
            )
    return rows


def _warning_count(warnings: str) -> int:
    if warnings == "none" or not isinstance(warnings, str):
        return 0
    return len([part for part in warnings.split(";") if part])
