from __future__ import annotations

import numpy as np
import pandas as pd

from volsurface.heston_diagnostics import (
    IMPROVEMENT_SUMMARY_COLUMNS,
    VARIANT_COMPARISON_BUCKET_COLUMNS,
    VARIANT_COMPARISON_EXPIRY_COLUMNS,
    VARIANT_PARAMS_COLUMNS,
    black_scholes_limit_summary,
    build_improvement_summary,
    build_variant_comparison_by_expiry,
    build_variant_comparison_by_moneyness,
    build_variant_failure_report,
    feller_penalty,
    generate_synthetic_heston_surface,
    regularization_penalty,
    run_black_scholes_limit_diagnostic,
    run_quadrature_sensitivity,
    run_synthetic_heston_recovery,
)


def test_black_scholes_limit_diagnostic_has_small_fixture_errors() -> None:
    result = run_black_scholes_limit_diagnostic(
        forwards=[100.0],
        strikes=[90.0, 100.0, 110.0],
        maturities=[0.25],
        variance_levels=[0.04],
        quadrature_order=32,
    )
    summary = black_scholes_limit_summary(result)

    assert summary["max_price_error"] < 1e-3
    assert summary["max_iv_error"] < 1e-3


def test_synthetic_heston_surface_generation_returns_finite_ivs() -> None:
    surface = generate_synthetic_heston_surface(
        maturities=[0.1],
        log_moneyness=[-0.04, 0.0, 0.04],
        quadrature_order=32,
    )

    assert len(surface) == 3
    assert np.isfinite(surface["implied_vol"]).all()
    assert (surface["implied_vol"] > 0.0).all()


def test_synthetic_recovery_runner_returns_required_columns() -> None:
    recovery, fit_points = run_synthetic_heston_recovery(
        quadrature_order=16,
        max_nfev=2,
    )

    required = {
        "true_v0",
        "recovered_v0",
        "abs_error_v0",
        "synthetic_rmse_iv",
        "optimizer_success",
        "runtime_seconds",
    }
    assert required.issubset(recovery.columns)
    assert {"market_iv", "heston_iv", "iv_error"}.issubset(fit_points.columns)


def test_quadrature_sensitivity_runner_returns_required_columns() -> None:
    surface = generate_synthetic_heston_surface(
        maturities=[0.1],
        log_moneyness=[-0.04, 0.0, 0.04],
        quadrature_order=32,
    )
    params = {"v0": 0.04, "theta": 0.04, "kappa": 2.0, "sigma": 0.6, "rho": -0.5}

    result = run_quadrature_sensitivity(surface, params, node_counts=[16, 32])

    required = {
        "node_count",
        "reference_node_count",
        "mean_abs_iv_difference_from_reference",
        "max_abs_iv_difference_from_reference",
        "pricing_failures",
        "iv_failures",
    }
    assert required.issubset(result.columns)
    assert len(result) == 2


def test_feller_penalty_zero_when_condition_is_satisfied() -> None:
    params = np.asarray([0.04, 0.04, 2.0, 0.2, -0.5])

    assert feller_penalty(params, penalty_lambda=10.0) == 0.0


def test_feller_penalty_positive_when_condition_is_violated() -> None:
    params = np.asarray([0.04, 0.01, 1.0, 0.6, -0.5])

    assert feller_penalty(params, penalty_lambda=10.0) > 0.0


def test_regularization_penalty_is_deterministic() -> None:
    params = np.asarray([0.05, 0.005, 30.0, 3.0, 0.97])

    first = regularization_penalty(params, penalty_lambda=1.5)
    second = regularization_penalty(params, penalty_lambda=1.5)

    assert first == second
    assert first > 0.0


def test_variant_comparison_tables_have_required_columns() -> None:
    sabr_points = _sabr_points()
    variants = _variant_points()

    by_expiry = build_variant_comparison_by_expiry(sabr_points, variants)
    by_moneyness = build_variant_comparison_by_moneyness(sabr_points, variants)

    assert list(by_expiry.columns) == VARIANT_COMPARISON_EXPIRY_COLUMNS
    assert list(by_moneyness.columns) == VARIANT_COMPARISON_BUCKET_COLUMNS
    assert by_expiry.loc[0, "best_heston_variant"] == "per_expiry_heston"


def test_improvement_summary_has_required_columns() -> None:
    bs_limit = run_black_scholes_limit_diagnostic(
        forwards=[100.0],
        strikes=[100.0],
        maturities=[0.25],
        variance_levels=[0.04],
    )
    synthetic = pd.DataFrame(
        [{"synthetic_rmse_iv": 1e-5, "synthetic_mae_iv": 1e-5}]
    )
    quadrature = pd.DataFrame(
        [{"max_abs_iv_difference_from_reference": 1e-6}]
    )
    params = _variant_params()
    comparison = build_variant_comparison_by_expiry(_sabr_points(), _variant_points())

    summary = build_improvement_summary(
        bs_limit,
        synthetic,
        quadrature,
        params,
        comparison,
    )

    assert list(summary.columns) == IMPROVEMENT_SUMMARY_COLUMNS
    assert "Surviving conclusion" in set(summary["diagnostic"])


def test_failure_report_is_generated() -> None:
    params = _variant_params()
    fit_points = _variant_points()
    comparison = build_variant_comparison_by_expiry(_sabr_points(), fit_points)

    report = build_variant_failure_report(
        params,
        fit_points,
        sabr_points=_sabr_points(),
        comparison_by_expiry=comparison,
    )

    assert not report.empty
    assert {"category", "severity", "variant_name", "expiry", "detail"}.issubset(
        report.columns
    )


def _sabr_points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_type": ["point"] * 4,
            "expiry": ["2026-08-21"] * 4,
            "option_type": ["put", "put", "call", "call"],
            "strike": [95.0, 99.0, 101.0, 105.0],
            "log_moneyness": [-0.05, -0.01, 0.01, 0.05],
            "iv_error": [0.002, -0.003, 0.001, 0.004],
        }
    )


def _variant_points() -> pd.DataFrame:
    rows = []
    for variant, error_scale in [
        ("global_unconstrained_stage5", 0.05),
        ("global_multistart_unconstrained", 0.04),
        ("global_regularized", 0.06),
        ("global_feller_penalty_1", 0.045),
        ("per_expiry_heston", 0.03),
    ]:
        for strike, log_moneyness, option_type in [
            (95.0, -0.05, "put"),
            (99.0, -0.01, "put"),
            (101.0, 0.01, "call"),
            (105.0, 0.05, "call"),
        ]:
            rows.append(
                {
                    "variant_name": variant,
                    "expiry": "2026-08-21",
                    "option_type": option_type,
                    "strike": strike,
                    "log_moneyness": log_moneyness,
                    "iv_error": error_scale,
                    "pricing_status": "success",
                    "iv_status": "success",
                }
            )
    return pd.DataFrame(rows)


def _variant_params() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant_name": "global_unconstrained_stage5",
                "expiry": "all",
                "v0": 0.03,
                "theta": 0.0001,
                "kappa": 10.0,
                "sigma": 0.4,
                "rho": -0.5,
                "objective": 1.0,
                "rmse_iv": 0.05,
                "mae_iv": 0.04,
                "weighted_rmse_iv": 0.05,
                "max_abs_error": 0.1,
                "feller_satisfied": False,
                "parameter_warnings": "feller_violation",
                "runtime_seconds": 1.0,
                "optimizer_success": True,
                "optimizer_message": "ok",
                "nfev": 5,
            },
            {
                "variant_name": "global_multistart_unconstrained",
                "expiry": "all",
                "v0": 0.03,
                "theta": 0.01,
                "kappa": 10.0,
                "sigma": 0.4,
                "rho": -0.5,
                "objective": 1.0,
                "rmse_iv": 0.04,
                "mae_iv": 0.03,
                "weighted_rmse_iv": 0.04,
                "max_abs_error": 0.09,
                "feller_satisfied": True,
                "parameter_warnings": "none",
                "runtime_seconds": 1.0,
                "optimizer_success": True,
                "optimizer_message": "ok",
                "nfev": 5,
            },
            {
                "variant_name": "global_regularized",
                "expiry": "all",
                "v0": 0.03,
                "theta": 0.02,
                "kappa": 10.0,
                "sigma": 0.2,
                "rho": -0.4,
                "objective": 1.0,
                "rmse_iv": 0.06,
                "mae_iv": 0.05,
                "weighted_rmse_iv": 0.06,
                "max_abs_error": 0.11,
                "feller_satisfied": True,
                "parameter_warnings": "none",
                "runtime_seconds": 1.0,
                "optimizer_success": True,
                "optimizer_message": "ok",
                "nfev": 5,
            },
            {
                "variant_name": "global_feller_penalty_1",
                "expiry": "all",
                "v0": 0.03,
                "theta": 0.01,
                "kappa": 10.0,
                "sigma": 0.3,
                "rho": -0.5,
                "objective": 1.0,
                "rmse_iv": 0.045,
                "mae_iv": 0.035,
                "weighted_rmse_iv": 0.045,
                "max_abs_error": 0.1,
                "feller_satisfied": True,
                "parameter_warnings": "none",
                "runtime_seconds": 1.0,
                "optimizer_success": True,
                "optimizer_message": "ok",
                "nfev": 5,
            },
            {
                "variant_name": "per_expiry_heston",
                "expiry": "2026-08-21",
                "v0": 0.03,
                "theta": 0.02,
                "kappa": 10.0,
                "sigma": 0.3,
                "rho": -0.5,
                "objective": 1.0,
                "rmse_iv": 0.03,
                "mae_iv": 0.02,
                "weighted_rmse_iv": 0.03,
                "max_abs_error": 0.08,
                "feller_satisfied": True,
                "parameter_warnings": "none",
                "runtime_seconds": 1.0,
                "optimizer_success": True,
                "optimizer_message": "ok",
                "nfev": 5,
            },
        ],
        columns=VARIANT_PARAMS_COLUMNS,
    )
