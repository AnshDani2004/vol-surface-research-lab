"""Generate deterministic Stage 5.5 Heston improvement diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from volsurface.heston_diagnostics import (
    VARIANT_PARAMS_COLUMNS,
    build_improvement_summary,
    build_variant_comparison_by_expiry,
    build_variant_comparison_by_moneyness,
    build_variant_failure_report,
    calibrate_heston_variant,
    calibrate_per_expiry_heston,
    compact_variant_guesses,
    default_multistart_guesses,
    parameter_warnings,
    run_black_scholes_limit_diagnostic,
    run_quadrature_sensitivity,
    run_synthetic_heston_recovery,
)
from volsurface.plotting import save_heston_improvement_plots

INPUT_PATH = Path(
    "outputs/stage4_5_sabr_robustness/recommended_calibration_universe.csv"
)
STAGE5_DIR = Path("outputs/stage5_heston")
OUTPUT_DIR = Path("outputs/stage5_5_heston_improvement")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    surface = pd.read_csv(INPUT_PATH)
    sabr_points = pd.read_csv(STAGE5_DIR / "sabr_recommended_fit_errors.csv")

    bs_limit = run_black_scholes_limit_diagnostic()
    synthetic_recovery, synthetic_fit_points = run_synthetic_heston_recovery()
    stage5_params, stage5_fit_errors, stage5_fit_points = _load_stage5_baseline()
    stage5_param_dict = _param_dict(stage5_params.iloc[0])
    quadrature = run_quadrature_sensitivity(surface, stage5_param_dict)

    multistart = calibrate_heston_variant(
        surface,
        "global_multistart_unconstrained",
        default_multistart_guesses(surface),
        max_nfev=20,
    )
    variant_results = [multistart]
    for penalty_lambda in [1.0, 10.0, 100.0]:
        variant_results.append(
            calibrate_heston_variant(
                surface,
                f"global_feller_penalty_{int(penalty_lambda)}",
                compact_variant_guesses(surface),
                max_nfev=25,
                feller_lambda=penalty_lambda,
            )
        )
    variant_results.append(
        calibrate_heston_variant(
            surface,
            "global_regularized",
            compact_variant_guesses(surface),
            max_nfev=25,
            regularization_lambda=1.0,
        )
    )
    per_expiry = calibrate_per_expiry_heston(surface, max_nfev=25)
    variant_results.append(per_expiry)

    variant_params = pd.concat(
        [stage5_params, *[result.params for result in variant_results]],
        ignore_index=True,
    )
    variant_fit_errors = pd.concat(
        [stage5_fit_errors, *[result.fit_errors for result in variant_results]],
        ignore_index=True,
    )
    variant_fit_points = pd.concat(
        [stage5_fit_points, *[result.fit_points for result in variant_results]],
        ignore_index=True,
    )
    multistart_summary = multistart.attempts.copy()

    comparison_by_expiry = build_variant_comparison_by_expiry(
        sabr_points,
        variant_fit_points,
    )
    comparison_by_moneyness = build_variant_comparison_by_moneyness(
        sabr_points,
        variant_fit_points,
    )
    failure_report = pd.concat(
        [
            *[result.failure_report for result in variant_results],
            build_variant_failure_report(
                variant_params,
                variant_fit_points,
                sabr_points=sabr_points,
                comparison_by_expiry=comparison_by_expiry,
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()
    improvement_summary = build_improvement_summary(
        bs_limit,
        synthetic_recovery,
        quadrature,
        variant_params,
        comparison_by_expiry,
    )

    bs_limit.to_csv(OUTPUT_DIR / "heston_black_scholes_limit.csv", index=False)
    synthetic_recovery.to_csv(
        OUTPUT_DIR / "synthetic_heston_recovery.csv",
        index=False,
    )
    quadrature.to_csv(OUTPUT_DIR / "quadrature_sensitivity.csv", index=False)
    multistart_summary.to_csv(
        OUTPUT_DIR / "heston_multistart_summary.csv",
        index=False,
    )
    variant_params.to_csv(OUTPUT_DIR / "heston_variant_params.csv", index=False)
    variant_fit_errors.to_csv(
        OUTPUT_DIR / "heston_variant_fit_errors.csv",
        index=False,
    )
    variant_fit_points.to_csv(
        OUTPUT_DIR / "heston_variant_fit_points.csv",
        index=False,
    )
    comparison_by_expiry.to_csv(
        OUTPUT_DIR / "heston_variant_comparison_by_expiry.csv",
        index=False,
    )
    comparison_by_moneyness.to_csv(
        OUTPUT_DIR / "heston_variant_comparison_by_moneyness.csv",
        index=False,
    )
    failure_report.to_csv(
        OUTPUT_DIR / "heston_variant_failure_report.csv",
        index=False,
    )
    improvement_summary.to_csv(
        OUTPUT_DIR / "heston_improvement_summary.csv",
        index=False,
    )
    save_heston_improvement_plots(
        variant_fit_points,
        variant_params,
        comparison_by_expiry,
        comparison_by_moneyness,
        synthetic_fit_points,
        quadrature,
        OUTPUT_DIR / "plots",
    )


def _load_stage5_baseline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    params = pd.read_csv(STAGE5_DIR / "heston_params.csv")
    errors = pd.read_csv(STAGE5_DIR / "heston_fit_errors.csv")
    points = pd.read_csv(STAGE5_DIR / "heston_fit_points.csv")
    param_row = params.iloc[0]
    global_errors = errors[errors["scope"] == "global"].iloc[0]
    variant_params = pd.DataFrame(
        [
            {
                "variant_name": "global_unconstrained_stage5",
                "expiry": "all",
                "v0": param_row["v0"],
                "theta": param_row["theta"],
                "kappa": param_row["kappa"],
                "sigma": param_row["sigma"],
                "rho": param_row["rho"],
                "objective": param_row["objective_value"],
                "rmse_iv": global_errors["rmse_iv"],
                "mae_iv": global_errors["mae_iv"],
                "weighted_rmse_iv": global_errors["weighted_rmse_iv"],
                "max_abs_error": global_errors["max_abs_error"],
                "feller_satisfied": bool(param_row["feller_condition"]),
                "parameter_warnings": parameter_warnings(
                    np.asarray(
                        [
                            param_row["v0"],
                            param_row["theta"],
                            param_row["kappa"],
                            param_row["sigma"],
                            param_row["rho"],
                        ],
                        dtype=float,
                    )
                ),
                "runtime_seconds": param_row["runtime_seconds"],
                "optimizer_success": bool(param_row["optimizer_success"]),
                "optimizer_message": param_row["optimizer_message"],
                "nfev": int(param_row["nfev"]),
            }
        ],
        columns=VARIANT_PARAMS_COLUMNS,
    )
    variant_errors = errors.copy()
    variant_errors.insert(0, "variant_name", "global_unconstrained_stage5")
    variant_points = points.copy()
    variant_points.insert(0, "variant_name", "global_unconstrained_stage5")
    return variant_params, variant_errors, variant_points


def _param_dict(row: pd.Series) -> dict[str, float]:
    return {
        "v0": float(row["v0"]),
        "theta": float(row["theta"]),
        "kappa": float(row["kappa"]),
        "sigma": float(row["sigma"]),
        "rho": float(row["rho"]),
    }


if __name__ == "__main__":
    main()
