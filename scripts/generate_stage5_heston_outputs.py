"""Generate deterministic Stage 5 Heston calibration outputs."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd

from volsurface.heston_calibration import calibrate_heston_global
from volsurface.model_comparison import (
    build_model_comparison_by_expiry,
    build_model_comparison_by_moneyness,
    build_underperformance_warnings,
    recompute_sabr_on_universe,
)
from volsurface.plotting import save_heston_diagnostic_plots

INPUT_PATH = Path(
    "outputs/stage4_5_sabr_robustness/recommended_calibration_universe.csv"
)
OUTPUT_DIR = Path("outputs/stage5_heston")


def main() -> None:
    total_start = perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    surface = pd.read_csv(INPUT_PATH)

    heston_result = calibrate_heston_global(surface, weighting="spread_weighted")

    sabr_start = perf_counter()
    sabr_combined, sabr_points = recompute_sabr_on_universe(
        surface,
        beta=0.5,
        weighting="spread_weighted",
    )
    sabr_runtime = perf_counter() - sabr_start

    comparison_by_expiry = build_model_comparison_by_expiry(
        sabr_points,
        heston_result.fit_points,
    )
    comparison_by_moneyness = build_model_comparison_by_moneyness(
        sabr_points,
        heston_result.fit_points,
    )
    underperformance = build_underperformance_warnings(
        comparison_by_expiry,
        comparison_by_moneyness,
    )
    failure_report = pd.concat(
        [heston_result.failure_report, underperformance],
        ignore_index=True,
    )

    runtime_summary = heston_result.runtime_summary.copy()
    runtime_summary["sabr_recompute_runtime_seconds"] = sabr_runtime
    runtime_summary["total_stage5_runtime_seconds"] = perf_counter() - total_start

    heston_result.params.to_csv(OUTPUT_DIR / "heston_params.csv", index=False)
    heston_result.fit_points.to_csv(OUTPUT_DIR / "heston_fit_points.csv", index=False)
    heston_result.fit_errors.to_csv(OUTPUT_DIR / "heston_fit_errors.csv", index=False)
    failure_report.to_csv(OUTPUT_DIR / "heston_failure_report.csv", index=False)
    sabr_combined.to_csv(OUTPUT_DIR / "sabr_recommended_fit_errors.csv", index=False)
    comparison_by_expiry.to_csv(
        OUTPUT_DIR / "model_comparison_by_expiry.csv",
        index=False,
    )
    comparison_by_moneyness.to_csv(
        OUTPUT_DIR / "model_comparison_by_moneyness.csv",
        index=False,
    )
    runtime_summary.to_csv(OUTPUT_DIR / "runtime_summary.csv", index=False)
    save_heston_diagnostic_plots(
        heston_result.fit_points,
        sabr_points,
        comparison_by_expiry,
        comparison_by_moneyness,
        OUTPUT_DIR / "plots",
    )


if __name__ == "__main__":
    main()
