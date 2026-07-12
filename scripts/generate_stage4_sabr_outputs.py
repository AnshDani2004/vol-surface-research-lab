"""Generate deterministic Stage 4 SABR calibration outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from volsurface.calibration import calibrate_sabr_per_expiry
from volsurface.plotting import save_sabr_diagnostic_plots

INPUT_PATH = Path("outputs/stage3_real_snapshot/otm_surface_table.csv")
OUTPUT_DIR = Path("outputs/stage4_sabr")
BETA = 0.5
WEIGHTING = "spread_weighted"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    surface = pd.read_csv(INPUT_PATH)
    result = calibrate_sabr_per_expiry(
        surface,
        beta=BETA,
        weighting=WEIGHTING,
    )

    fit_errors = result.params.loc[
        :,
        [
            "expiry",
            "T",
            "n_points",
            "beta",
            "rmse_iv",
            "mae_iv",
            "weighted_rmse_iv",
            "max_abs_error",
            "status",
            "failure_reason",
        ],
    ]
    result.params.to_csv(OUTPUT_DIR / "sabr_params.csv", index=False)
    fit_errors.to_csv(OUTPUT_DIR / "sabr_fit_errors.csv", index=False)
    result.fit_points.to_csv(OUTPUT_DIR / "sabr_fit_points.csv", index=False)
    result.failure_report.to_csv(
        OUTPUT_DIR / "sabr_failure_report.csv",
        index=False,
    )
    save_sabr_diagnostic_plots(
        result.fit_points,
        result.params,
        OUTPUT_DIR / "plots",
    )


if __name__ == "__main__":
    main()

