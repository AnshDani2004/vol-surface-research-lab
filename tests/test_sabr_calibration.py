from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volsurface.calibration import (
    SABR_FAILURE_COLUMNS,
    SABR_FIT_POINT_COLUMNS,
    calibrate_sabr_per_expiry,
)
from volsurface.sabr import hagan_sabr_iv


def _synthetic_surface(
    alpha: float = 2.0,
    beta: float = 0.5,
    rho: float = -0.25,
    nu: float = 0.8,
    expiry: str = "2026-09-18",
) -> pd.DataFrame:
    forward = 100.0
    time_to_expiry = 0.75
    strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
    rows = []
    for strike in strikes:
        option_type = "put" if strike < forward else "call"
        side = "atm" if strike == forward else f"otm_{option_type}"
        rows.append(
            {
                "expiry": pd.Timestamp(expiry),
                "option_type": option_type,
                "strike": strike,
                "forward": forward,
                "time_to_expiry": time_to_expiry,
                "log_moneyness": pd.NA,
                "implied_vol": hagan_sabr_iv(
                    forward,
                    strike,
                    time_to_expiry,
                    alpha,
                    beta,
                    rho,
                    nu,
                ),
                "bid": 1.0,
                "ask": 1.2,
                "option_side_for_surface": side,
            }
        )
    surface = pd.DataFrame(rows)
    surface["log_moneyness"] = np.log(surface["strike"] / surface["forward"])
    return surface


def test_calibration_recovers_known_synthetic_sabr_parameters() -> None:
    surface = _synthetic_surface(alpha=2.0, rho=-0.25, nu=0.8)

    result = calibrate_sabr_per_expiry(surface, beta=0.5, weighting="unweighted")
    row = result.params.iloc[0]

    assert row["status"] == "calibrated"
    assert row["alpha"] == pytest.approx(2.0, rel=1e-3)
    assert row["rho"] == pytest.approx(-0.25, abs=1e-3)
    assert row["nu"] == pytest.approx(0.8, rel=1e-3)
    assert row["rmse_iv"] < 1e-6


def test_calibration_skips_expiry_with_too_few_points() -> None:
    surface = _synthetic_surface().head(5)

    result = calibrate_sabr_per_expiry(surface, beta=0.5, min_points=6)

    assert result.params.loc[0, "status"] == "skipped"
    assert "too few valid points" in result.params.loc[0, "failure_reason"]
    assert result.fit_points.empty
    assert result.failure_report.loc[0, "reason"].startswith("too few")


def test_bound_hit_diagnostics_work() -> None:
    surface = _synthetic_surface(alpha=5.0, rho=0.0, nu=1e-4)

    result = calibrate_sabr_per_expiry(surface, beta=0.5, weighting="unweighted")

    assert bool(result.params.loc[0, "alpha_at_bound"])
    assert "alpha near calibration bound" in set(result.failure_report["reason"])


def test_fit_error_table_has_expected_columns() -> None:
    surface = _synthetic_surface()

    result = calibrate_sabr_per_expiry(surface, beta=0.5)

    assert list(result.fit_points.columns) == SABR_FIT_POINT_COLUMNS
    assert len(result.fit_points) == len(surface)


def test_failure_report_is_generated_with_expected_columns() -> None:
    surface = _synthetic_surface().head(2)

    result = calibrate_sabr_per_expiry(surface, beta=0.5)

    assert list(result.failure_report.columns) == SABR_FAILURE_COLUMNS
    assert len(result.failure_report) == 1
