from __future__ import annotations

import numpy as np
import pandas as pd

from volsurface.black_scholes import black_scholes_price
from volsurface.heston_calibration import (
    HESTON_ERROR_COLUMNS,
    HESTON_FIT_POINT_COLUMNS,
    HESTON_PARAM_BOUNDS,
    build_heston_failure_report,
    build_heston_fit_points,
    calibrate_heston_global,
    heston_bound_hits,
    summarize_heston_fit_errors,
)


def _surface() -> pd.DataFrame:
    forward = 100.0
    time_to_expiry = 0.75
    rows = []
    for strike in [85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0]:
        option_type = "put" if strike < forward else "call"
        vol = 0.22 + 0.001 * abs(strike - forward)
        rows.append(
            {
                "expiry": "2026-09-18",
                "option_type": option_type,
                "strike": strike,
                "forward": forward,
                "time_to_expiry": time_to_expiry,
                "log_moneyness": np.log(strike / forward),
                "option_side_for_surface": "atm" if strike == forward else option_type,
                "implied_vol": vol,
                "mid": black_scholes_price(
                    forward,
                    strike,
                    time_to_expiry,
                    0.0,
                    vol,
                    option_type,
                ),
                "bid": 1.0,
                "ask": 1.2,
            }
        )
    return pd.DataFrame(rows)


def test_build_heston_fit_points_and_error_summary() -> None:
    surface = _surface()
    params = np.asarray([0.04, 0.04, 2.0, 0.5, -0.4])
    weights = np.ones(len(surface))

    fit_points = build_heston_fit_points(
        surface,
        params,
        weights,
        quadrature_order=16,
    )
    errors = summarize_heston_fit_errors(fit_points)

    assert list(fit_points.columns) == HESTON_FIT_POINT_COLUMNS
    assert list(errors.columns) == HESTON_ERROR_COLUMNS
    assert set(fit_points["pricing_status"]) == {"success"}
    assert set(fit_points["iv_status"]) == {"success"}
    assert errors.loc[0, "successful_points"] == len(surface)


def test_calibrate_heston_global_returns_diagnostics() -> None:
    result = calibrate_heston_global(
        _surface(),
        weighting="unweighted",
        quadrature_order=16,
        max_nfev=2,
    )

    row = result.params.iloc[0]
    assert HESTON_PARAM_BOUNDS["v0"][0] <= row["v0"] <= HESTON_PARAM_BOUNDS["v0"][1]
    assert HESTON_PARAM_BOUNDS["rho"][0] <= row["rho"] <= HESTON_PARAM_BOUNDS["rho"][1]
    assert result.fit_errors.loc[0, "total_points"] == len(_surface())
    assert result.runtime_summary.loc[0, "heston_price_eval_count"] > 0


def test_heston_failure_report_includes_parameter_warnings() -> None:
    params = pd.DataFrame(
        [
            {
                "v0": 0.05,
                "theta": 0.01,
                "kappa": 1.0,
                "sigma": 0.5,
                "rho": -0.4,
                "optimizer_success": True,
                "optimizer_message": "`ftol` termination condition is satisfied.",
                "v0_at_bound": False,
                "theta_at_bound": False,
                "kappa_at_bound": False,
                "sigma_at_bound": False,
                "rho_at_bound": False,
                "feller_condition": False,
            }
        ]
    )
    fit_points = build_heston_fit_points(
        _surface(),
        np.asarray([0.04, 0.04, 2.0, 0.5, -0.4]),
        np.ones(7),
        quadrature_order=16,
    )

    report = build_heston_failure_report(params, fit_points)

    assert "feller_violation" in set(report["category"])
    assert "suspicious_parameter" in set(report["category"])


def test_heston_bound_hits() -> None:
    hits = heston_bound_hits(
        {"v0": 1e-4, "theta": 0.04, "kappa": 50.0, "sigma": 0.5, "rho": 0.0}
    )

    assert hits["v0_at_bound"]
    assert hits["kappa_at_bound"]
    assert not hits["rho_at_bound"]
