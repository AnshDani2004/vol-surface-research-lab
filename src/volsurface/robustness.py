"""SABR robustness scenarios and summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from volsurface.calibration import SabrCalibrationResult, calibrate_sabr_per_expiry

TRADING_DAYS_PER_YEAR = 252.0
DEFAULT_BETA = 0.5
DEFAULT_WEIGHTING = "spread_weighted"
DEFAULT_MIN_POINTS = 6

EXPIRY_FILTERS = {
    "all_expiries": 0.0,
    "exclude_T_lt_1_trading_day": 1.0 / TRADING_DAYS_PER_YEAR,
    "exclude_T_lt_2_trading_days": 2.0 / TRADING_DAYS_PER_YEAR,
    "exclude_T_lt_5_trading_days": 5.0 / TRADING_DAYS_PER_YEAR,
}

LIQUIDITY_FILTERS = [
    "base_stage4",
    "relative_spread_lte_0_25",
    "relative_spread_lte_0_10",
    "volume_gt_0",
    "open_interest_gt_0",
    "volume_gt_0_and_open_interest_gt_0",
    "relative_spread_lte_0_25_and_volume_gt_0_and_open_interest_gt_0",
]

WARNING_COLUMNS = [
    "alpha_near_bound",
    "rho_near_bound",
    "rho_near_plus_minus_one",
    "nu_near_bound",
    "high_nu_gt_5",
    "max_abs_error_gt_0_25",
    "optimizer_failure",
    "too_few_points",
]

ROBUSTNESS_GRID_COLUMNS = [
    "scenario_group",
    "scenario",
    "expiry",
    "beta",
    "weighting",
    "n_points",
    "alpha",
    "rho",
    "nu",
    "rmse_iv",
    "mae_iv",
    "weighted_rmse_iv",
    "max_abs_error",
    "optimizer_success",
    "status",
    "failure_reason",
    *WARNING_COLUMNS,
]

ROBUSTNESS_SUMMARY_COLUMNS = [
    "scenario_group",
    "scenario",
    "beta",
    "weighting",
    "fit_points",
    "expiries_attempted",
    "successful_calibrations",
    "skipped_calibrations",
    "median_rmse_iv",
    "median_mae_iv",
    "median_weighted_rmse_iv",
    "max_max_abs_error",
    "total_warnings",
    "bound_hit_warnings",
    "high_nu_warnings",
    "max_abs_error_warnings",
]

FILTERED_UNIVERSE_SUMMARY_COLUMNS = [
    "base_rows",
    "recommended_rows",
    "rows_removed",
    "expiries_retained",
    "min_expiry",
    "max_expiry",
    "min_time_to_expiry",
    "max_time_to_expiry",
    "otm_put_count",
    "atm_count",
    "otm_call_count",
    "median_relative_spread",
    "median_volume",
    "median_open_interest",
]


@dataclass(frozen=True)
class RobustnessOutputs:
    """Container for Stage 4.5 robustness outputs."""

    robustness_grid: pd.DataFrame
    robustness_summary: pd.DataFrame
    expiry_filter_sensitivity: pd.DataFrame
    liquidity_filter_sensitivity: pd.DataFrame
    beta_sensitivity: pd.DataFrame
    weighting_sensitivity: pd.DataFrame
    recommended_universe: pd.DataFrame
    filtered_universe_summary: pd.DataFrame
    warning_summary: pd.DataFrame
    base_fit_points: pd.DataFrame
    recommended_fit_points: pd.DataFrame


def apply_expiry_filter(surface: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Apply one named expiry filter scenario."""

    if scenario not in EXPIRY_FILTERS:
        raise ValueError(f"unknown expiry filter scenario: {scenario}")
    threshold = EXPIRY_FILTERS[scenario]
    return surface[surface["time_to_expiry"] >= threshold].copy()


def apply_liquidity_filter(surface: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Apply one named liquidity filter scenario."""

    if scenario not in LIQUIDITY_FILTERS:
        raise ValueError(f"unknown liquidity filter scenario: {scenario}")
    filtered = surface.copy()
    if scenario == "base_stage4":
        return filtered
    if "relative_spread_lte_0_25" in scenario:
        filtered = filtered[filtered["relative_spread"] <= 0.25]
    if "relative_spread_lte_0_10" in scenario:
        filtered = filtered[filtered["relative_spread"] <= 0.10]
    if "volume_gt_0" in scenario:
        filtered = filtered[filtered["volume"] > 0]
    if "open_interest_gt_0" in scenario:
        filtered = filtered[filtered["open_interest"] > 0]
    return filtered.copy()


def select_recommended_universe(surface: pd.DataFrame) -> pd.DataFrame:
    """Select the deterministic recommended calibration universe for Stage 5."""

    recommended = apply_expiry_filter(surface, "exclude_T_lt_2_trading_days")
    recommended = recommended[recommended["relative_spread"] <= 0.25]
    recommended = recommended[recommended["volume"] > 0]
    recommended = recommended[recommended["open_interest"] > 0]
    return recommended.sort_values(["expiry", "strike", "option_type"]).reset_index(
        drop=True
    )


def run_sabr_robustness(surface: pd.DataFrame) -> RobustnessOutputs:
    """Run all Stage 4.5 SABR robustness scenarios."""

    required = {
        "expiry",
        "time_to_expiry",
        "relative_spread",
        "volume",
        "open_interest",
    }
    missing = required.difference(surface.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"missing columns for SABR robustness: {missing_text}")

    grids: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []

    base_result = _scenario(
        surface,
        "base",
        "base_stage4",
        DEFAULT_BETA,
        DEFAULT_WEIGHTING,
    )
    grids.append(base_result[0])
    summaries.append(base_result[1])
    base_fit_points = base_result[2].fit_points

    expiry_summaries = []
    for scenario in EXPIRY_FILTERS:
        scenario_surface = apply_expiry_filter(surface, scenario)
        grid, summary, _ = _scenario(
            scenario_surface,
            "expiry_filter",
            scenario,
            DEFAULT_BETA,
            DEFAULT_WEIGHTING,
        )
        grids.append(grid)
        summaries.append(summary)
        expiry_summaries.append(summary)

    liquidity_summaries = []
    for scenario in LIQUIDITY_FILTERS:
        scenario_surface = apply_liquidity_filter(surface, scenario)
        grid, summary, _ = _scenario(
            scenario_surface,
            "liquidity_filter",
            scenario,
            DEFAULT_BETA,
            DEFAULT_WEIGHTING,
        )
        summary = _with_filter_counts(summary, len(surface), len(scenario_surface))
        grids.append(grid)
        summaries.append(summary)
        liquidity_summaries.append(summary)

    recommended = select_recommended_universe(surface)
    recommended_base_grid, recommended_base_summary, recommended_result = _scenario(
        recommended,
        "recommended_universe",
        "recommended",
        DEFAULT_BETA,
        DEFAULT_WEIGHTING,
    )
    grids.append(recommended_base_grid)
    summaries.append(recommended_base_summary)

    beta_sensitivity = []
    universes = (("base_stage4", surface), ("recommended", recommended))
    for universe_name, universe in universes:
        for beta in (0.0, 0.5, 1.0):
            grid, summary, _ = _scenario(
                universe,
                "beta_sensitivity",
                universe_name,
                beta,
                DEFAULT_WEIGHTING,
            )
            grids.append(grid)
            summaries.append(summary)
            beta_sensitivity.append(grid)

    weighting_sensitivity = []
    for universe_name, universe in universes:
        for weighting in ("unweighted", "spread_weighted", "vega_weighted"):
            grid, summary, _ = _scenario(
                universe,
                "weighting_sensitivity",
                universe_name,
                DEFAULT_BETA,
                weighting,
            )
            grids.append(grid)
            summaries.append(summary)
            weighting_sensitivity.append(grid)

    nonempty_grids = [grid for grid in grids if not grid.empty]
    robustness_grid = (
        pd.concat(nonempty_grids, ignore_index=True)
        if nonempty_grids
        else pd.DataFrame(columns=ROBUSTNESS_GRID_COLUMNS)
    )
    robustness_summary = pd.concat(summaries, ignore_index=True)
    warning_summary = build_warning_summary(robustness_grid)

    return RobustnessOutputs(
        robustness_grid=robustness_grid.loc[:, ROBUSTNESS_GRID_COLUMNS],
        robustness_summary=robustness_summary,
        expiry_filter_sensitivity=pd.concat(expiry_summaries, ignore_index=True),
        liquidity_filter_sensitivity=pd.concat(liquidity_summaries, ignore_index=True),
        beta_sensitivity=pd.concat(beta_sensitivity, ignore_index=True),
        weighting_sensitivity=pd.concat(weighting_sensitivity, ignore_index=True),
        recommended_universe=recommended,
        filtered_universe_summary=summarize_filtered_universe(surface, recommended),
        warning_summary=warning_summary,
        base_fit_points=base_fit_points,
        recommended_fit_points=recommended_result.fit_points,
    )


def build_warning_summary(grid: pd.DataFrame) -> pd.DataFrame:
    """Build long-form warning counts by scenario."""

    rows = []
    grouped = grid.groupby(
        ["scenario_group", "scenario", "beta", "weighting"],
        sort=True,
    )
    for keys, group in grouped:
        scenario_group, scenario, beta, weighting = keys
        for warning in WARNING_COLUMNS:
            rows.append(
                {
                    "scenario_group": scenario_group,
                    "scenario": scenario,
                    "beta": beta,
                    "weighting": weighting,
                    "warning": warning,
                    "count": int(group[warning].sum()),
                }
            )
    return pd.DataFrame(rows)


def summarize_filtered_universe(
    base: pd.DataFrame,
    recommended: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the recommended calibration universe."""

    side_counts = recommended["option_side_for_surface"].value_counts()
    row = {
        "base_rows": int(len(base)),
        "recommended_rows": int(len(recommended)),
        "rows_removed": int(len(base) - len(recommended)),
        "expiries_retained": int(recommended["expiry"].nunique()),
        "min_expiry": _safe_min(recommended["expiry"]),
        "max_expiry": _safe_max(recommended["expiry"]),
        "min_time_to_expiry": _safe_min(recommended["time_to_expiry"]),
        "max_time_to_expiry": _safe_max(recommended["time_to_expiry"]),
        "otm_put_count": int(side_counts.get("otm_put", 0)),
        "atm_count": int(side_counts.get("atm", 0)),
        "otm_call_count": int(side_counts.get("otm_call", 0)),
        "median_relative_spread": _safe_median(recommended["relative_spread"]),
        "median_volume": _safe_median(recommended["volume"]),
        "median_open_interest": _safe_median(recommended["open_interest"]),
    }
    return pd.DataFrame([row], columns=FILTERED_UNIVERSE_SUMMARY_COLUMNS)


def _scenario(
    surface: pd.DataFrame,
    scenario_group: str,
    scenario: str,
    beta: float,
    weighting: str,
) -> tuple[pd.DataFrame, pd.DataFrame, SabrCalibrationResult]:
    result = calibrate_sabr_per_expiry(
        surface,
        beta=beta,
        weighting=weighting,
        min_points=DEFAULT_MIN_POINTS,
    )
    grid = _params_to_grid(result.params, scenario_group, scenario, beta, weighting)
    summary = _summarize_grid(grid, scenario_group, scenario, beta, weighting)
    return grid, summary, result


def _params_to_grid(
    params: pd.DataFrame,
    scenario_group: str,
    scenario: str,
    beta: float,
    weighting: str,
) -> pd.DataFrame:
    grid = params.copy()
    grid["scenario_group"] = scenario_group
    grid["scenario"] = scenario
    grid["beta"] = beta
    grid["weighting"] = weighting
    grid["alpha_near_bound"] = grid["alpha_at_bound"].fillna(False).astype(bool)
    grid["rho_near_bound"] = grid["rho_at_bound"].fillna(False).astype(bool)
    grid["rho_near_plus_minus_one"] = grid["rho"].abs() > 0.95
    grid["nu_near_bound"] = grid["nu_at_bound"].fillna(False).astype(bool)
    grid["high_nu_gt_5"] = grid["nu"] > 5.0
    grid["max_abs_error_gt_0_25"] = grid["max_abs_error"] > 0.25
    grid["optimizer_failure"] = ~grid["optimizer_success"].fillna(False).astype(bool)
    grid["too_few_points"] = grid["status"] == "skipped"
    return grid.loc[:, ROBUSTNESS_GRID_COLUMNS]


def _summarize_grid(
    grid: pd.DataFrame,
    scenario_group: str,
    scenario: str,
    beta: float,
    weighting: str,
) -> pd.DataFrame:
    calibrated = grid[grid["status"] == "calibrated"]
    row = {
        "scenario_group": scenario_group,
        "scenario": scenario,
        "beta": beta,
        "weighting": weighting,
        "fit_points": int(grid["n_points"].fillna(0).sum()),
        "expiries_attempted": int(len(grid)),
        "successful_calibrations": int((grid["status"] == "calibrated").sum()),
        "skipped_calibrations": int((grid["status"] == "skipped").sum()),
        "median_rmse_iv": _safe_median(calibrated["rmse_iv"]),
        "median_mae_iv": _safe_median(calibrated["mae_iv"]),
        "median_weighted_rmse_iv": _safe_median(calibrated["weighted_rmse_iv"]),
        "max_max_abs_error": _safe_max(calibrated["max_abs_error"]),
        "total_warnings": int(grid[WARNING_COLUMNS].sum(axis=1).sum()),
        "bound_hit_warnings": int(
            grid[["alpha_near_bound", "rho_near_bound", "nu_near_bound"]]
            .sum(axis=1)
            .sum()
        ),
        "high_nu_warnings": int(grid["high_nu_gt_5"].sum()),
        "max_abs_error_warnings": int(grid["max_abs_error_gt_0_25"].sum()),
    }
    return pd.DataFrame([row], columns=ROBUSTNESS_SUMMARY_COLUMNS)


def _with_filter_counts(
    summary: pd.DataFrame,
    rows_before: int,
    rows_after: int,
) -> pd.DataFrame:
    result = summary.copy()
    result.insert(4, "rows_before", int(rows_before))
    result.insert(5, "rows_after", int(rows_after))
    result.insert(6, "rows_removed", int(rows_before - rows_after))
    result.insert(7, "expiries_retained", int(result["expiries_attempted"].iloc[0]))
    return result


def _safe_min(values: pd.Series) -> object:
    return values.min() if not values.empty else np.nan


def _safe_max(values: pd.Series) -> object:
    return values.max() if not values.empty else np.nan


def _safe_median(values: pd.Series) -> float:
    return float(values.median()) if not values.empty else np.nan
