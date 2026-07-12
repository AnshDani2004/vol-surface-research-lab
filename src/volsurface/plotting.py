"""Deterministic matplotlib plots for calibration diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_sabr_diagnostic_plots(
    fit_points: pd.DataFrame,
    params: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Save Stage 4 SABR diagnostic plots and return created paths."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = [
        output_path / "market_vs_sabr_by_expiry.png",
        output_path / "sabr_residuals_by_log_moneyness.png",
        output_path / "sabr_error_by_expiry.png",
    ]
    _plot_market_vs_sabr(fit_points, paths[0])
    _plot_residuals(fit_points, paths[1])
    _plot_error_by_expiry(params, paths[2])
    return paths


def save_sabr_robustness_plots(
    robustness_grid: pd.DataFrame,
    warning_summary: pd.DataFrame,
    beta_sensitivity: pd.DataFrame,
    weighting_sensitivity: pd.DataFrame,
    base_fit_points: pd.DataFrame,
    recommended_fit_points: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Save Stage 4.5 SABR robustness diagnostic plots."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = [
        output_path / "rmse_by_expiry_across_expiry_filters.png",
        output_path / "warning_counts_by_scenario.png",
        output_path / "beta_sensitivity_rmse.png",
        output_path / "weighting_sensitivity_rmse.png",
        output_path / "raw_vs_recommended_residuals.png",
    ]
    _plot_rmse_by_expiry_filters(robustness_grid, paths[0])
    _plot_warning_counts(warning_summary, paths[1])
    _plot_beta_sensitivity(beta_sensitivity, paths[2])
    _plot_weighting_sensitivity(weighting_sensitivity, paths[3])
    _plot_raw_vs_recommended_residuals(
        base_fit_points,
        recommended_fit_points,
        paths[4],
    )
    return paths


def _plot_market_vs_sabr(fit_points: pd.DataFrame, path: Path) -> None:
    fig, axes = _expiry_axes(fit_points)
    grouped = fit_points.groupby("expiry", sort=True)
    for ax, (expiry, group) in zip(axes, grouped, strict=False):
        group = group.sort_values("strike")
        ax.plot(
            group["strike"],
            group["market_iv"],
            marker="o",
            linewidth=1.0,
            label="market",
        )
        ax.plot(
            group["strike"],
            group["sabr_iv"],
            marker="x",
            linewidth=1.0,
            label="SABR",
        )
        ax.set_title(str(expiry))
        ax.set_xlabel("Strike")
        ax.set_ylabel("IV")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_residuals(fit_points: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for expiry, group in fit_points.groupby("expiry", sort=True):
        ax.scatter(group["log_moneyness"], group["iv_error"], s=14, label=str(expiry))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Log moneyness")
    ax.set_ylabel("SABR IV - market IV")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_error_by_expiry(params: pd.DataFrame, path: Path) -> None:
    calibrated = params[params["status"] == "calibrated"].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    expiries = calibrated["expiry"].astype(str)
    ax.plot(expiries, calibrated["rmse_iv"], marker="o", label="RMSE")
    ax.plot(expiries, calibrated["mae_iv"], marker="o", label="MAE")
    ax.plot(
        expiries,
        calibrated["max_abs_error"],
        marker="o",
        label="Max abs error",
    )
    ax.set_xlabel("Expiry")
    ax.set_ylabel("IV error")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _expiry_axes(fit_points: pd.DataFrame) -> tuple[plt.Figure, list[plt.Axes]]:
    n_expiries = fit_points["expiry"].nunique()
    n_cols = 2
    n_rows = max((n_expiries + n_cols - 1) // n_cols, 1)
    fig, axes_array = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = list(axes_array.flat) if hasattr(axes_array, "flat") else [axes_array]
    for ax in axes[n_expiries:]:
        ax.axis("off")
    return fig, axes


def _plot_rmse_by_expiry_filters(grid: pd.DataFrame, path: Path) -> None:
    data = grid[grid["scenario_group"] == "expiry_filter"].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    for scenario, group in data.groupby("scenario", sort=True):
        group = group.sort_values("expiry")
        ax.plot(
            group["expiry"].astype(str),
            group["rmse_iv"],
            marker="o",
            label=scenario,
        )
    ax.set_xlabel("Expiry")
    ax.set_ylabel("RMSE IV")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_warning_counts(warning_summary: pd.DataFrame, path: Path) -> None:
    data = warning_summary.copy()
    data["scenario_label"] = data["scenario_group"] + ":" + data["scenario"]
    totals = data.groupby("scenario_label")["count"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(totals.index, totals.values)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Warning count")
    ax.tick_params(axis="x", rotation=70)
    ax.grid(True, axis="y", alpha=0.3)
    fig.subplots_adjust(bottom=0.48)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_beta_sensitivity(beta_sensitivity: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    grouped = beta_sensitivity.groupby(["scenario", "beta"], sort=True)
    medians = grouped["rmse_iv"].median().reset_index()
    for scenario, group in medians.groupby("scenario", sort=True):
        ax.plot(group["beta"], group["rmse_iv"], marker="o", label=scenario)
    ax.set_xlabel("Beta")
    ax.set_ylabel("Median RMSE IV")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_weighting_sensitivity(
    weighting_sensitivity: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    grouped = weighting_sensitivity.groupby(["scenario", "weighting"], sort=True)
    medians = grouped["rmse_iv"].median().reset_index()
    for scenario, group in medians.groupby("scenario", sort=True):
        ax.plot(group["weighting"], group["rmse_iv"], marker="o", label=scenario)
    ax.set_xlabel("Weighting")
    ax.set_ylabel("Median RMSE IV")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_raw_vs_recommended_residuals(
    base_fit_points: pd.DataFrame,
    recommended_fit_points: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        base_fit_points["log_moneyness"],
        base_fit_points["iv_error"],
        s=12,
        alpha=0.45,
        label="base",
    )
    ax.scatter(
        recommended_fit_points["log_moneyness"],
        recommended_fit_points["iv_error"],
        s=16,
        alpha=0.75,
        label="recommended",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Log moneyness")
    ax.set_ylabel("SABR IV - market IV")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
