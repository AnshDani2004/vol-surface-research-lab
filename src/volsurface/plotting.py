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
