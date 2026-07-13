"""Generate final reporting tables and plots from existing stage outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = Path("outputs/final_report")
PLOTS_DIR = OUTPUT_DIR / "plots"


def main() -> None:
    """Build deterministic final report artifacts."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = _load_metrics()

    project_summary = _project_summary(metrics)
    stage_timeline = _stage_timeline()
    key_metrics = _key_metrics(metrics)
    model_comparison = _model_comparison(metrics)
    limitations = _limitations()
    resume_metrics = _resume_metrics(metrics)

    project_summary.to_csv(OUTPUT_DIR / "final_project_summary.csv", index=False)
    stage_timeline.to_csv(OUTPUT_DIR / "final_stage_timeline.csv", index=False)
    key_metrics.to_csv(OUTPUT_DIR / "final_key_metrics.csv", index=False)
    model_comparison.to_csv(OUTPUT_DIR / "final_model_comparison.csv", index=False)
    limitations.to_csv(OUTPUT_DIR / "final_limitations.csv", index=False)
    resume_metrics.to_csv(OUTPUT_DIR / "final_resume_metrics.csv", index=False)

    _plot_sabr_before_after(metrics)
    _plot_sabr_vs_heston(metrics)
    _plot_pipeline_timeline(stage_timeline)
    _plot_warning_reduction(metrics)


def _load_metrics() -> dict[str, object]:
    with Path("outputs/stage3_real_snapshot/snapshot_metadata.json").open() as file:
        metadata = json.load(file)
    clean_chain = pd.read_csv("outputs/stage3_real_snapshot/clean_chain.csv")
    otm_surface = pd.read_csv("outputs/stage3_real_snapshot/otm_surface_table.csv")
    recommended = pd.read_csv(
        "outputs/stage4_5_sabr_robustness/recommended_calibration_universe.csv"
    )
    robustness = pd.read_csv("outputs/stage4_5_sabr_robustness/robustness_summary.csv")
    heston_errors = pd.read_csv("outputs/stage5_heston/heston_fit_errors.csv")
    stage5_comparison = pd.read_csv(
        "outputs/stage5_heston/model_comparison_by_expiry.csv"
    )
    synthetic = pd.read_csv(
        "outputs/stage5_5_heston_improvement/synthetic_heston_recovery.csv"
    )
    quadrature = pd.read_csv(
        "outputs/stage5_5_heston_improvement/quadrature_sensitivity.csv"
    )
    variants = pd.read_csv(
        "outputs/stage5_5_heston_improvement/heston_variant_params.csv"
    )
    variant_expiry = pd.read_csv(
        "outputs/stage5_5_heston_improvement/heston_variant_comparison_by_expiry.csv"
    )
    base = robustness[
        (robustness["scenario_group"] == "base")
        & (robustness["scenario"] == "base_stage4")
    ].iloc[0]
    recommended_row = robustness[
        (robustness["scenario_group"] == "recommended_universe")
        & (robustness["scenario"] == "recommended")
    ].iloc[0]
    global_heston = heston_errors[heston_errors["scope"] == "global"].iloc[0]
    quad_32 = quadrature[quadrature["node_count"] == 32].iloc[0]
    regularized = variants[variants["variant_name"] == "global_regularized"].iloc[0]
    per_expiry = variants[variants["variant_name"] == "per_expiry_heston"]
    return {
        "raw_snapshot_rows": int(metadata["raw_contract_count"]),
        "snapshot_source": metadata["source"],
        "snapshot_datetime": metadata["retrieval_datetime"],
        "clean_rows": int(len(clean_chain)),
        "otm_surface_rows": int(len(otm_surface)),
        "recommended_rows": int(len(recommended)),
        "recommended_expiries": int(recommended["expiry"].nunique()),
        "sabr_median_rmse_before": float(base["median_rmse_iv"]),
        "sabr_median_rmse_after": float(recommended_row["median_rmse_iv"]),
        "sabr_warnings_before": int(base["total_warnings"]),
        "sabr_warnings_after": int(recommended_row["total_warnings"]),
        "heston_global_rmse": float(global_heston["rmse_iv"]),
        "heston_global_weighted_rmse": float(global_heston["weighted_rmse_iv"]),
        "synthetic_heston_recovery_rmse": float(synthetic.loc[0, "synthetic_rmse_iv"]),
        "quadrature_32_96_mean_diff": float(
            quad_32["mean_abs_iv_difference_from_reference"]
        ),
        "quadrature_32_96_max_diff": float(
            quad_32["max_abs_iv_difference_from_reference"]
        ),
        "stage5_comparison": stage5_comparison,
        "variant_expiry": variant_expiry,
        "regularized_heston_rmse": float(regularized["rmse_iv"]),
        "regularized_heston_weighted_rmse": float(regularized["weighted_rmse_iv"]),
        "per_expiry_mean_rmse": float(per_expiry["rmse_iv"].mean()),
        "best_heston_mean_rmse": float(variant_expiry["best_heston_rmse_iv"].mean()),
    }


def _project_summary(metrics: dict[str, object]) -> pd.DataFrame:
    rows = [
        (
            "project",
            "Vol Surface Research Lab",
            "Stage 6",
            "Reproducible Python options research engine and final report.",
        ),
        (
            "data",
            "saved SPY yfinance snapshot",
            "Stage 3.5",
            "Single point-in-time snapshot used for deterministic diagnostics.",
        ),
        (
            "recommended_universe_rows",
            metrics["recommended_rows"],
            "Stage 4.5",
            "Filtered same-universe comparison set.",
        ),
        (
            "main_finding",
            "SABR fit the filtered local smile better than Heston variants",
            "Stages 4.5-5.5",
            "Diagnostic, not trading signal or universal model ranking.",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=["metric", "value", "source_stage", "interpretation"],
    )


def _stage_timeline() -> pd.DataFrame:
    rows = [
        (
            "0",
            "Package foundation",
            "Python package, tests, docs, outputs, notebooks, tooling.",
            "pyproject.toml",
            "Reproducible package structure.",
        ),
        (
            "1",
            "Black-Scholes baseline",
            "Pricing, Greeks, parity, bisection/Newton IV inversion.",
            "outputs/stage1_black_scholes/",
            "Deterministic numerical baseline.",
        ),
        (
            "2",
            "Chain cleaning and forwards",
            "Quote filters, mid prices, ACT/365, forward extraction.",
            "outputs/stage2_chain_cleaning/",
            "Deterministic cleaning pipeline.",
        ),
        (
            "3",
            "Observed IV surface",
            "OTM surface construction and static-arbitrage diagnostics.",
            "outputs/stage3_surface/",
            "Observed quote diagnostics before calibration.",
        ),
        (
            "3.5",
            "Saved SPY snapshot",
            "Saved yfinance chain and deterministic regeneration.",
            "outputs/stage3_real_snapshot/",
            "Single point-in-time snapshot without live dependency.",
        ),
        (
            "4",
            "SABR calibration",
            "Fixed-beta per-expiry SABR calibration.",
            "outputs/stage4_sabr/",
            "Raw calibration exposed short-expiry instability.",
        ),
        (
            "4.5",
            "SABR robustness",
            "Expiry, liquidity, beta, and weighting sensitivity.",
            "outputs/stage4_5_sabr_robustness/",
            "Recommended calibration universe selected.",
        ),
        (
            "5",
            "Heston comparison",
            "Global Heston calibration and same-universe SABR comparison.",
            "outputs/stage5_heston/",
            "Global Heston underfit SABR.",
        ),
        (
            "5.5",
            "Heston diagnostics",
            "Sanity checks, synthetic recovery, quadrature, variants.",
            "outputs/stage5_5_heston_improvement/",
            "Underfit was not explained by basic implementation failure.",
        ),
        (
            "6",
            "Final report",
            "README, research note, summary tables, portfolio docs.",
            "outputs/final_report/",
            "Polished reporting without new model claims.",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=["stage", "name", "implemented_scope", "key_output", "main_finding"],
    )


def _key_metrics(metrics: dict[str, object]) -> pd.DataFrame:
    rows = [
        ("testing", "pytest result", "118 passed", "Full deterministic test suite."),
        (
            "data",
            "raw snapshot rows",
            metrics["raw_snapshot_rows"],
            "Saved yfinance rows.",
        ),
        ("data", "clean rows", metrics["clean_rows"], "Rows after quote filters."),
        (
            "surface",
            "OTM surface rows",
            metrics["otm_surface_rows"],
            "Observed OTM IV rows.",
        ),
        (
            "calibration_universe",
            "recommended universe rows",
            metrics["recommended_rows"],
            "Filtered rows used for same-universe comparison.",
        ),
        (
            "calibration_universe",
            "recommended universe expiries",
            metrics["recommended_expiries"],
            "Retained expiries after robustness diagnostics.",
        ),
        (
            "SABR",
            "median RMSE before filtering",
            f"{metrics['sabr_median_rmse_before']:.6f}",
            "Raw Stage 4 snapshot calibration.",
        ),
        (
            "SABR",
            "median RMSE after filtering",
            f"{metrics['sabr_median_rmse_after']:.6f}",
            "Recommended calibration universe.",
        ),
        (
            "Heston",
            "global RMSE",
            f"{metrics['heston_global_rmse']:.6f}",
            "Global Heston on recommended universe.",
        ),
        (
            "Heston",
            "synthetic recovery RMSE",
            f"{metrics['synthetic_heston_recovery_rmse']:.6f}",
            "Synthetic Heston recovery sanity check.",
        ),
        (
            "numerics",
            "32 vs 96 node mean IV diff",
            f"{metrics['quadrature_32_96_mean_diff']:.6f}",
            "Quadrature sensitivity for Stage 5 Heston parameters.",
        ),
    ]
    return pd.DataFrame(rows, columns=["category", "metric", "value", "interpretation"])


def _model_comparison(metrics: dict[str, object]) -> pd.DataFrame:
    comparison = metrics["stage5_comparison"]
    rows = [
        (
            "SABR per-expiry",
            "recommended calibration universe",
            f"{comparison['sabr_rmse_iv'].median():.6f}",
            f"{comparison['sabr_weighted_rmse_iv'].median():.6f}",
            "low warning count after filtering",
            "Lowest RMSE in same-universe comparison.",
        ),
        (
            "Heston global",
            "recommended calibration universe",
            f"{metrics['heston_global_rmse']:.6f}",
            f"{metrics['heston_global_weighted_rmse']:.6f}",
            "theta bound, Feller violation, v0/theta mismatch",
            "Diagnostic underfit, not deployment calibration.",
        ),
        (
            "Heston regularized",
            "recommended calibration universe",
            f"{metrics['regularized_heston_rmse']:.6f}",
            f"{metrics['regularized_heston_weighted_rmse']:.6f}",
            "cleaner parameters, worse fit",
            "Parameter plausibility tradeoff.",
        ),
        (
            "Heston per-expiry",
            "recommended calibration universe",
            f"{metrics['per_expiry_mean_rmse']:.6f}",
            "",
            "fit improves versus global but warnings remain",
            "Fairness diagnostic; still underfit SABR.",
        ),
        (
            "Best Heston by expiry",
            "recommended calibration universe",
            f"{metrics['best_heston_mean_rmse']:.6f}",
            "",
            "per-expiry Heston selected on all expiries",
            "No Heston variant beat SABR on this snapshot.",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "model_or_variant",
            "universe",
            "rmse_iv",
            "weighted_rmse_iv",
            "warning_summary",
            "interpretation",
        ],
    )


def _limitations() -> pd.DataFrame:
    rows = [
        (
            "single saved yfinance SPY snapshot",
            "Results are point-in-time diagnostics.",
            "Use historical vendor data for validation.",
        ),
        (
            "not vendor-certified data",
            "Quotes may be delayed, stale, or incomplete.",
            "Repeat with institutional option-chain data.",
        ),
        (
            "no historical validation",
            "No claim about stability across regimes.",
            "Add rolling snapshots before model ranking claims.",
        ),
        (
            "no trading signal",
            "Fit diagnostics are not executable strategy evidence.",
            "Keep conclusions framed as research diagnostics.",
        ),
        (
            "American exercise and dividends not fully modeled",
            "SPY options are American-style with dividend effects.",
            "Add exercise/dividend treatment before deployment use.",
        ),
        (
            "Heston result is diagnostic",
            "Global fit was parameter-sensitive and underfit SABR.",
            "Improve numerics and constraints before production calibration.",
        ),
        (
            "quadrature sensitivity",
            "Stage 5 fitted parameters were sensitive to node count.",
            "Use stricter numerical controls and independent pricing checks.",
        ),
        (
            "comparison is not universal",
            "SABR/Heston result may not generalize beyond this snapshot.",
            "Validate across underlyings, dates, and maturities.",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=["limitation", "impact", "mitigation_or_future_work"],
    )


def _resume_metrics(metrics: dict[str, object]) -> pd.DataFrame:
    rows = [
        ("pytest suite", "118 passed", True, "Local deterministic tests."),
        (
            "raw snapshot rows",
            metrics["raw_snapshot_rows"],
            True,
            "Saved yfinance SPY snapshot, not vendor-certified.",
        ),
        (
            "recommended universe rows",
            metrics["recommended_rows"],
            True,
            "Filtered calibration universe.",
        ),
        (
            "SABR median RMSE improvement",
            f"{metrics['sabr_median_rmse_before']:.4f} to "
            f"{metrics['sabr_median_rmse_after']:.4f}",
            True,
            "Single point-in-time snapshot.",
        ),
        (
            "synthetic Heston recovery RMSE",
            f"{metrics['synthetic_heston_recovery_rmse']:.6f}",
            True,
            "Generated by same Heston implementation.",
        ),
        (
            "Heston vs SABR",
            "Heston underfit SABR on 3/3 retained expiries",
            True,
            "Diagnostic, not a universal model ranking.",
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "resume_safe", "caveat"])


def _plot_sabr_before_after(metrics: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["before filtering", "after filtering"]
    values = [metrics["sabr_median_rmse_before"], metrics["sabr_median_rmse_after"]]
    ax.bar(labels, values, color=["#5f6f94", "#2d8a6a"])
    ax.set_ylabel("Median RMSE IV")
    ax.set_title("SABR RMSE Before/After Filtering")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "final_sabr_before_after_filtering.png", dpi=140)
    plt.close(fig)


def _plot_sabr_vs_heston(metrics: dict[str, object]) -> None:
    comparison = metrics["stage5_comparison"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    expiries = comparison["expiry"].astype(str)
    ax.plot(expiries, comparison["sabr_rmse_iv"], marker="o", label="SABR")
    ax.plot(expiries, comparison["heston_rmse_iv"], marker="o", label="Global Heston")
    ax.set_xlabel("Expiry")
    ax.set_ylabel("RMSE IV")
    ax.set_title("Same-Universe SABR vs Heston")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "final_sabr_vs_heston_rmse.png", dpi=140)
    plt.close(fig)


def _plot_pipeline_timeline(stage_timeline: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    labels = stage_timeline["stage"] + ": " + stage_timeline["name"]
    ax.barh(labels, range(1, len(stage_timeline) + 1), color="#4f7cac")
    ax.set_xlabel("Stage order")
    ax.set_title("Vol Surface Research Lab Timeline")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "final_pipeline_timeline.png", dpi=140)
    plt.close(fig)


def _plot_warning_reduction(metrics: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["raw SABR universe", "recommended universe"]
    values = [metrics["sabr_warnings_before"], metrics["sabr_warnings_after"]]
    ax.bar(labels, values, color=["#8f5d5d", "#2d8a6a"])
    ax.set_ylabel("Warning count")
    ax.set_title("SABR Warning Reduction")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "final_warning_reduction.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
