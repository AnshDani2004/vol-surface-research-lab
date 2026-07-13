"""Cross-date volatility-surface validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from volsurface.calibration import calibrate_sabr_per_expiry
from volsurface.filters import FILTER_REASONS, compute_mid_price, filter_option_quotes
from volsurface.forwards import add_forward_moneyness, pair_calls_and_puts_for_forward
from volsurface.heston_calibration import calibrate_heston_global
from volsurface.heston_diagnostics import calibrate_per_expiry_heston
from volsurface.market_data import load_snapshot
from volsurface.robustness import select_recommended_universe
from volsurface.surface import build_otm_surface_table, compute_implied_vol_table
from volsurface.time import compute_time_to_expiry

MANIFEST_COLUMNS = [
    "snapshot_id",
    "underlying",
    "snapshot_datetime",
    "spot",
    "source",
    "raw_path",
    "metadata_path",
    "market_regime",
    "notes",
]

SNAPSHOT_SUMMARY_COLUMNS = [
    "snapshot_id",
    "underlying",
    "snapshot_datetime",
    "source",
    "market_regime",
    "raw_row_count",
    "clean_row_count",
    "otm_surface_row_count",
    "filtered_calibration_row_count",
    "expiries_retained",
    "min_time_to_expiry",
    "max_time_to_expiry",
    "min_log_moneyness",
    "max_log_moneyness",
    "rows_removed_total",
    *[f"removed_{reason}" for reason in FILTER_REASONS],
]

AGGREGATE_SUMMARY_COLUMNS = [
    "metric",
    "value",
    "interpretation",
]

SABR_RESULT_COLUMNS = [
    "snapshot_id",
    "universe",
    "expiry",
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
    "warning_count",
]

HESTON_RESULT_COLUMNS = [
    "snapshot_id",
    "scope",
    "expiry",
    "n_points",
    "v0",
    "theta",
    "kappa",
    "sigma",
    "rho",
    "rmse_iv",
    "mae_iv",
    "weighted_rmse_iv",
    "max_abs_error",
    "optimizer_success",
    "optimizer_message",
    "feller_condition",
    "parameter_warnings",
    "status",
    "failure_reason",
]

MODEL_COMPARISON_COLUMNS = [
    "snapshot_id",
    "expiry",
    "n_points",
    "sabr_rmse_iv",
    "heston_scope",
    "heston_rmse_iv",
    "sabr_weighted_rmse_iv",
    "heston_weighted_rmse_iv",
    "winner_by_rmse",
    "interpretation",
]

FILTER_SENSITIVITY_COLUMNS = [
    "snapshot_id",
    "base_rows",
    "recommended_rows",
    "rows_removed",
    "base_median_rmse_iv",
    "recommended_median_rmse_iv",
    "rmse_change",
    "base_warning_count",
    "recommended_warning_count",
    "warning_count_change",
    "interpretation",
]


@dataclass(frozen=True)
class CrossDateSnapshotResult:
    """Container for one processed snapshot."""

    metadata: pd.Series
    clean_chain: pd.DataFrame
    filter_report: pd.DataFrame
    otm_surface: pd.DataFrame
    recommended_universe: pd.DataFrame
    snapshot_summary: pd.DataFrame
    sabr_results: pd.DataFrame
    heston_results: pd.DataFrame
    model_comparison: pd.DataFrame
    filter_sensitivity: pd.DataFrame


@dataclass(frozen=True)
class CrossDateOutputs:
    """Container for cross-date validation artifacts."""

    snapshot_summary: pd.DataFrame
    aggregate_summary: pd.DataFrame
    sabr_results: pd.DataFrame
    heston_results: pd.DataFrame
    model_comparison: pd.DataFrame
    filter_sensitivity: pd.DataFrame
    failure_modes_markdown: str
    research_note_markdown: str


def load_snapshot_manifest(
    manifest_path: str | Path,
    base_dir: str | Path = ".",
) -> pd.DataFrame:
    """Load and validate a multi-snapshot manifest."""

    manifest_path = Path(manifest_path)
    manifest = pd.read_csv(manifest_path)
    missing = set(MANIFEST_COLUMNS).difference(manifest.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"missing snapshot manifest columns: {missing_text}")
    if manifest["snapshot_id"].duplicated().any():
        raise ValueError("snapshot_id values must be unique")
    if manifest.empty:
        raise ValueError("snapshot manifest must contain at least one row")

    base = Path(base_dir)
    validated = manifest.loc[:, MANIFEST_COLUMNS].copy()
    validated["snapshot_datetime"] = pd.to_datetime(
        validated["snapshot_datetime"],
        errors="coerce",
        utc=True,
    )
    if validated["snapshot_datetime"].isna().any():
        raise ValueError("snapshot_datetime must parse for every snapshot")
    validated["spot"] = pd.to_numeric(validated["spot"], errors="coerce")
    if validated["spot"].isna().any():
        raise ValueError("spot must be numeric for every snapshot")
    for column in ("raw_path", "metadata_path"):
        paths = validated[column].map(lambda value: base / str(value))
        missing_paths = [str(path) for path in paths if not path.exists()]
        if missing_paths:
            raise ValueError(f"{column} paths do not exist: {', '.join(missing_paths)}")
    return validated


def run_cross_date_validation(
    manifest: pd.DataFrame,
    base_dir: str | Path = ".",
    risk_free_rate: float = 0.05,
    run_heston: bool = True,
    heston_max_nfev: int = 30,
) -> CrossDateOutputs:
    """Run deterministic cross-date validation over saved snapshots."""

    results = [
        process_snapshot(
            row,
            base_dir=base_dir,
            risk_free_rate=risk_free_rate,
            run_heston=run_heston,
            heston_max_nfev=heston_max_nfev,
        )
        for _, row in manifest.iterrows()
    ]
    snapshot_summary = pd.concat(
        [result.snapshot_summary for result in results],
        ignore_index=True,
    )
    sabr_results = pd.concat(
        [result.sabr_results for result in results],
        ignore_index=True,
    )
    heston_results = pd.concat(
        [result.heston_results for result in results],
        ignore_index=True,
    )
    model_comparison = pd.concat(
        [result.model_comparison for result in results],
        ignore_index=True,
    )
    filter_sensitivity = pd.concat(
        [result.filter_sensitivity for result in results],
        ignore_index=True,
    )
    aggregate_summary = build_aggregate_summary(
        snapshot_summary,
        sabr_results,
        heston_results,
        model_comparison,
    )
    failure_modes = build_failure_modes_markdown(
        snapshot_summary,
        sabr_results,
        heston_results,
        model_comparison,
    )
    research_note = build_cross_date_research_note(
        snapshot_summary,
        aggregate_summary,
        sabr_results,
        heston_results,
        model_comparison,
        filter_sensitivity,
    )
    return CrossDateOutputs(
        snapshot_summary=snapshot_summary,
        aggregate_summary=aggregate_summary,
        sabr_results=sabr_results,
        heston_results=heston_results,
        model_comparison=model_comparison,
        filter_sensitivity=filter_sensitivity,
        failure_modes_markdown=failure_modes,
        research_note_markdown=research_note,
    )


def process_snapshot(
    metadata: pd.Series,
    base_dir: str | Path = ".",
    risk_free_rate: float = 0.05,
    run_heston: bool = True,
    heston_max_nfev: int = 30,
) -> CrossDateSnapshotResult:
    """Process one saved snapshot through cleaning, SABR, and Heston diagnostics."""

    base = Path(base_dir)
    raw = load_snapshot(base / str(metadata["raw_path"]))
    clean_chain, filter_report, otm_surface = _build_surface(raw, risk_free_rate)
    recommended = select_recommended_universe(_enrich_surface(otm_surface, clean_chain))
    snapshot_summary = summarize_snapshot(
        metadata,
        raw,
        clean_chain,
        otm_surface,
        recommended,
        filter_report,
    )
    sabr_results, base_sabr_summary, recommended_sabr_summary = run_snapshot_sabr(
        str(metadata["snapshot_id"]),
        otm_surface,
        recommended,
    )
    heston_results = (
        run_snapshot_heston(
            str(metadata["snapshot_id"]),
            recommended,
            max_nfev=heston_max_nfev,
        )
        if run_heston
        else _skipped_heston_result(str(metadata["snapshot_id"]), "heston disabled")
    )
    model_comparison = build_model_comparison(
        str(metadata["snapshot_id"]),
        sabr_results,
        heston_results,
    )
    filter_sensitivity = build_filter_sensitivity(
        str(metadata["snapshot_id"]),
        otm_surface,
        recommended,
        base_sabr_summary,
        recommended_sabr_summary,
    )
    return CrossDateSnapshotResult(
        metadata=metadata,
        clean_chain=clean_chain,
        filter_report=filter_report,
        otm_surface=otm_surface,
        recommended_universe=recommended,
        snapshot_summary=snapshot_summary,
        sabr_results=sabr_results,
        heston_results=heston_results,
        model_comparison=model_comparison,
        filter_sensitivity=filter_sensitivity,
    )


def summarize_snapshot(
    metadata: pd.Series,
    raw: pd.DataFrame,
    clean_chain: pd.DataFrame,
    otm_surface: pd.DataFrame,
    recommended: pd.DataFrame,
    filter_report: pd.DataFrame,
) -> pd.DataFrame:
    """Build a one-row summary for a processed snapshot."""

    row = {
        "snapshot_id": metadata["snapshot_id"],
        "underlying": metadata["underlying"],
        "snapshot_datetime": metadata["snapshot_datetime"],
        "source": metadata["source"],
        "market_regime": metadata["market_regime"],
        "raw_row_count": int(len(raw)),
        "clean_row_count": int(len(clean_chain)),
        "otm_surface_row_count": int(len(otm_surface)),
        "filtered_calibration_row_count": int(len(recommended)),
        "expiries_retained": int(recommended["expiry"].nunique()),
        "min_time_to_expiry": _safe_min(recommended, "time_to_expiry"),
        "max_time_to_expiry": _safe_max(recommended, "time_to_expiry"),
        "min_log_moneyness": _safe_min(recommended, "log_moneyness"),
        "max_log_moneyness": _safe_max(recommended, "log_moneyness"),
        "rows_removed_total": int(len(raw) - len(clean_chain)),
    }
    counts = filter_report.set_index("reason")["count"].to_dict()
    for reason in FILTER_REASONS:
        row[f"removed_{reason}"] = int(counts.get(reason, 0))
    return pd.DataFrame([row], columns=SNAPSHOT_SUMMARY_COLUMNS)


def run_snapshot_sabr(
    snapshot_id: str,
    otm_surface: pd.DataFrame,
    recommended: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Run base and recommended SABR calibration for one snapshot."""

    base_result = calibrate_sabr_per_expiry(otm_surface)
    recommended_result = calibrate_sabr_per_expiry(recommended)
    base_rows = _sabr_params_to_rows(
        snapshot_id,
        "base_otm_surface",
        base_result.params,
    )
    recommended_rows = _sabr_params_to_rows(
        snapshot_id,
        "recommended_calibration_universe",
        recommended_result.params,
    )
    return (
        pd.concat([base_rows, recommended_rows], ignore_index=True),
        _summarize_sabr_params(base_result.params),
        _summarize_sabr_params(recommended_result.params),
    )


def run_snapshot_heston(
    snapshot_id: str,
    recommended: pd.DataFrame,
    max_nfev: int = 30,
) -> pd.DataFrame:
    """Run global and per-expiry Heston calibration for one snapshot."""

    if len(recommended) < 6:
        return _skipped_heston_result(snapshot_id, "too few recommended rows")
    frames = []
    try:
        global_result = calibrate_heston_global(recommended, max_nfev=max_nfev)
    except (RuntimeError, ValueError, OverflowError, FloatingPointError) as error:
        frames.append(_skipped_heston_result(snapshot_id, f"global Heston: {error}"))
    else:
        frames.append(_global_heston_rows(snapshot_id, global_result))

    try:
        per_expiry_result = calibrate_per_expiry_heston(
            recommended,
            max_nfev=min(max_nfev, 25),
        )
    except (RuntimeError, ValueError, OverflowError, FloatingPointError) as error:
        frames.append(
            _per_expiry_skipped_rows(
                snapshot_id,
                recommended,
                f"per-expiry Heston: {error}",
            )
        )
    else:
        frames.append(
            _per_expiry_heston_rows(snapshot_id, recommended, per_expiry_result)
        )

    return pd.concat(frames, ignore_index=True).loc[:, HESTON_RESULT_COLUMNS]


def _global_heston_rows(snapshot_id: str, result) -> pd.DataFrame:
    params = result.params.iloc[0]
    rows = []
    for error in result.fit_errors.itertuples(index=False):
        warnings = _heston_parameter_warnings(params)
        scope = "global" if error.scope == "global" else "global_by_expiry"
        rows.append(
            {
                "snapshot_id": snapshot_id,
                "scope": scope,
                "expiry": "" if pd.isna(error.expiry) else error.expiry,
                "n_points": int(error.total_points),
                "v0": params["v0"],
                "theta": params["theta"],
                "kappa": params["kappa"],
                "sigma": params["sigma"],
                "rho": params["rho"],
                "rmse_iv": error.rmse_iv,
                "mae_iv": error.mae_iv,
                "weighted_rmse_iv": error.weighted_rmse_iv,
                "max_abs_error": error.max_abs_error,
                "optimizer_success": bool(params["optimizer_success"]),
                "optimizer_message": params["optimizer_message"],
                "feller_condition": bool(params["feller_condition"]),
                "parameter_warnings": warnings,
                "status": params["status"],
                "failure_reason": ""
                if pd.isna(params["failure_reason"])
                else params["failure_reason"],
            }
        )
    return pd.DataFrame(rows, columns=HESTON_RESULT_COLUMNS)


def _per_expiry_heston_rows(
    snapshot_id: str,
    recommended: pd.DataFrame,
    result,
) -> pd.DataFrame:
    rows = []
    params_by_expiry = (
        result.params.assign(
            _expiry_key=lambda df: df["expiry"].map(_expiry_key),
        )
        if not result.params.empty
        else pd.DataFrame()
    )
    for error in result.fit_errors.itertuples(index=False):
        expiry_key = _expiry_key(error.expiry)
        params_match = params_by_expiry[
            params_by_expiry["_expiry_key"] == expiry_key
        ]
        if params_match.empty:
            continue
        params = params_match.iloc[0]
        status = (
            "calibrated" if bool(params["optimizer_success"]) else "optimizer_failed"
        )
        rows.append(
            {
                "snapshot_id": snapshot_id,
                "scope": "per_expiry",
                "expiry": error.expiry,
                "n_points": int(error.total_points),
                "v0": params["v0"],
                "theta": params["theta"],
                "kappa": params["kappa"],
                "sigma": params["sigma"],
                "rho": params["rho"],
                "rmse_iv": error.rmse_iv,
                "mae_iv": error.mae_iv,
                "weighted_rmse_iv": error.weighted_rmse_iv,
                "max_abs_error": error.max_abs_error,
                "optimizer_success": bool(params["optimizer_success"]),
                "optimizer_message": params["optimizer_message"],
                "feller_condition": bool(params["feller_satisfied"]),
                "parameter_warnings": params["parameter_warnings"],
                "status": status,
                "failure_reason": ""
                if bool(params["optimizer_success"])
                else params["optimizer_message"],
            }
        )

    reported_expiries = {_expiry_key(row["expiry"]) for row in rows}
    for expiry, group in recommended.groupby("expiry", sort=True):
        if _expiry_key(expiry) not in reported_expiries:
            rows.append(
                _skipped_heston_row(
                    snapshot_id,
                    "per_expiry",
                    expiry,
                    len(group),
                    _per_expiry_failure_reason(result.failure_report, expiry),
                )
            )
    return pd.DataFrame(rows, columns=HESTON_RESULT_COLUMNS)


def build_model_comparison(
    snapshot_id: str,
    sabr_results: pd.DataFrame,
    heston_results: pd.DataFrame,
) -> pd.DataFrame:
    """Compare SABR and Heston on the same recommended universe."""

    sabr = sabr_results[
        (sabr_results["universe"] == "recommended_calibration_universe")
        & (sabr_results["status"] == "calibrated")
    ]
    heston = heston_results[heston_results["status"] != "skipped"]
    rows = []
    for sabr_row in sabr.itertuples(index=False):
        heston_match = _best_heston_match(heston, sabr_row.expiry)
        if heston_match.empty:
            rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "expiry": sabr_row.expiry,
                    "n_points": sabr_row.n_points,
                    "sabr_rmse_iv": sabr_row.rmse_iv,
                    "heston_scope": "not_available",
                    "heston_rmse_iv": np.nan,
                    "sabr_weighted_rmse_iv": sabr_row.weighted_rmse_iv,
                    "heston_weighted_rmse_iv": np.nan,
                    "winner_by_rmse": "not_available",
                    "interpretation": "Heston result unavailable for expiry",
                }
            )
            continue
        heston_row = heston_match.iloc[0]
        winner = "sabr" if sabr_row.rmse_iv < heston_row["rmse_iv"] else "heston"
        rows.append(
            {
                "snapshot_id": snapshot_id,
                "expiry": sabr_row.expiry,
                "n_points": sabr_row.n_points,
                "sabr_rmse_iv": sabr_row.rmse_iv,
                "heston_scope": heston_row["scope"],
                "heston_rmse_iv": heston_row["rmse_iv"],
                "sabr_weighted_rmse_iv": sabr_row.weighted_rmse_iv,
                "heston_weighted_rmse_iv": heston_row["weighted_rmse_iv"],
                "winner_by_rmse": winner,
                "interpretation": _comparison_interpretation(winner),
            }
        )
    return pd.DataFrame(rows, columns=MODEL_COMPARISON_COLUMNS)


def build_filter_sensitivity(
    snapshot_id: str,
    otm_surface: pd.DataFrame,
    recommended: pd.DataFrame,
    base_sabr: pd.Series,
    recommended_sabr: pd.Series,
) -> pd.DataFrame:
    """Summarize how deterministic filtering changed SABR fit."""

    before = base_sabr["median_rmse_iv"]
    after = recommended_sabr["median_rmse_iv"]
    warning_before = int(base_sabr["warning_count"])
    warning_after = int(recommended_sabr["warning_count"])
    return pd.DataFrame(
        [
            {
                "snapshot_id": snapshot_id,
                "base_rows": int(len(otm_surface)),
                "recommended_rows": int(len(recommended)),
                "rows_removed": int(len(otm_surface) - len(recommended)),
                "base_median_rmse_iv": before,
                "recommended_median_rmse_iv": after,
                "rmse_change": after - before,
                "base_warning_count": warning_before,
                "recommended_warning_count": warning_after,
                "warning_count_change": warning_after - warning_before,
                "interpretation": _filter_interpretation(before, after),
            }
        ],
        columns=FILTER_SENSITIVITY_COLUMNS,
    )


def build_aggregate_summary(
    snapshot_summary: pd.DataFrame,
    sabr_results: pd.DataFrame,
    heston_results: pd.DataFrame,
    model_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Build aggregate cross-date summary metrics."""

    recommended_sabr = sabr_results[
        (sabr_results["universe"] == "recommended_calibration_universe")
        & (sabr_results["status"] == "calibrated")
    ]
    global_heston = heston_results[heston_results["scope"] == "global"]
    sabr_wins = int((model_comparison["winner_by_rmse"] == "sabr").sum())
    heston_wins = int((model_comparison["winner_by_rmse"] == "heston").sum())
    n_snapshots = int(len(snapshot_summary))
    rows = [
        (
            "snapshot_count",
            n_snapshots,
            "Number of saved snapshots processed.",
        ),
        (
            "total_raw_rows",
            int(snapshot_summary["raw_row_count"].sum()),
            "Total raw option rows across saved snapshots.",
        ),
        (
            "total_recommended_rows",
            int(snapshot_summary["filtered_calibration_row_count"].sum()),
            "Total rows in recommended calibration universes.",
        ),
        (
            "median_recommended_sabr_rmse",
            _safe_series_median(recommended_sabr["rmse_iv"]),
            "Median per-expiry SABR RMSE after filtering.",
        ),
        (
            "median_global_heston_rmse",
            _safe_series_median(global_heston["rmse_iv"]),
            "Median global Heston RMSE across snapshots.",
        ),
        (
            "sabr_expiry_wins",
            sabr_wins,
            "Expiries where SABR RMSE was lower than Heston.",
        ),
        (
            "heston_expiry_wins",
            heston_wins,
            "Expiries where Heston RMSE was lower than SABR.",
        ),
        (
            "generalization_status",
            "insufficient_snapshots" if n_snapshots < 2 else "multi_snapshot_run",
            "At least two saved snapshots are needed for cross-date claims.",
        ),
    ]
    return pd.DataFrame(rows, columns=AGGREGATE_SUMMARY_COLUMNS)


def build_failure_modes_markdown(
    snapshot_summary: pd.DataFrame,
    sabr_results: pd.DataFrame,
    heston_results: pd.DataFrame,
    model_comparison: pd.DataFrame,
) -> str:
    """Build markdown failure-mode notes from cross-date outputs."""

    sabr_failures = sabr_results[sabr_results["status"] != "calibrated"]
    heston_failures = heston_results[
        (heston_results["status"] != "calibrated")
        | (heston_results["optimizer_success"] != True)  # noqa: E712
        | (heston_results["feller_condition"] != True)  # noqa: E712
    ]
    heston_underfit = model_comparison[model_comparison["winner_by_rmse"] == "sabr"]
    lines = [
        "# Cross-Date Failure Modes",
        "",
        "## Scope",
        "",
        (
            f"Processed {len(snapshot_summary)} saved snapshot(s). This report is "
            "diagnostic and not a trading signal."
        ),
        "",
        "## SABR Calibration Failures Or Warnings",
        "",
        _markdown_count_line("Non-calibrated SABR rows", len(sabr_failures)),
        "",
        "## Heston Calibration Failures Or Warnings",
        "",
        _markdown_count_line("Heston warning/failure rows", len(heston_failures)),
        "",
        "## Same-Universe Underfit Cases",
        "",
        _markdown_count_line(
            "Expiries where Heston underfit SABR",
            len(heston_underfit),
        ),
        "",
        "## Interpretation",
        "",
        (
            "With the currently saved data, the cross-date layer validates the "
            "pipeline mechanics but cannot establish cross-date generalization. "
            "Add more saved snapshots to `data/raw_snapshots/snapshot_manifest.csv` "
            "to test additional dates and regimes."
        ),
        "",
    ]
    return "\n".join(lines)


def build_cross_date_research_note(
    snapshot_summary: pd.DataFrame,
    aggregate_summary: pd.DataFrame,
    sabr_results: pd.DataFrame,
    heston_results: pd.DataFrame,
    model_comparison: pd.DataFrame,
    filter_sensitivity: pd.DataFrame,
) -> str:
    """Build markdown research note for the cross-date validation extension."""

    n_snapshots = int(len(snapshot_summary))
    total_rows = int(snapshot_summary["raw_row_count"].sum())
    total_recommended = int(snapshot_summary["filtered_calibration_row_count"].sum())
    recommended_sabr = sabr_results[
        (sabr_results["universe"] == "recommended_calibration_universe")
        & (sabr_results["status"] == "calibrated")
    ]
    global_heston = heston_results[heston_results["scope"] == "global"]
    per_expiry_heston = heston_results[heston_results["scope"] == "per_expiry"]
    sabr_median = _safe_series_median(recommended_sabr["rmse_iv"])
    heston_median = _safe_series_median(global_heston["rmse_iv"])
    per_expiry_heston_median = _safe_series_median(per_expiry_heston["rmse_iv"])
    sabr_wins = int((model_comparison["winner_by_rmse"] == "sabr").sum())
    heston_wins = int((model_comparison["winner_by_rmse"] == "heston").sum())
    generalization = (
        "The current repository has only one saved SPY snapshot, so this run "
        "does not prove cross-date generalization."
        if n_snapshots < 2
        else "This run processes multiple saved snapshots and can be used to "
        "evaluate cross-date robustness."
    )
    lines = [
        "# Cross-Date Vol Surface Validation",
        "",
        "## Motivation",
        "",
        (
            "The original project was intentionally honest about its main "
            "limitation: most real-data conclusions came from one saved SPY "
            "option-chain snapshot. This extension adds a deterministic "
            "multi-snapshot validation layer."
        ),
        "",
        "## Data Snapshots Used",
        "",
        f"- Saved snapshots processed: {n_snapshots}",
        f"- Total raw option rows processed: {total_rows}",
        f"- Total recommended calibration rows: {total_recommended}",
        "",
        "## Methodology",
        "",
        (
            "For each saved snapshot, the pipeline applies the existing quote "
            "cleaning, forward extraction, OTM IV surface construction, "
            "recommended expiry/liquidity filtering, SABR calibration, and "
            "global and per-expiry Heston calibration where feasible."
        ),
        "",
        "## Filtering Rules",
        "",
        (
            "The recommended calibration universe keeps expiries with at least "
            "two trading days to expiry, relative spread <= 0.25, positive "
            "volume, and positive open interest."
        ),
        "",
        "## SABR Results",
        "",
        f"- Median recommended-universe SABR RMSE: {_format_float(sabr_median)}",
        "",
        "## Heston Results",
        "",
        f"- Median global Heston RMSE: {_format_float(heston_median)}",
        (
            "- Median per-expiry Heston RMSE: "
            f"{_format_float(per_expiry_heston_median)}"
        ),
        "",
        "## Cross-Date Comparison",
        "",
        f"- SABR expiry wins: {sabr_wins}",
        f"- Heston expiry wins: {heston_wins}",
        f"- Generalization status: {generalization}",
        "",
        "## Failure Modes",
        "",
        (
            "The failure-mode report records non-calibrated SABR rows, Heston "
            "optimizer/Feller/parameter warnings, and same-universe Heston "
            "underfit cases."
        ),
        "",
        "## Limitations",
        "",
        "- This is not a trading strategy.",
        "- This does not prove a universal model ranking.",
        "- Results depend on snapshot quality, maturity coverage, and filtering rules.",
        "- Additional saved snapshots are required for real cross-date claims.",
        "",
        "## What This Does And Does Not Prove",
        "",
        (
            "This extension proves the pipeline can run deterministically over a "
            "manifest of saved snapshots. With the current one-snapshot manifest, "
            "it does not prove that the SABR/Heston conclusion generalizes."
        ),
        "",
        "## Resume Bullet Candidates",
        "",
        _resume_bullets(n_snapshots, total_rows, sabr_median, heston_median),
        "",
    ]
    return "\n".join(lines)


def _build_surface(
    raw: pd.DataFrame,
    risk_free_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with_mid = compute_mid_price(raw)
    with_time = compute_time_to_expiry(with_mid)
    clean, filter_report = filter_option_quotes(with_time)
    forward_by_expiry = pair_calls_and_puts_for_forward(clean, risk_free_rate)
    clean_with_moneyness = add_forward_moneyness(clean, forward_by_expiry)
    iv_table = compute_implied_vol_table(clean_with_moneyness, risk_free_rate)
    otm_surface = build_otm_surface_table(iv_table)
    return clean_with_moneyness, filter_report, otm_surface


def _enrich_surface(surface: pd.DataFrame, clean_chain: pd.DataFrame) -> pd.DataFrame:
    keys = ["underlying", "quote_datetime", "expiry", "option_type", "strike"]
    optional = [
        "bid_ask_spread",
        "relative_spread",
        "volume",
        "open_interest",
        "last_trade_datetime",
    ]
    available = [column for column in optional if column in clean_chain]
    enrich = clean_chain.loc[:, [*keys, *available]].drop_duplicates(keys)
    merged = surface.merge(enrich, on=keys, how="left", validate="one_to_one")
    for column in ("relative_spread", "volume", "open_interest"):
        if column not in merged:
            raise ValueError(f"missing required enrichment column: {column}")
    return merged


def _sabr_params_to_rows(
    snapshot_id: str,
    universe: str,
    params: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for row in params.itertuples(index=False):
        warning_count = int(
            bool(row.alpha_at_bound)
            + bool(row.rho_at_bound)
            + (abs(row.rho) > 0.95 if pd.notna(row.rho) else False)
            + bool(row.nu_at_bound)
            + (row.nu > 5.0 if pd.notna(row.nu) else False)
            + (row.max_abs_error > 0.25 if pd.notna(row.max_abs_error) else False)
            + (not bool(row.optimizer_success))
            + (row.status != "calibrated")
        )
        rows.append(
            {
                "snapshot_id": snapshot_id,
                "universe": universe,
                "expiry": row.expiry,
                "n_points": int(row.n_points),
                "alpha": row.alpha,
                "rho": row.rho,
                "nu": row.nu,
                "rmse_iv": row.rmse_iv,
                "mae_iv": row.mae_iv,
                "weighted_rmse_iv": row.weighted_rmse_iv,
                "max_abs_error": row.max_abs_error,
                "optimizer_success": bool(row.optimizer_success),
                "status": row.status,
                "failure_reason": ""
                if pd.isna(row.failure_reason)
                else row.failure_reason,
                "warning_count": warning_count,
            }
        )
    return pd.DataFrame(rows, columns=SABR_RESULT_COLUMNS)


def _summarize_sabr_params(params: pd.DataFrame) -> pd.Series:
    calibrated = params[params["status"] == "calibrated"]
    warning_count = int(
        sum(
            bool(row.alpha_at_bound)
            + bool(row.rho_at_bound)
            + (abs(row.rho) > 0.95 if pd.notna(row.rho) else False)
            + bool(row.nu_at_bound)
            + (row.nu > 5.0 if pd.notna(row.nu) else False)
            + (row.max_abs_error > 0.25 if pd.notna(row.max_abs_error) else False)
            + (not bool(row.optimizer_success))
            + (row.status != "calibrated")
            for row in params.itertuples(index=False)
        )
    )
    return pd.Series(
        {
            "median_rmse_iv": _safe_series_median(calibrated["rmse_iv"]),
            "warning_count": warning_count,
        }
    )


def _skipped_heston_result(snapshot_id: str, reason: str) -> pd.DataFrame:
    return pd.DataFrame(
        [_skipped_heston_row(snapshot_id, "global", "", 0, reason)],
        columns=HESTON_RESULT_COLUMNS,
    )


def _per_expiry_skipped_rows(
    snapshot_id: str,
    recommended: pd.DataFrame,
    reason: str,
) -> pd.DataFrame:
    rows = [
        _skipped_heston_row(snapshot_id, "per_expiry", expiry, len(group), reason)
        for expiry, group in recommended.groupby("expiry", sort=True)
    ]
    return pd.DataFrame(rows, columns=HESTON_RESULT_COLUMNS)


def _skipped_heston_row(
    snapshot_id: str,
    scope: str,
    expiry,
    n_points: int,
    reason: str,
) -> dict[str, object]:
    return {
        "snapshot_id": snapshot_id,
        "scope": scope,
        "expiry": expiry,
        "n_points": int(n_points),
        "v0": np.nan,
        "theta": np.nan,
        "kappa": np.nan,
        "sigma": np.nan,
        "rho": np.nan,
        "rmse_iv": np.nan,
        "mae_iv": np.nan,
        "weighted_rmse_iv": np.nan,
        "max_abs_error": np.nan,
        "optimizer_success": False,
        "optimizer_message": "",
        "feller_condition": False,
        "parameter_warnings": "not_available",
        "status": "skipped",
        "failure_reason": reason,
    }


def _best_heston_match(heston: pd.DataFrame, expiry) -> pd.DataFrame:
    expiry_key = _expiry_key(expiry)
    for scope in ("per_expiry", "global_by_expiry"):
        match = heston[
            (heston["scope"] == scope)
            & (heston["expiry"].map(_expiry_key) == expiry_key)
        ]
        if not match.empty:
            return match
    return pd.DataFrame(columns=HESTON_RESULT_COLUMNS)


def _per_expiry_failure_reason(failure_report: pd.DataFrame, expiry) -> str:
    if failure_report.empty:
        return "per-expiry Heston unavailable"
    expiry_key = _expiry_key(expiry)
    expiry_failures = failure_report[
        failure_report["expiry"].map(_expiry_key) == expiry_key
    ]
    if expiry_failures.empty:
        return "per-expiry Heston unavailable"
    return "; ".join(expiry_failures["detail"].astype(str).tolist())


def _expiry_key(value) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return parsed.date().isoformat()


def _heston_parameter_warnings(params: pd.Series) -> str:
    warnings = []
    for name in ("v0", "theta", "kappa", "sigma", "rho"):
        if bool(params[f"{name}_at_bound"]):
            warnings.append(f"{name}_at_bound")
    if not bool(params["feller_condition"]):
        warnings.append("feller_violation")
    theta = float(params["theta"])
    if theta > 0:
        ratio = float(params["v0"]) / theta
        if ratio > 4.0 or ratio < 0.25:
            warnings.append("v0_theta_mismatch")
    return ";".join(warnings) if warnings else "none"


def _comparison_interpretation(winner: str) -> str:
    if winner == "sabr":
        return "SABR lower RMSE on the same filtered expiry"
    if winner == "heston":
        return "Heston lower RMSE on the same filtered expiry"
    return "Comparison unavailable"


def _filter_interpretation(before: float, after: float) -> str:
    if not np.isfinite(before) or not np.isfinite(after):
        return "insufficient calibrated SABR rows"
    if after < before:
        return "deterministic filtering improved SABR median RMSE"
    if after > before:
        return "deterministic filtering worsened SABR median RMSE"
    return "deterministic filtering left SABR median RMSE unchanged"


def _markdown_count_line(label: str, count: int) -> str:
    return f"- {label}: {count}"


def _resume_bullets(
    n_snapshots: int,
    total_rows: int,
    sabr_median: float,
    heston_median: float,
) -> str:
    if n_snapshots < 2:
        return "\n".join(
            [
                (
                    "- Conservative QR bullet: Extended a reproducible SPY "
                    f"vol-surface engine to support multi-snapshot validation; "
                    f"current checked manifest processes {n_snapshots} saved "
                    f"snapshot and {total_rows} option rows, so cross-date "
                    "generalization remains future work."
                ),
                (
                    "- Stronger QR bullet if results generalize: After adding "
                    "more saved snapshots, report the number of snapshots, total "
                    "rows, SABR/Heston RMSE distributions, and whether the "
                    "same-universe conclusion persists."
                ),
                (
                    "- Honest negative-result bullet if mixed: Built a "
                    "cross-date SABR/Heston validation framework that preserved "
                    "failed calibrations and showed model conclusions were "
                    "snapshot-dependent rather than universal."
                ),
            ]
        )
    return "\n".join(
        [
            (
                "- Conservative QR bullet: Built a deterministic cross-date "
                f"SPY vol-surface validation pipeline over {n_snapshots} saved "
                f"snapshots and {total_rows} option rows, comparing SABR and "
                "Heston on identical filtered universes."
            ),
            (
                "- Stronger QR bullet if results generalize: Validated that "
                f"SABR median RMSE {_format_float(sabr_median)} remained below "
                f"Heston median RMSE {_format_float(heston_median)} across saved "
                "SPY snapshots, with failed calibrations preserved."
            ),
            (
                "- Honest negative-result bullet if mixed: Built a cross-date "
                "SABR/Heston validation framework and found the original "
                "single-snapshot conclusion was sensitive to date, maturity "
                "coverage, or filtering assumptions."
            ),
        ]
    )


def _safe_min(df: pd.DataFrame, column: str) -> float:
    return float(df[column].min()) if len(df) else np.nan


def _safe_max(df: pd.DataFrame, column: str) -> float:
    return float(df[column].max()) if len(df) else np.nan


def _safe_series_median(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.median()) if len(clean) else np.nan


def _format_float(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.6f}"
