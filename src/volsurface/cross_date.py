"""Cross-date volatility-surface validation utilities."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from volsurface.calibration import calibrate_sabr_per_expiry
from volsurface.filters import FILTER_REASONS, compute_mid_price, filter_option_quotes
from volsurface.forwards import add_forward_moneyness, pair_calls_and_puts_for_forward
from volsurface.heston_calibration import calibrate_heston_global
from volsurface.heston_diagnostics import calibrate_per_expiry_heston
from volsurface.market_data import load_snapshot, normalize_market_snapshot
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

SNAPSHOT_QUALITY_COLUMNS = [
    "snapshot_id",
    "underlying",
    "snapshot_datetime",
    "raw_rows",
    "clean_rows",
    "otm_rows",
    "recommended_rows",
    "retained_expiries",
    "min_time_to_expiry",
    "max_time_to_expiry",
    "min_log_moneyness",
    "max_log_moneyness",
    "median_relative_spread",
    "failed_sabr_fits",
    "failed_heston_fits",
    "usable_for_model_comparison",
    "quality_notes",
]

MODEL_SUMMARY_BY_SNAPSHOT_COLUMNS = [
    "snapshot_id",
    "sabr_median_rmse",
    "heston_median_rmse",
    "sabr_expiry_wins",
    "heston_expiry_wins",
    "available_comparison_expiries",
    "winner_summary",
]

CROSS_SNAPSHOT_AGGREGATION_COLUMNS = [
    "metric",
    "value",
    "interpretation",
]

SNAPSHOT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


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
    snapshot_quality: pd.DataFrame


@dataclass(frozen=True)
class CrossDateOutputs:
    """Container for cross-date validation artifacts."""

    snapshot_summary: pd.DataFrame
    aggregate_summary: pd.DataFrame
    sabr_results: pd.DataFrame
    heston_results: pd.DataFrame
    model_comparison: pd.DataFrame
    filter_sensitivity: pd.DataFrame
    snapshot_quality: pd.DataFrame
    model_summary_by_snapshot: pd.DataFrame
    cross_snapshot_aggregation: pd.DataFrame
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


def validate_snapshot_metadata(metadata_path: str | Path) -> dict[str, object]:
    """Validate metadata JSON for a saved cross-date snapshot."""

    path = Path(metadata_path)
    if not path.exists():
        raise ValueError(f"metadata_path does not exist: {path}")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"metadata_path is not valid JSON: {path}") from error
    if not isinstance(metadata, dict):
        raise ValueError("snapshot metadata must be a JSON object")

    required = ["underlying", "source", "spot"]
    missing = [field for field in required if field not in metadata]
    if missing:
        raise ValueError(f"metadata missing required fields: {', '.join(missing)}")
    timestamp = _metadata_snapshot_datetime(metadata)
    if pd.isna(pd.to_datetime(timestamp, errors="coerce", utc=True)):
        raise ValueError(
            "metadata must include a parseable retrieval_datetime, "
            "quote_datetime, or snapshot_datetime"
        )
    spot = pd.to_numeric(pd.Series([metadata["spot"]]), errors="coerce").iloc[0]
    if pd.isna(spot) or float(spot) <= 0:
        raise ValueError("metadata spot must be a positive number")
    return metadata


def append_snapshot_to_manifest(
    manifest_path: str | Path,
    snapshot_id: str,
    raw_path: str | Path,
    metadata_path: str | Path,
    market_regime: str,
    notes: str,
    base_dir: str | Path = ".",
) -> pd.DataFrame:
    """Append one already-saved snapshot row to the manifest with validation."""

    _validate_snapshot_id(snapshot_id)
    base = Path(base_dir)
    manifest_path = Path(manifest_path)
    manifest = (
        pd.read_csv(manifest_path)
        if manifest_path.exists()
        else pd.DataFrame(columns=MANIFEST_COLUMNS)
    )
    missing = set(MANIFEST_COLUMNS).difference(manifest.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"missing snapshot manifest columns: {missing_text}")
    if snapshot_id in set(manifest["snapshot_id"].astype(str)):
        raise ValueError(f"snapshot_id already exists in manifest: {snapshot_id}")

    raw_path = Path(raw_path)
    metadata_path = Path(metadata_path)
    absolute_raw = raw_path if raw_path.is_absolute() else base / raw_path
    absolute_metadata = (
        metadata_path if metadata_path.is_absolute() else base / metadata_path
    )
    if not absolute_raw.exists():
        raise ValueError(f"raw_path does not exist: {absolute_raw}")
    metadata = validate_snapshot_metadata(absolute_metadata)
    load_snapshot(absolute_raw)

    row = {
        "snapshot_id": snapshot_id,
        "underlying": str(metadata["underlying"]),
        "snapshot_datetime": _metadata_snapshot_datetime(metadata),
        "spot": float(metadata["spot"]),
        "source": str(metadata["source"]),
        "raw_path": _manifest_path(raw_path, base),
        "metadata_path": _manifest_path(metadata_path, base),
        "market_regime": market_regime,
        "notes": notes,
    }
    updated = pd.concat([manifest.loc[:, MANIFEST_COLUMNS], pd.DataFrame([row])])
    updated.to_csv(manifest_path, index=False)
    return load_snapshot_manifest(manifest_path, base_dir=base)


def ingest_snapshot_to_manifest(
    manifest_path: str | Path,
    snapshot_id: str,
    input_raw_csv: str | Path,
    input_metadata_json: str | Path,
    market_regime: str,
    notes: str,
    dest_dir: str | Path = "data/raw_snapshots",
    base_dir: str | Path = ".",
    overwrite: bool = False,
) -> pd.DataFrame:
    """Copy a normalized snapshot CSV/metadata pair and append it to manifest."""

    _validate_snapshot_id(snapshot_id)
    base = Path(base_dir)
    source_raw = Path(input_raw_csv)
    source_metadata = Path(input_metadata_json)
    if not source_raw.exists():
        raise ValueError(f"input_raw_csv does not exist: {source_raw}")
    if not source_metadata.exists():
        raise ValueError(f"input_metadata_json does not exist: {source_metadata}")
    metadata = validate_snapshot_metadata(source_metadata)
    normalized = normalize_market_snapshot(pd.read_csv(source_raw))
    snapshot_dir = Path(dest_dir) / snapshot_id
    raw_output = snapshot_dir / "option_chain.csv"
    metadata_output = snapshot_dir / "metadata.json"
    absolute_snapshot_dir = base / snapshot_dir
    if absolute_snapshot_dir.exists() and not overwrite:
        raise ValueError(
            f"destination snapshot directory already exists: {absolute_snapshot_dir}"
        )
    absolute_snapshot_dir.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(base / raw_output, index=False)
    shutil.copyfile(source_metadata, base / metadata_output)
    return append_snapshot_to_manifest(
        manifest_path,
        snapshot_id=snapshot_id,
        raw_path=raw_output,
        metadata_path=metadata_output,
        market_regime=market_regime,
        notes=notes or str(metadata.get("notes", "")),
        base_dir=base,
    )


def generalization_label(snapshot_count: int) -> str:
    """Return conservative evidence label from saved snapshot count."""

    if snapshot_count < 5:
        return "insufficient_snapshots"
    if snapshot_count < 10:
        return "weak_evidence"
    if snapshot_count < 20:
        return "moderate_evidence"
    return "stronger_evidence"


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
    snapshot_summary = _concat_frames(
        [result.snapshot_summary for result in results],
        SNAPSHOT_SUMMARY_COLUMNS,
    )
    sabr_results = _concat_frames(
        [result.sabr_results for result in results],
        SABR_RESULT_COLUMNS,
    )
    heston_results = _concat_frames(
        [result.heston_results for result in results],
        HESTON_RESULT_COLUMNS,
    )
    model_comparison = _concat_frames(
        [result.model_comparison for result in results],
        MODEL_COMPARISON_COLUMNS,
    )
    filter_sensitivity = _concat_frames(
        [result.filter_sensitivity for result in results],
        FILTER_SENSITIVITY_COLUMNS,
    )
    snapshot_quality = _concat_frames(
        [result.snapshot_quality for result in results],
        SNAPSHOT_QUALITY_COLUMNS,
    )
    model_summary_by_snapshot = build_model_summary_by_snapshot(
        sabr_results,
        heston_results,
        model_comparison,
    )
    cross_snapshot_aggregation = build_cross_snapshot_aggregation(
        snapshot_summary,
        sabr_results,
        heston_results,
        model_comparison,
        filter_sensitivity,
    )
    aggregate_summary = build_aggregate_summary(
        snapshot_summary,
        sabr_results,
        heston_results,
        model_comparison,
        cross_snapshot_aggregation,
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
        snapshot_quality,
        model_summary_by_snapshot,
        cross_snapshot_aggregation,
    )
    return CrossDateOutputs(
        snapshot_summary=snapshot_summary,
        aggregate_summary=aggregate_summary,
        sabr_results=sabr_results,
        heston_results=heston_results,
        model_comparison=model_comparison,
        filter_sensitivity=filter_sensitivity,
        snapshot_quality=snapshot_quality,
        model_summary_by_snapshot=model_summary_by_snapshot,
        cross_snapshot_aggregation=cross_snapshot_aggregation,
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
    surface_failure_reason = ""
    try:
        clean_chain, filter_report, otm_surface = _build_surface(raw, risk_free_rate)
    except ValueError as error:
        surface_failure_reason = str(error)
        clean_chain, filter_report = _build_clean_chain(raw)
        otm_surface = _empty_otm_surface()
    recommended = select_recommended_universe(_enrich_surface(otm_surface, clean_chain))
    snapshot_summary = summarize_snapshot(
        metadata,
        raw,
        clean_chain,
        otm_surface,
        recommended,
        filter_report,
    )
    if surface_failure_reason:
        sabr_results = _failed_sabr_results(
            str(metadata["snapshot_id"]),
            surface_failure_reason,
        )
        base_sabr_summary = _failed_sabr_summary()
        recommended_sabr_summary = _failed_sabr_summary()
    else:
        sabr_results, base_sabr_summary, recommended_sabr_summary = run_snapshot_sabr(
            str(metadata["snapshot_id"]),
            otm_surface,
            recommended,
        )
    heston_results = (
        _skipped_heston_result(str(metadata["snapshot_id"]), surface_failure_reason)
        if surface_failure_reason
        else (
            run_snapshot_heston(
                str(metadata["snapshot_id"]),
                recommended,
                max_nfev=heston_max_nfev,
            )
            if run_heston
            else _skipped_heston_result(
                str(metadata["snapshot_id"]),
                "heston disabled",
            )
        )
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
    snapshot_quality = build_snapshot_quality_report(
        metadata,
        raw,
        clean_chain,
        otm_surface,
        recommended,
        sabr_results,
        heston_results,
        model_comparison,
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
        snapshot_quality=snapshot_quality,
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


def build_snapshot_quality_report(
    metadata: pd.Series,
    raw: pd.DataFrame,
    clean_chain: pd.DataFrame,
    otm_surface: pd.DataFrame,
    recommended: pd.DataFrame,
    sabr_results: pd.DataFrame,
    heston_results: pd.DataFrame,
    model_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Build one-row snapshot quality and usability diagnostics."""

    snapshot_id = str(metadata["snapshot_id"])
    recommended_sabr = sabr_results[
        sabr_results["universe"] == "recommended_calibration_universe"
    ]
    failed_sabr = int((recommended_sabr["status"] != "calibrated").sum())
    comparable = model_comparison[
        model_comparison["winner_by_rmse"].isin(["sabr", "heston"])
    ]
    failed_heston = int((heston_results["status"] != "calibrated").sum())
    usable = bool(len(comparable) > 0 and len(recommended) > 0)
    notes = _quality_notes(
        recommended_rows=len(recommended),
        comparable_expiries=len(comparable),
        failed_sabr=failed_sabr,
        failed_heston=failed_heston,
    )
    row = {
        "snapshot_id": snapshot_id,
        "underlying": metadata["underlying"],
        "snapshot_datetime": metadata["snapshot_datetime"],
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean_chain)),
        "otm_rows": int(len(otm_surface)),
        "recommended_rows": int(len(recommended)),
        "retained_expiries": int(recommended["expiry"].nunique()),
        "min_time_to_expiry": _safe_min(recommended, "time_to_expiry"),
        "max_time_to_expiry": _safe_max(recommended, "time_to_expiry"),
        "min_log_moneyness": _safe_min(recommended, "log_moneyness"),
        "max_log_moneyness": _safe_max(recommended, "log_moneyness"),
        "median_relative_spread": _safe_series_median(
            clean_chain.get("relative_spread", pd.Series(dtype=float))
        ),
        "failed_sabr_fits": failed_sabr,
        "failed_heston_fits": failed_heston,
        "usable_for_model_comparison": usable,
        "quality_notes": notes,
    }
    return pd.DataFrame([row], columns=SNAPSHOT_QUALITY_COLUMNS)


def build_model_summary_by_snapshot(
    sabr_results: pd.DataFrame,
    heston_results: pd.DataFrame,
    model_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize SABR and Heston fit quality by snapshot."""

    snapshot_ids = sorted(
        set(sabr_results["snapshot_id"].astype(str))
        | set(heston_results["snapshot_id"].astype(str))
        | set(model_comparison["snapshot_id"].astype(str))
    )
    rows = []
    for snapshot_id in snapshot_ids:
        sabr = sabr_results[
            (sabr_results["snapshot_id"].astype(str) == snapshot_id)
            & (sabr_results["universe"] == "recommended_calibration_universe")
            & (sabr_results["status"] == "calibrated")
        ]
        comparison = model_comparison[
            model_comparison["snapshot_id"].astype(str) == snapshot_id
        ]
        available = comparison[comparison["winner_by_rmse"].isin(["sabr", "heston"])]
        heston = heston_results[
            (heston_results["snapshot_id"].astype(str) == snapshot_id)
            & (heston_results["scope"] == "per_expiry")
            & (heston_results["status"] != "skipped")
        ]
        if heston.empty:
            heston = heston_results[
                (heston_results["snapshot_id"].astype(str) == snapshot_id)
                & (heston_results["scope"] == "global")
            ]
        sabr_wins = int((available["winner_by_rmse"] == "sabr").sum())
        heston_wins = int((available["winner_by_rmse"] == "heston").sum())
        rows.append(
            {
                "snapshot_id": snapshot_id,
                "sabr_median_rmse": _safe_series_median(sabr["rmse_iv"]),
                "heston_median_rmse": _safe_series_median(heston["rmse_iv"]),
                "sabr_expiry_wins": sabr_wins,
                "heston_expiry_wins": heston_wins,
                "available_comparison_expiries": int(len(available)),
                "winner_summary": _winner_summary(sabr_wins, heston_wins),
            }
        )
    return pd.DataFrame(rows, columns=MODEL_SUMMARY_BY_SNAPSHOT_COLUMNS)


def build_cross_snapshot_aggregation(
    snapshot_summary: pd.DataFrame,
    sabr_results: pd.DataFrame,
    heston_results: pd.DataFrame,
    model_comparison: pd.DataFrame,
    filter_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    """Build cross-snapshot robustness metrics."""

    n_snapshots = int(len(snapshot_summary))
    recommended_sabr = sabr_results[
        sabr_results["universe"] == "recommended_calibration_universe"
    ]
    comparable = model_comparison[
        model_comparison["winner_by_rmse"].isin(["sabr", "heston"])
    ]
    sabr_wins = int((comparable["winner_by_rmse"] == "sabr").sum())
    heston_wins = int((comparable["winner_by_rmse"] == "heston").sum())
    conclusion = conclusion_stability(n_snapshots, sabr_wins, heston_wins)
    rows = [
        (
            "snapshot_count",
            n_snapshots,
            "Number of saved snapshots processed.",
        ),
        (
            "generalization_status",
            generalization_label(n_snapshots),
            "Evidence label based only on saved snapshot count.",
        ),
        (
            "conclusion_stability",
            conclusion,
            "Observed same-universe winner stability; not a universal claim.",
        ),
        (
            "sabr_expiry_wins",
            sabr_wins,
            "Comparable expiries where SABR RMSE was lower.",
        ),
        (
            "heston_expiry_wins",
            heston_wins,
            "Comparable expiries where Heston RMSE was lower.",
        ),
        (
            "sabr_failed_calibration_rate",
            _failure_rate(recommended_sabr),
            "Share of recommended-universe SABR rows that did not calibrate.",
        ),
        (
            "heston_failed_calibration_rate",
            _failure_rate(heston_results),
            "Share of Heston scope rows that did not calibrate.",
        ),
        (
            "mean_filter_retention_rate",
            _safe_series_mean(
                filter_sensitivity["recommended_rows"]
                / filter_sensitivity["base_rows"].replace(0, np.nan)
            ),
            "Average recommended/base OTM row retention.",
        ),
        (
            "median_sabr_snapshot_rmse",
            _safe_group_median(recommended_sabr, "snapshot_id", "rmse_iv"),
            "Median of per-snapshot SABR median RMSE values.",
        ),
        (
            "median_heston_snapshot_rmse",
            _safe_group_median(
                heston_results[heston_results["scope"] == "per_expiry"],
                "snapshot_id",
                "rmse_iv",
            ),
            "Median of per-snapshot per-expiry Heston median RMSE values.",
        ),
        *parameter_stability_rows(sabr_results, heston_results),
    ]
    return pd.DataFrame(rows, columns=CROSS_SNAPSHOT_AGGREGATION_COLUMNS)


def build_aggregate_summary(
    snapshot_summary: pd.DataFrame,
    sabr_results: pd.DataFrame,
    heston_results: pd.DataFrame,
    model_comparison: pd.DataFrame,
    cross_snapshot_aggregation: pd.DataFrame,
) -> pd.DataFrame:
    """Build aggregate cross-date summary metrics."""

    recommended_sabr = sabr_results[
        (sabr_results["universe"] == "recommended_calibration_universe")
        & (sabr_results["status"] == "calibrated")
    ]
    global_heston = heston_results[heston_results["scope"] == "global"]
    comparable = model_comparison[
        model_comparison["winner_by_rmse"].isin(["sabr", "heston"])
    ]
    sabr_wins = int((comparable["winner_by_rmse"] == "sabr").sum())
    heston_wins = int((comparable["winner_by_rmse"] == "heston").sum())
    n_snapshots = int(len(snapshot_summary))
    generalization_status = _aggregation_metric(
        cross_snapshot_aggregation,
        "generalization_status",
    )
    conclusion = _aggregation_metric(cross_snapshot_aggregation, "conclusion_stability")
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
            generalization_status,
            "Conservative evidence label; it does not imply universal truth.",
        ),
        (
            "conclusion_stability",
            conclusion,
            "Stable, mixed, or insufficient under current saved snapshots.",
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
        _failure_mode_interpretation(len(snapshot_summary)),
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
    snapshot_quality: pd.DataFrame,
    model_summary_by_snapshot: pd.DataFrame,
    cross_snapshot_aggregation: pd.DataFrame,
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
    generalization_status = _aggregation_metric(
        cross_snapshot_aggregation,
        "generalization_status",
    )
    conclusion = _aggregation_metric(
        cross_snapshot_aggregation,
        "conclusion_stability",
    )
    usable_snapshots = int(snapshot_quality["usable_for_model_comparison"].sum())
    generalization = _generalization_sentence(
        n_snapshots,
        generalization_status,
        conclusion,
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
        f"- Snapshots usable for model comparison: {usable_snapshots}",
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
        f"- Generalization status: {generalization_status}",
        f"- Conclusion stability: {conclusion}",
        f"- Interpretation: {generalization}",
        "",
        "## Per-Snapshot Model Summary",
        "",
        _markdown_table(model_summary_by_snapshot),
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
            "manifest of saved snapshots. The evidence label is based on saved "
            "snapshot count and should not be read as a universal model ranking."
        ),
        "",
        "## Resume Bullet Candidates",
        "",
        _resume_bullets(
            n_snapshots,
            total_rows,
            sabr_median,
            heston_median,
            conclusion,
            generalization_status,
        ),
        "",
    ]
    return "\n".join(lines)


def _build_surface(
    raw: pd.DataFrame,
    risk_free_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean, filter_report = _build_clean_chain(raw)
    forward_by_expiry = pair_calls_and_puts_for_forward(clean, risk_free_rate)
    clean_with_moneyness = add_forward_moneyness(clean, forward_by_expiry)
    iv_table = compute_implied_vol_table(clean_with_moneyness, risk_free_rate)
    otm_surface = build_otm_surface_table(iv_table)
    return clean_with_moneyness, filter_report, otm_surface


def _build_clean_chain(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    with_mid = compute_mid_price(raw)
    with_time = compute_time_to_expiry(with_mid)
    return filter_option_quotes(with_time)


def _empty_otm_surface() -> pd.DataFrame:
    columns = [
        "underlying",
        "quote_datetime",
        "expiry",
        "option_type",
        "strike",
        "mid",
        "bid",
        "ask",
        "spot",
        "forward",
        "time_to_expiry",
        "log_moneyness",
        "option_side_for_surface",
        "implied_vol",
        "iv_status",
        "iv_failure_reason",
        "relative_spread",
        "volume",
        "open_interest",
    ]
    return pd.DataFrame(columns=columns)


def _enrich_surface(surface: pd.DataFrame, clean_chain: pd.DataFrame) -> pd.DataFrame:
    if surface.empty:
        return surface.copy()
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


def _failed_sabr_results(snapshot_id: str, reason: str) -> pd.DataFrame:
    rows = [
        _failed_sabr_row(snapshot_id, "base_otm_surface", reason),
        _failed_sabr_row(snapshot_id, "recommended_calibration_universe", reason),
    ]
    return pd.DataFrame(rows, columns=SABR_RESULT_COLUMNS)


def _failed_sabr_row(snapshot_id: str, universe: str, reason: str) -> dict[str, object]:
    return {
        "snapshot_id": snapshot_id,
        "universe": universe,
        "expiry": "",
        "n_points": 0,
        "alpha": np.nan,
        "rho": np.nan,
        "nu": np.nan,
        "rmse_iv": np.nan,
        "mae_iv": np.nan,
        "weighted_rmse_iv": np.nan,
        "max_abs_error": np.nan,
        "optimizer_success": False,
        "status": "skipped",
        "failure_reason": reason,
        "warning_count": 1,
    }


def _failed_sabr_summary() -> pd.Series:
    return pd.Series({"median_rmse_iv": np.nan, "warning_count": 1})


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


def conclusion_stability(
    snapshot_count: int,
    sabr_wins: int,
    heston_wins: int,
) -> str:
    """Classify observed model-comparison stability conservatively."""

    if snapshot_count < 5:
        return "insufficient"
    if sabr_wins == 0 and heston_wins == 0:
        return "insufficient"
    if sabr_wins > 0 and heston_wins == 0:
        return "stable_sabr_under_current_snapshots"
    if heston_wins > 0 and sabr_wins == 0:
        return "stable_heston_under_current_snapshots"
    return "mixed"


def parameter_stability_rows(
    sabr_results: pd.DataFrame,
    heston_results: pd.DataFrame,
) -> list[tuple[str, float, str]]:
    """Return simple parameter dispersion rows across saved snapshots."""

    rows: list[tuple[str, float, str]] = []
    recommended_sabr = sabr_results[
        (sabr_results["universe"] == "recommended_calibration_universe")
        & (sabr_results["status"] == "calibrated")
    ]
    for name in ("alpha", "rho", "nu"):
        rows.append(
            (
                f"sabr_{name}_std",
                _safe_series_std(recommended_sabr[name]),
                f"Standard deviation of calibrated SABR {name}.",
            )
        )

    heston = heston_results[
        (heston_results["scope"] == "per_expiry")
        & (heston_results["status"] != "skipped")
    ]
    if heston.empty:
        heston = heston_results[heston_results["scope"] == "global"]
    for name in ("v0", "theta", "kappa", "sigma", "rho"):
        rows.append(
            (
                f"heston_{name}_std",
                _safe_series_std(heston[name]),
                f"Standard deviation of available Heston {name}.",
            )
        )
    return rows


def _quality_notes(
    recommended_rows: int,
    comparable_expiries: int,
    failed_sabr: int,
    failed_heston: int,
) -> str:
    notes = []
    if recommended_rows == 0:
        notes.append("no recommended rows")
    if comparable_expiries == 0:
        notes.append("no comparable model expiries")
    if failed_sabr:
        notes.append(f"{failed_sabr} failed SABR fits")
    if failed_heston:
        notes.append(f"{failed_heston} failed Heston fits")
    return "; ".join(notes) if notes else "usable"


def _winner_summary(sabr_wins: int, heston_wins: int) -> str:
    if sabr_wins > 0 and heston_wins == 0:
        return "sabr_only"
    if heston_wins > 0 and sabr_wins == 0:
        return "heston_only"
    if sabr_wins > 0 and heston_wins > 0:
        return "mixed"
    return "not_available"


def _failure_rate(results: pd.DataFrame) -> float:
    if results.empty:
        return np.nan
    return float((results["status"] != "calibrated").mean())


def _safe_series_mean(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.mean()) if len(clean) else np.nan


def _safe_series_std(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.std(ddof=0)) if len(clean) else np.nan


def _safe_group_median(df: pd.DataFrame, group_column: str, value_column: str) -> float:
    if df.empty:
        return np.nan
    grouped = (
        df.assign(_value=pd.to_numeric(df[value_column], errors="coerce"))
        .dropna(subset=["_value"])
        .groupby(group_column)["_value"]
        .median()
    )
    return _safe_series_median(grouped)


def _concat_frames(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return pd.DataFrame(columns=columns)
    return pd.concat(nonempty, ignore_index=True).loc[:, columns]


def _aggregation_metric(aggregation: pd.DataFrame, metric: str) -> object:
    match = aggregation[aggregation["metric"] == metric]
    if match.empty:
        return ""
    return match["value"].iloc[0]


def _generalization_sentence(
    snapshot_count: int,
    generalization_status: object,
    conclusion: object,
) -> str:
    if snapshot_count < 5:
        return (
            f"The run has {snapshot_count} saved snapshot(s), so it remains below "
            "the minimum threshold for cross-date evidence."
        )
    return (
        f"The evidence label is {generalization_status}; the observed conclusion "
        f"stability is {conclusion}. This is still data-set-specific, not a "
        "universal model ranking."
    )


def _failure_mode_interpretation(snapshot_count: int) -> str:
    if snapshot_count < 5:
        return (
            "With the currently saved data, the cross-date layer validates the "
            "pipeline mechanics but remains below the threshold for cross-date "
            "evidence. Add more saved snapshots to "
            "`data/raw_snapshots/snapshot_manifest.csv` to test additional "
            "dates and regimes."
        )
    return (
        "This report summarizes model failures and underfit cases across saved "
        "snapshots. The evidence label remains conservative and data-specific."
    )


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No per-snapshot model summary rows were generated."
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(_format_float)
    header = "| " + " | ".join(formatted.columns) + " |"
    separator = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in formatted.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])


def _validate_snapshot_id(snapshot_id: str) -> None:
    if not snapshot_id or not SNAPSHOT_ID_PATTERN.match(snapshot_id):
        raise ValueError(
            "snapshot_id must contain only letters, numbers, '.', '_', or '-' "
            "and must start with a letter or number"
        )


def _metadata_snapshot_datetime(metadata: dict[str, object]) -> object:
    for key in ("snapshot_datetime", "retrieval_datetime", "quote_datetime"):
        value = metadata.get(key)
        if value:
            return value
    return pd.NA


def _manifest_path(path: Path, base: Path) -> str:
    candidate = path if path.is_absolute() else base / path
    try:
        return str(candidate.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def _resume_bullets(
    n_snapshots: int,
    total_rows: int,
    sabr_median: float,
    heston_median: float,
    conclusion: object,
    generalization_status: object,
) -> str:
    if generalization_status == "insufficient_snapshots":
        return "\n".join(
            [
                (
                    "- Conservative QR bullet: Extended a reproducible SPY "
                    f"vol-surface engine to support multi-snapshot validation; "
                    f"current checked manifest processes {n_snapshots} saved "
                    f"{'snapshot' if n_snapshots == 1 else 'snapshots'} and "
                    f"{total_rows} option rows, so cross-date "
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
                f"SPY snapshots, with conclusion status {conclusion} and failed "
                "calibrations preserved."
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
