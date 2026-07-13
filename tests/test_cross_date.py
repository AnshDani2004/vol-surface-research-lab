from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from volsurface.black_scholes import black_scholes_price
from volsurface.cross_date import (
    CROSS_SNAPSHOT_AGGREGATION_COLUMNS,
    FILTER_SENSITIVITY_COLUMNS,
    HESTON_RESULT_COLUMNS,
    MODEL_COMPARISON_COLUMNS,
    MODEL_SUMMARY_BY_SNAPSHOT_COLUMNS,
    SABR_RESULT_COLUMNS,
    SNAPSHOT_QUALITY_COLUMNS,
    SNAPSHOT_SUMMARY_COLUMNS,
    append_snapshot_to_manifest,
    generalization_label,
    ingest_snapshot_to_manifest,
    load_snapshot_manifest,
    process_snapshot,
    run_cross_date_validation,
)


def test_multi_snapshot_manifest_loading(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, ["snap_a", "snap_b"])

    manifest = load_snapshot_manifest(manifest_path, base_dir=tmp_path)

    assert manifest["snapshot_id"].tolist() == ["snap_a", "snap_b"]
    assert manifest["snapshot_datetime"].dt.tz is not None


def test_snapshot_metadata_validation_rejects_duplicate_ids(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, ["snap_a", "snap_a"])

    with pytest.raises(ValueError, match="snapshot_id"):
        load_snapshot_manifest(manifest_path, base_dir=tmp_path)


def test_manifest_append_validation_ingests_snapshot_copy(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, ["snap_a"])
    raw_path = tmp_path / "new_snapshot.csv"
    metadata_path = tmp_path / "new_snapshot.json"
    _write_snapshot(raw_path, spot=105.0, vol=0.24)
    _write_metadata(metadata_path, spot=105.0)

    manifest = ingest_snapshot_to_manifest(
        manifest_path=manifest_path,
        snapshot_id="snap_b",
        input_raw_csv=raw_path,
        input_metadata_json=metadata_path,
        market_regime="unit_test_fixture",
        notes="ingested deterministic test fixture",
        dest_dir=tmp_path / "raw_snapshots",
        base_dir=tmp_path,
    )

    assert manifest["snapshot_id"].tolist() == ["snap_a", "snap_b"]
    assert (tmp_path / "raw_snapshots" / "snap_b" / "option_chain.csv").exists()
    assert (tmp_path / "raw_snapshots" / "snap_b" / "metadata.json").exists()


def test_manifest_append_rejects_duplicate_snapshot_id(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, ["snap_a"])
    raw_path = tmp_path / "new_snapshot.csv"
    metadata_path = tmp_path / "new_snapshot.json"
    _write_snapshot(raw_path, spot=105.0, vol=0.24)
    _write_metadata(metadata_path, spot=105.0)

    with pytest.raises(ValueError, match="already exists"):
        append_snapshot_to_manifest(
            manifest_path=manifest_path,
            snapshot_id="snap_a",
            raw_path=raw_path,
            metadata_path=metadata_path,
            market_regime="unit_test_fixture",
            notes="duplicate",
            base_dir=tmp_path,
        )


def test_missing_file_detection_is_explicit(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, ["snap_a"])
    metadata_path = tmp_path / "new_snapshot.json"
    _write_metadata(metadata_path, spot=105.0)

    with pytest.raises(ValueError, match="raw_path does not exist"):
        append_snapshot_to_manifest(
            manifest_path=manifest_path,
            snapshot_id="snap_b",
            raw_path=tmp_path / "missing.csv",
            metadata_path=metadata_path,
            market_regime="unit_test_fixture",
            notes="missing raw file",
            base_dir=tmp_path,
        )


def test_malformed_metadata_validation_is_explicit(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, ["snap_a"])
    raw_path = tmp_path / "new_snapshot.csv"
    metadata_path = tmp_path / "bad_metadata.json"
    _write_snapshot(raw_path, spot=105.0, vol=0.24)
    metadata_path.write_text('{"source": "fixture"}\n')

    with pytest.raises(ValueError, match="metadata missing required fields"):
        append_snapshot_to_manifest(
            manifest_path=manifest_path,
            snapshot_id="snap_b",
            raw_path=raw_path,
            metadata_path=metadata_path,
            market_regime="unit_test_fixture",
            notes="bad metadata",
            base_dir=tmp_path,
        )


def test_deterministic_filtering_across_snapshot_runs(tmp_path: Path) -> None:
    manifest = load_snapshot_manifest(
        _write_manifest(tmp_path, ["snap_a"]),
        base_dir=tmp_path,
    )

    first = process_snapshot(manifest.iloc[0], base_dir=tmp_path, run_heston=False)
    second = process_snapshot(manifest.iloc[0], base_dir=tmp_path, run_heston=False)

    pd.testing.assert_frame_equal(first.snapshot_summary, second.snapshot_summary)
    pd.testing.assert_frame_equal(
        first.recommended_universe.reset_index(drop=True),
        second.recommended_universe.reset_index(drop=True),
    )


def test_cross_date_output_schemas_and_failed_heston_are_preserved(
    tmp_path: Path,
) -> None:
    manifest = load_snapshot_manifest(
        _write_manifest(tmp_path, ["snap_a", "snap_b"]),
        base_dir=tmp_path,
    )

    outputs = run_cross_date_validation(manifest, base_dir=tmp_path, run_heston=False)

    assert list(outputs.snapshot_summary.columns) == SNAPSHOT_SUMMARY_COLUMNS
    assert list(outputs.sabr_results.columns) == SABR_RESULT_COLUMNS
    assert list(outputs.heston_results.columns) == HESTON_RESULT_COLUMNS
    assert list(outputs.model_comparison.columns) == MODEL_COMPARISON_COLUMNS
    assert list(outputs.filter_sensitivity.columns) == FILTER_SENSITIVITY_COLUMNS
    assert list(outputs.snapshot_quality.columns) == SNAPSHOT_QUALITY_COLUMNS
    assert (
        list(outputs.model_summary_by_snapshot.columns)
        == MODEL_SUMMARY_BY_SNAPSHOT_COLUMNS
    )
    assert (
        list(outputs.cross_snapshot_aggregation.columns)
        == CROSS_SNAPSHOT_AGGREGATION_COLUMNS
    )
    assert "skipped" in set(outputs.heston_results["status"])
    assert not bool(outputs.snapshot_quality["usable_for_model_comparison"].any())


def test_generalization_status_thresholds() -> None:
    assert generalization_label(1) == "insufficient_snapshots"
    assert generalization_label(4) == "insufficient_snapshots"
    assert generalization_label(5) == "weak_evidence"
    assert generalization_label(9) == "weak_evidence"
    assert generalization_label(10) == "moderate_evidence"
    assert generalization_label(19) == "moderate_evidence"
    assert generalization_label(20) == "stronger_evidence"


def test_cross_date_report_generation_and_reproducibility() -> None:
    subprocess.run(
        [sys.executable, "scripts/generate_cross_date_outputs.py"],
        check=True,
    )
    first = Path("artifacts/cross_date/snapshot_summary.csv").read_text()
    subprocess.run(
        [sys.executable, "scripts/generate_cross_date_outputs.py"],
        check=True,
    )
    second = Path("artifacts/cross_date/snapshot_summary.csv").read_text()

    assert first == second
    for path in [
        "artifacts/cross_date/snapshot_summary.csv",
        "artifacts/cross_date/sabr_cross_date_results.csv",
        "artifacts/cross_date/heston_cross_date_results.csv",
        "artifacts/cross_date/model_comparison_summary.csv",
        "artifacts/cross_date/filter_sensitivity_summary.csv",
        "artifacts/cross_date/snapshot_quality_report.csv",
        "artifacts/cross_date/model_summary_by_snapshot.csv",
        "artifacts/cross_date/cross_snapshot_aggregation.csv",
        "artifacts/cross_date/failure_modes.md",
        "artifacts/cross_date/cross_date_research_note.md",
    ]:
        assert Path(path).exists()


def _write_manifest(tmp_path: Path, snapshot_ids: list[str]) -> Path:
    rows = []
    for index, snapshot_id in enumerate(snapshot_ids):
        raw_path = tmp_path / f"{snapshot_id}.csv"
        metadata_path = tmp_path / f"{snapshot_id}.json"
        _write_snapshot(raw_path, spot=100.0 + index, vol=0.20 + 0.02 * index)
        _write_metadata(metadata_path, spot=100.0 + index)
        rows.append(
            {
                "snapshot_id": snapshot_id,
                "underlying": "SPY",
                "snapshot_datetime": "2026-01-01T14:30:00+00:00",
                "spot": 100.0 + index,
                "source": "fixture",
                "raw_path": raw_path.relative_to(tmp_path),
                "metadata_path": metadata_path.relative_to(tmp_path),
                "market_regime": "unit_test_fixture",
                "notes": "deterministic test fixture",
            }
        )
    manifest_path = tmp_path / "snapshot_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def _write_metadata(path: Path, spot: float) -> None:
    path.write_text(
        "{"
        '"underlying": "SPY", '
        '"source": "fixture", '
        '"retrieval_datetime": "2026-01-01T14:30:00+00:00", '
        f'"spot": {spot}, '
        '"notes": "deterministic test fixture"'
        "}\n"
    )


def _write_snapshot(path: Path, spot: float, vol: float) -> None:
    quote_datetime = pd.Timestamp("2026-01-01T14:30:00+00:00")
    expiry = pd.Timestamp("2026-02-20T21:00:00+00:00")
    time_to_expiry = (expiry - quote_datetime).total_seconds() / (
        365.0 * 24.0 * 60.0 * 60.0
    )
    rows = []
    for strike in [90.0, 95.0, 100.0, 105.0, 110.0]:
        for option_type in ["call", "put"]:
            price = black_scholes_price(
                spot,
                strike,
                time_to_expiry,
                0.05,
                vol,
                option_type,
            )
            spread = max(float(price) * 0.02, 0.02)
            bid = max(float(price) - spread / 2.0, 0.01)
            ask = max(float(price) + spread / 2.0, bid + 0.01)
            rows.append(
                {
                    "underlying": "SPY",
                    "quote_datetime": quote_datetime.isoformat(),
                    "expiry": expiry.isoformat(),
                    "option_type": option_type,
                    "strike": strike,
                    "bid": bid,
                    "ask": ask,
                    "last": price,
                    "volume": 10,
                    "open_interest": 100,
                    "spot": spot,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
