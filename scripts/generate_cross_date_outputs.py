"""Generate deterministic cross-date volatility validation artifacts."""

from __future__ import annotations

from pathlib import Path

from volsurface.cross_date import (
    load_snapshot_manifest,
    process_snapshot,
    run_cross_date_validation,
)

MANIFEST_PATH = Path("data/raw_snapshots/snapshot_manifest.csv")
PROCESSED_DIR = Path("data/processed_snapshots")
OUTPUT_DIR = Path("artifacts/cross_date")


def main() -> None:
    """Run cross-date validation from saved snapshot manifest."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_snapshot_manifest(MANIFEST_PATH)
    outputs = run_cross_date_validation(manifest)
    _write_processed_snapshot_tables(manifest)

    outputs.snapshot_summary.to_csv(OUTPUT_DIR / "snapshot_summary.csv", index=False)
    outputs.aggregate_summary.to_csv(
        OUTPUT_DIR / "aggregate_summary.csv",
        index=False,
    )
    outputs.sabr_results.to_csv(
        OUTPUT_DIR / "sabr_cross_date_results.csv",
        index=False,
    )
    outputs.heston_results.to_csv(
        OUTPUT_DIR / "heston_cross_date_results.csv",
        index=False,
    )
    outputs.model_comparison.to_csv(
        OUTPUT_DIR / "model_comparison_summary.csv",
        index=False,
    )
    outputs.filter_sensitivity.to_csv(
        OUTPUT_DIR / "filter_sensitivity_summary.csv",
        index=False,
    )
    outputs.snapshot_quality.to_csv(
        OUTPUT_DIR / "snapshot_quality_report.csv",
        index=False,
    )
    outputs.model_summary_by_snapshot.to_csv(
        OUTPUT_DIR / "model_summary_by_snapshot.csv",
        index=False,
    )
    outputs.cross_snapshot_aggregation.to_csv(
        OUTPUT_DIR / "cross_snapshot_aggregation.csv",
        index=False,
    )
    outputs.filter_attrition.to_csv(
        OUTPUT_DIR / "filter_attrition_by_snapshot.csv",
        index=False,
    )
    outputs.failed_examples.to_csv(
        OUTPUT_DIR / "failed_snapshot_examples.csv",
        index=False,
    )
    outputs.forward_pairing_audit.to_csv(
        OUTPUT_DIR / "forward_pairing_audit.csv",
        index=False,
    )
    (OUTPUT_DIR / "failure_modes.md").write_text(
        outputs.failure_modes_markdown,
        encoding="utf-8",
    )
    (OUTPUT_DIR / "cross_date_research_note.md").write_text(
        outputs.research_note_markdown,
        encoding="utf-8",
    )


def _write_processed_snapshot_tables(manifest) -> None:
    for _, row in manifest.iterrows():
        result = process_snapshot(row, run_heston=False)
        snapshot_dir = PROCESSED_DIR / str(row["snapshot_id"])
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        _drop_audit_columns(result.clean_chain).to_csv(
            snapshot_dir / "clean_chain.csv",
            index=False,
        )
        result.filter_report.to_csv(snapshot_dir / "filter_report.csv", index=False)
        _drop_audit_columns(result.otm_surface).to_csv(
            snapshot_dir / "otm_surface_table.csv",
            index=False,
        )
        _drop_audit_columns(result.recommended_universe).to_csv(
            snapshot_dir / "recommended_calibration_universe.csv",
            index=False,
        )


def _drop_audit_columns(df):
    return df.drop(columns=["_audit_row_id"], errors="ignore")


if __name__ == "__main__":
    main()
