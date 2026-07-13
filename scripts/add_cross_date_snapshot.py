"""Validate and add a saved option-chain snapshot to the cross-date manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from volsurface.cross_date import ingest_snapshot_to_manifest


def main() -> None:
    args = _parse_args()
    manifest = ingest_snapshot_to_manifest(
        manifest_path=args.manifest,
        snapshot_id=args.snapshot_id,
        input_raw_csv=args.raw_csv,
        input_metadata_json=args.metadata_json,
        market_regime=args.market_regime,
        notes=args.notes,
        dest_dir=args.dest_dir,
        overwrite=args.overwrite,
    )
    print(
        f"added {args.snapshot_id}; manifest now contains "
        f"{len(manifest)} snapshot(s)"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a normalized option-chain CSV and metadata JSON into "
            "data/raw_snapshots/ and append a validated manifest row."
        )
    )
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--raw-csv", required=True, type=Path)
    parser.add_argument("--metadata-json", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        default=Path("data/raw_snapshots/snapshot_manifest.csv"),
        type=Path,
    )
    parser.add_argument(
        "--dest-dir",
        default=Path("data/raw_snapshots"),
        type=Path,
    )
    parser.add_argument("--market-regime", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing copied snapshot directory before appending.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
