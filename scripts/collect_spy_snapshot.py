"""Collect a saved SPY yfinance option-chain snapshot for cross-date validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from volsurface.cross_date import (
    append_snapshot_to_manifest,
    build_pre_ingestion_quality_report,
    passes_pre_ingestion_quality_gate,
)
from volsurface.market_data import combine_calls_puts, save_snapshot


def main() -> None:
    args = _parse_args()
    retrieval_datetime = datetime.now(timezone.utc)
    snapshot_id = args.snapshot_id or _default_snapshot_id(
        args.underlying,
        retrieval_datetime,
    )
    snapshot, metadata = _download_yfinance_snapshot(
        args.underlying,
        retrieval_datetime,
        args.max_expiries,
    )
    metadata.update(
        {
            "snapshot_id": snapshot_id,
            "snapshot_datetime": retrieval_datetime.isoformat(),
            "market_regime": args.market_regime,
            "notes": args.notes,
            "vendor_certification": "not_vendor_certified",
            "source_note": (
                "Saved yfinance option-chain snapshot; yfinance is not a "
                "vendor-certified market-data source."
            ),
        }
    )
    quality_report = build_pre_ingestion_quality_report(snapshot, snapshot_id)
    quality = quality_report.iloc[0].to_dict()
    metadata["pre_ingestion_quality"] = quality
    print(quality_report.to_string(index=False))

    if args.dry_run:
        print(f"dry run only; snapshot {snapshot_id} was not saved")
        return

    snapshot_dir = args.dest_dir / snapshot_id
    raw_path = snapshot_dir / "option_chain.csv"
    metadata_path = snapshot_dir / "metadata.json"
    if snapshot_dir.exists() or raw_path.exists() or metadata_path.exists():
        raise ValueError(f"snapshot output already exists: {snapshot_dir}")

    manifest_raw_path = raw_path
    manifest_metadata_path = metadata_path
    metadata["raw_path"] = str(manifest_raw_path)
    metadata["metadata_path"] = str(manifest_metadata_path)
    save_snapshot(snapshot, raw_path, metadata_path, metadata)
    quality_report.to_csv(snapshot_dir / "pre_ingestion_quality.csv", index=False)

    if not args.allow_low_quality and not passes_pre_ingestion_quality_gate(
        quality_report
    ):
        classification = quality["quality_classification"]
        print(
            f"saved {snapshot_id} but did not append manifest; "
            f"classification={classification}. Re-run with --allow-low-quality "
            "to preserve it in cross-date outputs."
        )
        return
    if not args.append_manifest:
        print(f"saved {snapshot_id}; manifest append disabled")
        return

    manifest = append_snapshot_to_manifest(
        manifest_path=args.manifest,
        snapshot_id=snapshot_id,
        raw_path=manifest_raw_path,
        metadata_path=manifest_metadata_path,
        market_regime=args.market_regime,
        notes=args.notes,
    )
    print(
        f"saved and appended {snapshot_id}; manifest now contains "
        f"{len(manifest)} snapshot(s)"
    )


def _download_yfinance_snapshot(
    underlying: str,
    retrieval_datetime: datetime,
    max_expiries: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        import yfinance as yf
    except ImportError as error:
        raise RuntimeError(
            "yfinance is required for live snapshot collection. "
            "Install the market extra with `python -m pip install -e .[market]`."
        ) from error

    ticker = yf.Ticker(underlying)
    expiries = list(ticker.options)
    if not expiries:
        raise ValueError(f"no option expiries returned by yfinance for {underlying}")
    selected_expiries = expiries[:max_expiries]
    spot = _extract_spot(ticker)
    chains = []
    for expiry in selected_expiries:
        chain = ticker.option_chain(expiry)
        metadata = {
            "underlying": underlying,
            "retrieval_datetime": retrieval_datetime.isoformat(),
            "quote_datetime": retrieval_datetime.isoformat(),
            "expiry": expiry,
            "spot": spot,
        }
        chains.append(combine_calls_puts(chain.calls, chain.puts, metadata))
    snapshot = pd.concat(chains, ignore_index=True)
    metadata = {
        "underlying": underlying,
        "retrieval_datetime": retrieval_datetime.isoformat(),
        "quote_datetime": retrieval_datetime.isoformat(),
        "spot": spot,
        "source": "yfinance",
        "expiries": selected_expiries,
        "max_expiries_requested": max_expiries,
    }
    return snapshot, metadata


def _extract_spot(ticker: object) -> float:
    fast_info = getattr(ticker, "fast_info", {})
    candidates = [
        _mapping_get(fast_info, "last_price"),
        _mapping_get(fast_info, "lastPrice"),
        _mapping_get(fast_info, "regular_market_price"),
    ]
    info = getattr(ticker, "info", {})
    candidates.extend(
        [
            _mapping_get(info, "regularMarketPrice"),
            _mapping_get(info, "currentPrice"),
            _mapping_get(info, "previousClose"),
        ]
    )
    for value in candidates:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric) and float(numeric) > 0:
            return float(numeric)
    raise ValueError("could not determine a positive underlying spot from yfinance")


def _mapping_get(mapping: object, key: str) -> object:
    try:
        return mapping[key]  # type: ignore[index]
    except (KeyError, TypeError):
        return None


def _default_snapshot_id(underlying: str, retrieval_datetime: datetime) -> str:
    stamp = retrieval_datetime.strftime("%Y_%m_%d_%H%M")
    return f"{underlying.lower()}_{stamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a saved yfinance option-chain snapshot and optionally append "
            "it to the cross-date manifest after deterministic quality checks."
        )
    )
    parser.add_argument("--underlying", default="SPY")
    parser.add_argument("--snapshot-id")
    parser.add_argument(
        "--market-regime",
        required=True,
        help="Short regime label, for example active_market_hours or after_hours.",
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-low-quality",
        action="store_true",
        help="Append snapshots that fail the pre-ingestion quality gate.",
    )
    parser.add_argument(
        "--append-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append passing snapshots to the manifest after saving.",
    )
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
    parser.add_argument("--max-expiries", default=6, type=int)
    args = parser.parse_args()
    if args.max_expiries <= 0:
        raise ValueError("--max-expiries must be positive")
    if not args.append_manifest and args.allow_low_quality:
        raise ValueError("--allow-low-quality requires manifest append to be enabled")
    return args


if __name__ == "__main__":
    main()
