"""Market option-chain snapshot normalization and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from volsurface.data import REQUIRED_COLUMNS, load_option_chain

MARKET_COLUMN_ALIASES = {
    "lastprice": "last",
    "last_price": "last",
    "openinterest": "open_interest",
    "open_interest": "open_interest",
    "expiration": "expiry",
    "expiration_date": "expiry",
    "lasttradedate": "last_trade_datetime",
}


def normalize_market_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a vendor-shaped market snapshot to the Stage 2 schema."""

    snapshot = df.copy()
    snapshot = snapshot.rename(
        columns={column: _normalize_market_column(column) for column in snapshot}
    )
    for column in REQUIRED_COLUMNS:
        if column not in snapshot:
            snapshot[column] = _default_value_for_missing_column(column)

    snapshot["quote_datetime"] = pd.to_datetime(
        snapshot["quote_datetime"],
        errors="coerce",
    )
    snapshot["expiry"] = pd.to_datetime(snapshot["expiry"], errors="coerce")
    snapshot["option_type"] = (
        snapshot["option_type"].astype("string").str.strip().str.lower()
    )
    snapshot["option_type"] = snapshot["option_type"].replace(
        {"c": "call", "calls": "call", "p": "put", "puts": "put"}
    )

    for column in ["strike", "bid", "ask", "last", "volume", "open_interest", "spot"]:
        snapshot[column] = pd.to_numeric(snapshot[column], errors="coerce")

    snapshot["volume"] = snapshot["volume"].fillna(0)
    snapshot["open_interest"] = snapshot["open_interest"].fillna(0)
    extras = sorted(
        column for column in snapshot.columns if column not in REQUIRED_COLUMNS
    )
    return snapshot.loc[:, [*REQUIRED_COLUMNS, *extras]]


def combine_calls_puts(
    calls_df: pd.DataFrame,
    puts_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    """Combine vendor call and put tables into one normalized snapshot."""

    underlying = metadata.get("underlying")
    quote_datetime = metadata.get("quote_datetime") or metadata.get(
        "retrieval_datetime"
    )
    expiry = metadata.get("expiry")
    spot = metadata.get("spot")
    if not underlying:
        raise ValueError("metadata must include underlying")
    if not quote_datetime:
        raise ValueError("metadata must include quote_datetime or retrieval_datetime")
    if spot is None:
        raise ValueError("metadata must include spot")

    calls = _prepare_side(calls_df, "call", underlying, quote_datetime, expiry, spot)
    puts = _prepare_side(puts_df, "put", underlying, quote_datetime, expiry, spot)
    return normalize_market_snapshot(pd.concat([calls, puts], ignore_index=True))


def save_snapshot(
    df: pd.DataFrame,
    path: str | Path,
    metadata_path: str | Path,
    metadata: dict[str, Any],
) -> None:
    """Save a normalized snapshot CSV and companion JSON metadata."""

    path = Path(path)
    metadata_path = Path(metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = normalize_market_snapshot(df)
    snapshot.to_csv(path, index=False)

    metadata_to_write = {
        **metadata,
        "raw_contract_count": int(len(snapshot)),
        "expiries": _metadata_expiries(snapshot, metadata.get("expiries")),
    }
    metadata_path.write_text(
        json.dumps(metadata_to_write, indent=2, sort_keys=True, default=str) + "\n"
    )


def load_snapshot(path: str | Path) -> pd.DataFrame:
    """Load a saved market snapshot from disk."""

    return load_option_chain(path)


def load_snapshot_metadata(path: str | Path) -> dict[str, Any]:
    """Load JSON metadata for a saved market snapshot."""

    return json.loads(Path(path).read_text())


def _prepare_side(
    df: pd.DataFrame,
    option_type: str,
    underlying: str,
    quote_datetime: object,
    expiry: object,
    spot: object,
) -> pd.DataFrame:
    side = df.copy()
    side["underlying"] = underlying
    side["quote_datetime"] = quote_datetime
    if "expiry" not in side:
        side["expiry"] = expiry
    side["option_type"] = option_type
    side["spot"] = spot
    return side


def _normalize_market_column(column: str) -> str:
    normalized = (
        column.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )
    normalized = "_".join(part for part in normalized.split("_") if part)
    return MARKET_COLUMN_ALIASES.get(normalized, normalized)


def _default_value_for_missing_column(column: str) -> object:
    if column in {"volume", "open_interest"}:
        return 0
    if column in {"bid", "ask", "last", "strike", "spot"}:
        return np.nan
    return pd.NA


def _metadata_expiries(snapshot: pd.DataFrame, fallback: object) -> list[str]:
    if fallback:
        return [str(expiry) for expiry in fallback]
    expiries = pd.to_datetime(snapshot["expiry"], errors="coerce").dropna().unique()
    return sorted(str(pd.Timestamp(expiry).date()) for expiry in expiries)
