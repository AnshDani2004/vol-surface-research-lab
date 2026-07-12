"""Option-chain loading and schema normalization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "underlying",
    "quote_datetime",
    "expiry",
    "option_type",
    "strike",
    "bid",
    "ask",
    "last",
    "volume",
    "open_interest",
    "spot",
]

NUMERIC_COLUMNS = [
    "strike",
    "bid",
    "ask",
    "last",
    "volume",
    "open_interest",
    "spot",
]

OPTION_TYPE_MAP = {
    "c": "call",
    "call": "call",
    "calls": "call",
    "p": "put",
    "put": "put",
    "puts": "put",
}

COLUMN_ALIASES = {
    "expiration": "expiry",
    "expiration_date": "expiry",
    "expiry_date": "expiry",
    "quote_date": "quote_datetime",
    "quotedatetime": "quote_datetime",
    "type": "option_type",
    "right": "option_type",
    "cp_flag": "option_type",
    "underlying_symbol": "underlying",
    "symbol": "underlying",
    "underlying_price": "spot",
}


def load_option_chain(path: str | Path) -> pd.DataFrame:
    """Load an option-chain CSV and normalize it into the Stage 2 schema."""

    chain = pd.read_csv(path)
    normalized_columns = {column: _normalize_column(column) for column in chain}
    chain = chain.rename(columns=normalized_columns)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in chain]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"missing required option-chain columns: {missing}")

    chain["quote_datetime"] = pd.to_datetime(chain["quote_datetime"], errors="coerce")
    chain["expiry"] = pd.to_datetime(chain["expiry"], errors="coerce")
    chain["option_type"] = (
        chain["option_type"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(OPTION_TYPE_MAP)
    )

    invalid_option_type = chain["option_type"].isna()
    if invalid_option_type.any():
        raise ValueError("option_type must normalize to 'call' or 'put'")

    for column in NUMERIC_COLUMNS:
        chain[column] = pd.to_numeric(chain[column], errors="coerce")

    extras = sorted(
        column for column in chain.columns if column not in REQUIRED_COLUMNS
    )
    return chain.loc[:, [*REQUIRED_COLUMNS, *extras]]


def _normalize_column(column: str) -> str:
    normalized = (
        column.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )
    normalized = "_".join(part for part in normalized.split("_") if part)
    return COLUMN_ALIASES.get(normalized, normalized)
