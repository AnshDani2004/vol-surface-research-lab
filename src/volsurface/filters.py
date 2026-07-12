"""Deterministic option quote cleaning filters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from volsurface.data import REQUIRED_COLUMNS
from volsurface.time import compute_time_to_expiry

FILTER_REASONS = [
    "missing_required_fields",
    "negative_bid",
    "ask_below_bid",
    "zero_or_negative_ask",
    "zero_or_negative_mid",
    "expired_contract",
    "extremely_wide_relative_spread",
    "nonpositive_strike",
    "nonpositive_spot",
]


def compute_mid_price(df: pd.DataFrame) -> pd.DataFrame:
    """Add mid price, absolute bid-ask spread, and relative spread."""

    required = {"bid", "ask"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing columns for mid-price: {', '.join(sorted(missing))}")

    result = df.copy()
    result["bid_ask_spread"] = result["ask"] - result["bid"]
    result["mid"] = (result["bid"] + result["ask"]) / 2.0
    result["relative_spread"] = np.where(
        result["mid"] > 0.0,
        result["bid_ask_spread"] / result["mid"],
        np.nan,
    )
    return result


def filter_option_quotes(
    df: pd.DataFrame,
    max_relative_spread: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter bad quotes and return `(cleaned_df, filter_report)`.

    Rows are assigned the first failing reason in `FILTER_REASONS` order. The
    report includes `kept` and every configured filter reason, including zeros.
    """

    if max_relative_spread <= 0.0 or not np.isfinite(max_relative_spread):
        raise ValueError("max_relative_spread must be a finite positive number")

    result = df.copy()
    has_spread_columns = {"mid", "bid_ask_spread", "relative_spread"}.issubset(
        result.columns
    )
    if not has_spread_columns:
        result = compute_mid_price(result)
    if "time_to_expiry" not in result:
        result = compute_time_to_expiry(result)

    result["filter_reason"] = pd.NA
    _mark_reason(result, _missing_required_fields(result), "missing_required_fields")
    _mark_reason(result, result["bid"] < 0.0, "negative_bid")
    _mark_reason(result, result["ask"] < result["bid"], "ask_below_bid")
    _mark_reason(result, result["ask"] <= 0.0, "zero_or_negative_ask")
    _mark_reason(result, result["mid"] <= 0.0, "zero_or_negative_mid")
    _mark_reason(result, result["time_to_expiry"] <= 0.0, "expired_contract")
    _mark_reason(
        result,
        result["relative_spread"] > max_relative_spread,
        "extremely_wide_relative_spread",
    )
    _mark_reason(result, result["strike"] <= 0.0, "nonpositive_strike")
    _mark_reason(result, result["spot"] <= 0.0, "nonpositive_spot")

    clean = result[result["filter_reason"].isna()].copy()
    clean = clean.drop(columns=["filter_reason"]).reset_index(drop=True)
    report = _build_filter_report(result)
    return clean, report


def _missing_required_fields(df: pd.DataFrame) -> pd.Series:
    present_required = [column for column in REQUIRED_COLUMNS if column in df]
    missing = df[present_required].isna().any(axis=1)
    if len(present_required) != len(REQUIRED_COLUMNS):
        missing = pd.Series(True, index=df.index)
    return missing


def _mark_reason(df: pd.DataFrame, mask: pd.Series, reason: str) -> None:
    unset = df["filter_reason"].isna()
    df.loc[unset & mask.fillna(False), "filter_reason"] = reason


def _build_filter_report(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["filter_reason"].value_counts(dropna=False)
    rows = [
        {"reason": "kept", "count": int(df["filter_reason"].isna().sum())},
        *[
            {"reason": reason, "count": int(counts.get(reason, 0))}
            for reason in FILTER_REASONS
        ],
    ]
    return pd.DataFrame(rows, columns=["reason", "count"])
