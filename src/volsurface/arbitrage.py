"""Simple static-arbitrage diagnostics on observed filtered quotes."""

from __future__ import annotations

import numpy as np
import pandas as pd

ARBITRAGE_REPORT_COLUMNS = [
    "diagnostic",
    "underlying",
    "expiry",
    "option_type",
    "strike_low",
    "strike_mid",
    "strike_high",
    "strike_or_pair",
    "violation_size",
    "description",
]


def check_call_monotonicity(
    clean_df: pd.DataFrame,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Report calls where price increases as strike increases."""

    return _check_strike_monotonicity(
        clean_df,
        option_type="call",
        diagnostic="call_monotonicity",
        tolerance=tolerance,
    )


def check_put_monotonicity(
    clean_df: pd.DataFrame,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Report puts where price decreases as strike increases."""

    return _check_strike_monotonicity(
        clean_df,
        option_type="put",
        diagnostic="put_monotonicity",
        tolerance=tolerance,
    )


def check_call_convexity(
    clean_df: pd.DataFrame,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Report call butterfly convexity violations using nonuniform strikes."""

    _validate_columns(
        clean_df,
        {"underlying", "expiry", "option_type", "strike", "mid"},
    )
    rows = []
    calls = clean_df[clean_df["option_type"] == "call"]
    for (underlying, expiry), group in calls.groupby(
        ["underlying", "expiry"],
        sort=True,
    ):
        group = _dedupe_by_strike(group)
        if len(group) < 3:
            continue
        strikes = group["strike"].to_numpy(dtype=float)
        mids = group["mid"].to_numpy(dtype=float)
        for index in range(1, len(group) - 1):
            strike_low = strikes[index - 1]
            strike_mid = strikes[index]
            strike_high = strikes[index + 1]
            left_slope = (mids[index] - mids[index - 1]) / (strike_mid - strike_low)
            right_slope = (mids[index + 1] - mids[index]) / (strike_high - strike_mid)
            violation_size = left_slope - right_slope
            if violation_size > tolerance:
                rows.append(
                    _report_row(
                        diagnostic="call_convexity",
                        underlying=underlying,
                        expiry=expiry,
                        option_type="call",
                        strike_low=strike_low,
                        strike_mid=strike_mid,
                        strike_high=strike_high,
                        strike_or_pair=f"{strike_low:g}/{strike_mid:g}/{strike_high:g}",
                        violation_size=violation_size,
                        description="Call prices are not convex in strike.",
                    )
                )
    return pd.DataFrame(rows, columns=ARBITRAGE_REPORT_COLUMNS)


def check_calendar_monotonicity(
    clean_df: pd.DataFrame,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Report simplified calendar monotonicity violations.

    This diagnostic treats same-type, same-strike option value as non-decreasing
    with expiry. It is intentionally simplified because rates, dividends,
    American exercise, and sparse listed strikes can complicate interpretation.
    """

    _validate_columns(
        clean_df,
        {"underlying", "expiry", "option_type", "strike", "mid"},
    )
    rows = []
    for (underlying, option_type, strike), group in clean_df.groupby(
        ["underlying", "option_type", "strike"],
        sort=True,
    ):
        group = group.sort_values("expiry")
        if len(group) < 2:
            continue
        previous = group.iloc[0]
        for _, current in group.iloc[1:].iterrows():
            violation_size = float(previous["mid"] - current["mid"])
            if violation_size > tolerance:
                strike_pair = f"{previous['expiry']}->{current['expiry']} @ {strike:g}"
                rows.append(
                    _report_row(
                        diagnostic="calendar_monotonicity",
                        underlying=underlying,
                        expiry=current["expiry"],
                        option_type=option_type,
                        strike_low=strike,
                        strike_mid=np.nan,
                        strike_high=np.nan,
                        strike_or_pair=strike_pair,
                        violation_size=violation_size,
                        description=(
                            "Later-expiry option mid is lower than earlier-expiry "
                            "mid under a simplified calendar diagnostic."
                        ),
                    )
                )
            previous = current
    return pd.DataFrame(rows, columns=ARBITRAGE_REPORT_COLUMNS)


def run_static_arbitrage_diagnostics(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Combine simple observed-quote static-arbitrage diagnostics."""

    diagnostics = [
        check_call_monotonicity(clean_df),
        check_put_monotonicity(clean_df),
        check_call_convexity(clean_df),
        check_calendar_monotonicity(clean_df),
    ]
    nonempty = [diagnostic for diagnostic in diagnostics if not diagnostic.empty]
    if not nonempty:
        return pd.DataFrame(columns=ARBITRAGE_REPORT_COLUMNS)
    return pd.concat(nonempty, ignore_index=True).loc[:, ARBITRAGE_REPORT_COLUMNS]


def _check_strike_monotonicity(
    clean_df: pd.DataFrame,
    option_type: str,
    diagnostic: str,
    tolerance: float,
) -> pd.DataFrame:
    _validate_columns(
        clean_df,
        {"underlying", "expiry", "option_type", "strike", "mid"},
    )
    rows = []
    quotes = clean_df[clean_df["option_type"] == option_type]
    for (underlying, expiry), group in quotes.groupby(
        ["underlying", "expiry"],
        sort=True,
    ):
        group = _dedupe_by_strike(group)
        if len(group) < 2:
            continue
        previous = group.iloc[0]
        for _, current in group.iloc[1:].iterrows():
            if option_type == "call":
                violation_size = float(current["mid"] - previous["mid"])
                description = "Call mid price increases as strike increases."
            else:
                violation_size = float(previous["mid"] - current["mid"])
                description = "Put mid price decreases as strike increases."
            if violation_size > tolerance:
                rows.append(
                    _report_row(
                        diagnostic=diagnostic,
                        underlying=underlying,
                        expiry=expiry,
                        option_type=option_type,
                        strike_low=float(previous["strike"]),
                        strike_mid=np.nan,
                        strike_high=float(current["strike"]),
                        strike_or_pair=(
                            f"{previous['strike']:g}->{current['strike']:g}"
                        ),
                        violation_size=violation_size,
                        description=description,
                    )
                )
            previous = current
    return pd.DataFrame(rows, columns=ARBITRAGE_REPORT_COLUMNS)


def _dedupe_by_strike(group: pd.DataFrame) -> pd.DataFrame:
    return (
        group.groupby("strike", as_index=False)
        .agg(
            underlying=("underlying", "first"),
            expiry=("expiry", "first"),
            option_type=("option_type", "first"),
            mid=("mid", "mean"),
        )
        .sort_values("strike")
        .reset_index(drop=True)
    )


def _validate_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"missing columns for arbitrage diagnostics: {missing_text}")


def _report_row(
    diagnostic: str,
    underlying: str,
    expiry: object,
    option_type: str,
    strike_low: float,
    strike_mid: float,
    strike_high: float,
    strike_or_pair: str,
    violation_size: float,
    description: str,
) -> dict[str, object]:
    return {
        "diagnostic": diagnostic,
        "underlying": underlying,
        "expiry": expiry,
        "option_type": option_type,
        "strike_low": strike_low,
        "strike_mid": strike_mid,
        "strike_high": strike_high,
        "strike_or_pair": strike_or_pair,
        "violation_size": float(violation_size),
        "description": description,
    }
