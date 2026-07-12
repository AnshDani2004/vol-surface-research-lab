"""Time-to-expiry calculations."""

from __future__ import annotations

import pandas as pd

SECONDS_PER_ACT_365_YEAR = 365.0 * 24.0 * 60.0 * 60.0


def compute_time_to_expiry(df: pd.DataFrame) -> pd.DataFrame:
    """Add ACT/365 annualized time-to-expiry in years.

    The calculation is `(expiry - quote_datetime).total_seconds() / 365 days`.
    Expired contracts are kept here with non-positive year fractions so the
    quote filter can report them deterministically.
    """

    required = {"quote_datetime", "expiry"}
    missing = required.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"missing columns for time-to-expiry: {missing_text}")

    result = df.copy()
    quote_datetime = pd.to_datetime(result["quote_datetime"], errors="coerce")
    expiry = pd.to_datetime(result["expiry"], errors="coerce")
    seconds = (expiry - quote_datetime).dt.total_seconds()
    result["time_to_expiry"] = seconds / SECONDS_PER_ACT_365_YEAR
    return result
