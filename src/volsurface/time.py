"""Time-to-expiry calculations."""

from __future__ import annotations

import pandas as pd

SECONDS_PER_ACT_365_YEAR = 365.0 * 24.0 * 60.0 * 60.0
EXPIRY_CLOSE_TIMEZONE = "America/New_York"
EXPIRY_CLOSE_HOUR = 16


def compute_time_to_expiry(df: pd.DataFrame) -> pd.DataFrame:
    """Add ACT/365 annualized time-to-expiry in years.

    The calculation is `(expiry - quote_datetime).total_seconds() / 365 days`.
    A date-only expiry (midnight, as in the yfinance chain) is read as the
    16:00 New York close on that date, when SPY options stop trading; the
    conversion to UTC follows daylight saving time. Expiries with an explicit
    time of day are used as given. Expired contracts are kept here with
    non-positive year fractions so the quote filter can report them
    deterministically.
    """

    required = {"quote_datetime", "expiry"}
    missing = required.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"missing columns for time-to-expiry: {missing_text}")

    result = df.copy()
    quote_datetime = pd.to_datetime(result["quote_datetime"], errors="coerce", utc=True)
    expiry = pd.to_datetime(result["expiry"], errors="coerce", utc=True)
    date_only = expiry.notna() & (expiry == expiry.dt.normalize())
    if date_only.any():
        close = (
            expiry[date_only].dt.tz_localize(None)
            + pd.Timedelta(hours=EXPIRY_CLOSE_HOUR)
        )
        close = (
            close.dt.tz_localize(EXPIRY_CLOSE_TIMEZONE)
            .dt.tz_convert("UTC")
        )
        expiry = expiry.where(~date_only, close)
    seconds = (expiry - quote_datetime).dt.total_seconds()
    result["time_to_expiry"] = seconds / SECONDS_PER_ACT_365_YEAR
    return result
