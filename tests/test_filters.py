from __future__ import annotations

import pandas as pd
import pytest

from volsurface.filters import compute_mid_price, filter_option_quotes
from volsurface.time import compute_time_to_expiry


def test_time_to_expiry_uses_act_365() -> None:
    df = pd.DataFrame(
        {
            "quote_datetime": pd.to_datetime(["2024-01-01 12:00:00"]),
            "expiry": pd.to_datetime(["2024-01-31 18:00:00"]),
        }
    )

    result = compute_time_to_expiry(df)

    assert result.loc[0, "time_to_expiry"] == pytest.approx(30.25 / 365.0)


def test_date_only_expiry_is_the_new_york_close() -> None:
    df = pd.DataFrame(
        {
            "quote_datetime": [
                "2026-07-12 20:00:00+00:00",
                "2026-01-15 21:00:00+00:00",
            ],
            "expiry": ["2026-07-13", "2026-01-16"],
        }
    )

    result = compute_time_to_expiry(df)

    # July: 16:00 EDT is 20:00 UTC; January: 16:00 EST is 21:00 UTC.
    assert result.loc[0, "time_to_expiry"] == pytest.approx(1.0 / 365.0)
    assert result.loc[1, "time_to_expiry"] == pytest.approx(1.0 / 365.0)


def test_compute_mid_price_allows_zero_bid_with_positive_mid() -> None:
    df = pd.DataFrame({"bid": [0.0], "ask": [0.20]})

    result = compute_mid_price(df)

    assert result.loc[0, "mid"] == pytest.approx(0.10)
    assert result.loc[0, "bid_ask_spread"] == pytest.approx(0.20)
    assert result.loc[0, "relative_spread"] == pytest.approx(2.0)


def test_filter_report_counts_match_expected_fixture() -> None:
    df = pd.DataFrame(
        {
            "underlying": ["SPY"] * 8,
            "quote_datetime": ["2024-03-15 15:45:00"] * 8,
            "expiry": [
                "2024-04-19 16:00:00",
                "2024-04-19 16:00:00",
                "2024-04-19 16:00:00",
                "2024-04-19 16:00:00",
                "2024-03-08 16:00:00",
                "2024-04-19 16:00:00",
                "2024-04-19 16:00:00",
                "2024-04-19 16:00:00",
            ],
            "option_type": ["call"] * 8,
            "strike": [500.0, 501.0, 502.0, 503.0, 504.0, 505.0, -1.0, 507.0],
            "bid": [1.0, -0.01, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            "ask": [1.2, 0.10, 0.9, 0.0, 1.2, 0.5, 1.2, 1.2],
            "last": [1.1] * 7 + [None],
            "volume": [1] * 8,
            "open_interest": [1] * 8,
            "spot": [500.0] * 8,
        }
    )
    df = compute_time_to_expiry(compute_mid_price(df))

    clean, report = filter_option_quotes(df)
    counts = dict(zip(report["reason"], report["count"], strict=True))

    assert len(clean) == 1
    assert counts == {
        "kept": 1,
        "missing_required_fields": 1,
        "negative_bid": 1,
        "ask_below_bid": 1,
        "zero_or_negative_ask": 1,
        "zero_or_negative_mid": 0,
        "expired_contract": 1,
        "extremely_wide_relative_spread": 1,
        "nonpositive_strike": 1,
        "nonpositive_spot": 0,
    }


def test_zero_bid_positive_ask_is_allowed_when_not_wide() -> None:
    df = pd.DataFrame(
        {
            "underlying": ["SPY"],
            "quote_datetime": pd.to_datetime(["2024-03-15 15:45:00"]),
            "expiry": pd.to_datetime(["2024-04-19 16:00:00"]),
            "option_type": ["call"],
            "strike": [500.0],
            "bid": [0.0],
            "ask": [0.2],
            "last": [0.1],
            "volume": [1],
            "open_interest": [1],
            "spot": [515.2],
        }
    )

    clean, report = filter_option_quotes(
        compute_time_to_expiry(compute_mid_price(df)),
        max_relative_spread=2.5,
    )

    assert len(clean) == 1
    assert dict(zip(report["reason"], report["count"], strict=True))["kept"] == 1


def test_crossed_quotes_are_filtered() -> None:
    df = pd.DataFrame(
        {
            "underlying": ["SPY"],
            "quote_datetime": pd.to_datetime(["2024-03-15 15:45:00"]),
            "expiry": pd.to_datetime(["2024-04-19 16:00:00"]),
            "option_type": ["call"],
            "strike": [500.0],
            "bid": [2.0],
            "ask": [1.0],
            "last": [1.5],
            "volume": [1],
            "open_interest": [1],
            "spot": [515.2],
        }
    )

    clean, report = filter_option_quotes(compute_time_to_expiry(compute_mid_price(df)))
    counts = dict(zip(report["reason"], report["count"], strict=True))

    assert clean.empty
    assert counts["ask_below_bid"] == 1
