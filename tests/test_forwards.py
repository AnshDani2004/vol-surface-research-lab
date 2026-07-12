from __future__ import annotations

import math

import pandas as pd
import pytest

from volsurface.forwards import add_forward_moneyness, pair_calls_and_puts_for_forward


def test_put_call_parity_forward_recovers_known_forward() -> None:
    expiry = pd.Timestamp("2024-04-19 16:00:00")
    rate = 0.05
    time_to_expiry = 0.25
    forward = 105.0
    rows = []
    for strike, put_mid in [(95.0, 1.8), (105.0, 5.0), (115.0, 12.2)]:
        call_mid = put_mid + math.exp(-rate * time_to_expiry) * (forward - strike)
        rows.extend(
            [
                {
                    "expiry": expiry,
                    "option_type": "call",
                    "strike": strike,
                    "mid": call_mid,
                    "bid_ask_spread": 0.2,
                    "time_to_expiry": time_to_expiry,
                },
                {
                    "expiry": expiry,
                    "option_type": "put",
                    "strike": strike,
                    "mid": put_mid,
                    "bid_ask_spread": 0.2,
                    "time_to_expiry": time_to_expiry,
                },
            ]
        )

    forward_by_expiry = pair_calls_and_puts_for_forward(
        pd.DataFrame(rows),
        risk_free_rate=rate,
    )

    assert len(forward_by_expiry) == 1
    assert forward_by_expiry.loc[0, "forward"] == pytest.approx(forward)
    assert forward_by_expiry.loc[0, "paired_strike_count"] == 3


def test_log_moneyness_uses_forward_not_spot() -> None:
    df = pd.DataFrame(
        {
            "expiry": pd.to_datetime(["2024-04-19 16:00:00"]),
            "option_type": ["call"],
            "strike": [110.0],
            "spot": [100.0],
        }
    )
    forward_by_expiry = pd.DataFrame(
        {
            "expiry": pd.to_datetime(["2024-04-19 16:00:00"]),
            "forward": [105.0],
        }
    )

    result = add_forward_moneyness(df, forward_by_expiry)

    assert result.loc[0, "log_moneyness"] == pytest.approx(math.log(110.0 / 105.0))
    assert result.loc[0, "option_side_for_surface"] == "otm_call"


def test_atm_region_is_labeled_explicitly() -> None:
    df = pd.DataFrame(
        {
            "expiry": pd.to_datetime(["2024-04-19 16:00:00"]),
            "option_type": ["put"],
            "strike": [100.2],
        }
    )
    forward_by_expiry = pd.DataFrame(
        {
            "expiry": pd.to_datetime(["2024-04-19 16:00:00"]),
            "forward": [100.0],
        }
    )

    result = add_forward_moneyness(df, forward_by_expiry, atm_relative_tolerance=0.005)

    assert result.loc[0, "option_side_for_surface"] == "atm"

