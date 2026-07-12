from __future__ import annotations

import pandas as pd

from volsurface.arbitrage import (
    check_calendar_monotonicity,
    check_call_convexity,
    check_call_monotonicity,
    check_put_monotonicity,
    run_static_arbitrage_diagnostics,
)


def _quotes(
    option_type: str,
    strikes: list[float],
    mids: list[float],
    expiry: str = "2024-04-19",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "underlying": ["TEST"] * len(strikes),
            "expiry": pd.to_datetime([expiry] * len(strikes)),
            "option_type": [option_type] * len(strikes),
            "strike": strikes,
            "mid": mids,
        }
    )


def test_call_monotonicity_detects_crafted_violation() -> None:
    df = _quotes("call", [90.0, 100.0, 110.0], [12.0, 13.0, 4.0])

    report = check_call_monotonicity(df)

    assert len(report) == 1
    assert report.loc[0, "diagnostic"] == "call_monotonicity"
    assert report.loc[0, "violation_size"] == 1.0


def test_put_monotonicity_detects_crafted_violation() -> None:
    df = _quotes("put", [90.0, 100.0, 110.0], [2.0, 1.0, 12.0])

    report = check_put_monotonicity(df)

    assert len(report) == 1
    assert report.loc[0, "diagnostic"] == "put_monotonicity"
    assert report.loc[0, "violation_size"] == 1.0


def test_call_convexity_detects_crafted_butterfly_violation() -> None:
    df = _quotes("call", [90.0, 100.0, 110.0], [20.0, 14.0, 0.0])

    report = check_call_convexity(df)

    assert len(report) == 1
    assert report.loc[0, "diagnostic"] == "call_convexity"
    assert report.loc[0, "strike_mid"] == 100.0


def test_calendar_monotonicity_detects_crafted_violation() -> None:
    df = pd.DataFrame(
        {
            "underlying": ["TEST", "TEST"],
            "expiry": pd.to_datetime(["2024-04-19", "2024-05-17"]),
            "option_type": ["call", "call"],
            "strike": [100.0, 100.0],
            "mid": [5.0, 4.0],
        }
    )

    report = check_calendar_monotonicity(df)

    assert len(report) == 1
    assert report.loc[0, "diagnostic"] == "calendar_monotonicity"
    assert report.loc[0, "violation_size"] == 1.0


def test_static_arbitrage_diagnostics_return_zero_for_clean_synthetic_fixture() -> None:
    expiries = pd.to_datetime(["2024-04-19", "2024-05-17"])
    rows = []
    for expiry, call_mids, put_mids in [
        (expiries[0], [20.0, 11.0, 5.0], [2.0, 6.0, 13.0]),
        (expiries[1], [21.0, 12.0, 6.0], [3.0, 7.0, 14.0]),
    ]:
        for strike, mid in zip([90.0, 100.0, 110.0], call_mids, strict=True):
            rows.append(
                {
                    "underlying": "TEST",
                    "expiry": expiry,
                    "option_type": "call",
                    "strike": strike,
                    "mid": mid,
                }
            )
        for strike, mid in zip([90.0, 100.0, 110.0], put_mids, strict=True):
            rows.append(
                {
                    "underlying": "TEST",
                    "expiry": expiry,
                    "option_type": "put",
                    "strike": strike,
                    "mid": mid,
                }
            )

    report = run_static_arbitrage_diagnostics(pd.DataFrame(rows))

    assert report.empty
    assert list(report.columns) == [
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
