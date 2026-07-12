from __future__ import annotations

import pandas as pd

from volsurface.black_scholes import black_scholes_price
from volsurface.data import load_option_chain
from volsurface.filters import compute_mid_price, filter_option_quotes
from volsurface.forwards import add_forward_moneyness, pair_calls_and_puts_for_forward
from volsurface.surface import (
    build_otm_surface_table,
    compute_implied_vol_table,
    summarize_surface,
)
from volsurface.time import compute_time_to_expiry

RISK_FREE_RATE = 0.05


def _stage2_fixture_with_moneyness() -> pd.DataFrame:
    raw = load_option_chain("data/samples/sample_option_chain.csv")
    clean, _ = filter_option_quotes(compute_time_to_expiry(compute_mid_price(raw)))
    forward_by_expiry = pair_calls_and_puts_for_forward(
        clean,
        risk_free_rate=RISK_FREE_RATE,
    )
    return add_forward_moneyness(clean, forward_by_expiry)


def test_iv_table_computes_finite_positive_ivs_on_stage2_fixture() -> None:
    clean = _stage2_fixture_with_moneyness()

    iv_table = compute_implied_vol_table(clean, risk_free_rate=RISK_FREE_RATE)

    assert len(iv_table) == 18
    assert set(iv_table["iv_status"]) == {"success"}
    assert (iv_table["implied_vol"] > 0.0).all()
    assert iv_table["implied_vol"].notna().all()


def test_failed_iv_solve_is_recorded_without_crashing() -> None:
    df = pd.DataFrame(
        {
            "underlying": ["TEST"],
            "quote_datetime": pd.to_datetime(["2024-01-01 10:00:00"]),
            "expiry": pd.to_datetime(["2024-02-01 16:00:00"]),
            "option_type": ["call"],
            "strike": [100.0],
            "mid": [200.0],
            "bid": [199.0],
            "ask": [201.0],
            "spot": [100.0],
            "forward": [100.0],
            "time_to_expiry": [30.0 / 365.0],
            "log_moneyness": [0.0],
            "option_side_for_surface": ["atm"],
        }
    )

    iv_table = compute_implied_vol_table(df, risk_free_rate=0.01)

    assert iv_table.loc[0, "iv_status"] == "failed"
    assert pd.isna(iv_table.loc[0, "implied_vol"])
    assert "no-arbitrage" in iv_table.loc[0, "iv_failure_reason"]


def test_otm_surface_table_keeps_otm_and_atm_success_rows() -> None:
    iv_table = pd.DataFrame(
        {
            "expiry": pd.to_datetime(["2024-02-01"] * 5),
            "strike": [90.0, 95.0, 100.0, 105.0, 110.0],
            "option_type": ["put", "call", "call", "put", "call"],
            "iv_status": ["success", "success", "success", "failed", "success"],
            "option_side_for_surface": [
                "otm_put",
                "not_surface_otm",
                "atm",
                "otm_put",
                "otm_call",
            ],
        }
    )

    surface = build_otm_surface_table(iv_table)

    assert list(surface["option_side_for_surface"]) == ["otm_put", "atm", "otm_call"]
    assert list(surface["strike"]) == [90.0, 100.0, 110.0]


def test_surface_summary_counts_match_expected_fixture() -> None:
    clean = _stage2_fixture_with_moneyness()
    iv_table = compute_implied_vol_table(clean, risk_free_rate=RISK_FREE_RATE)
    surface = build_otm_surface_table(iv_table)

    summary = summarize_surface(iv_table, surface)

    assert summary["raw_rows"].sum() == 18
    assert summary["successful_iv_rows"].sum() == 18
    assert summary["failed_iv_rows"].sum() == 0
    assert summary["otm_surface_rows"].sum() == 11
    assert summary["atm_iv_estimate"].notna().sum() == 2


def test_surface_summary_counts_failed_iv_rows() -> None:
    t = 0.25
    price = black_scholes_price(100.0, 100.0, t, 0.01, 0.2, "call")
    df = pd.DataFrame(
        {
            "underlying": ["TEST", "TEST"],
            "quote_datetime": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "expiry": pd.to_datetime(["2024-04-01", "2024-04-01"]),
            "option_type": ["call", "call"],
            "strike": [100.0, 100.0],
            "mid": [price, 200.0],
            "bid": [price - 0.05, 199.0],
            "ask": [price + 0.05, 201.0],
            "spot": [100.0, 100.0],
            "forward": [100.0, 100.0],
            "time_to_expiry": [t, t],
            "log_moneyness": [0.0, 0.0],
            "option_side_for_surface": ["atm", "atm"],
        }
    )

    iv_table = compute_implied_vol_table(df, risk_free_rate=0.01)
    surface = build_otm_surface_table(iv_table)
    summary = summarize_surface(iv_table, surface)

    assert summary.loc[0, "raw_rows"] == 2
    assert summary.loc[0, "successful_iv_rows"] == 1
    assert summary.loc[0, "failed_iv_rows"] == 1
