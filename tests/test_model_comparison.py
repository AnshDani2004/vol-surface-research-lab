from __future__ import annotations

import numpy as np
import pandas as pd

from volsurface.model_comparison import (
    MODEL_COMPARISON_BUCKET_COLUMNS,
    MODEL_COMPARISON_EXPIRY_COLUMNS,
    assign_moneyness_bucket,
    build_model_comparison_by_expiry,
    build_model_comparison_by_moneyness,
    build_underperformance_warnings,
)


def _sabr_points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "expiry": ["2026-08-21"] * 5,
            "option_type": ["put", "put", "call", "call", "call"],
            "strike": [90.0, 98.0, 100.0, 103.0, 108.0],
            "log_moneyness": [-0.105, -0.0202, 0.0, 0.0296, 0.077],
            "weight": [1.0, 2.0, 2.0, 2.0, 1.0],
            "iv_error": [0.01, -0.005, 0.0, 0.004, -0.01],
            "abs_iv_error": [0.01, 0.005, 0.0, 0.004, 0.01],
            "market_iv": [0.2] * 5,
            "sabr_iv": [0.21, 0.195, 0.2, 0.204, 0.19],
        }
    )


def _heston_points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "expiry": ["2026-08-21"] * 5,
            "option_type": ["put", "put", "call", "call", "call"],
            "strike": [90.0, 98.0, 100.0, 103.0, 108.0],
            "log_moneyness": [-0.105, -0.0202, 0.0, 0.0296, 0.077],
            "weight": [1.0, 2.0, 2.0, 2.0, 1.0],
            "iv_error": [0.03, -0.02, 0.01, 0.02, -0.015],
            "abs_iv_error": [0.03, 0.02, 0.01, 0.02, 0.015],
            "market_iv": [0.2] * 5,
            "heston_iv": [0.23, 0.18, 0.21, 0.22, 0.185],
            "iv_status": ["success"] * 5,
        }
    )


def test_assign_moneyness_bucket() -> None:
    buckets = assign_moneyness_bucket(np.asarray([-0.2, -0.02, 0.0, 0.02, 0.2]))

    assert buckets.tolist() == [
        "left_wing",
        "left_mid",
        "atm",
        "right_mid",
        "right_wing",
    ]


def test_model_comparison_by_expiry_has_expected_columns_and_winner() -> None:
    comparison = build_model_comparison_by_expiry(_sabr_points(), _heston_points())

    assert list(comparison.columns) == MODEL_COMPARISON_EXPIRY_COLUMNS
    assert comparison.loc[0, "n_points"] == 5
    assert comparison.loc[0, "winner_by_rmse"] == "sabr"


def test_model_comparison_by_moneyness_has_all_buckets() -> None:
    comparison = build_model_comparison_by_moneyness(_sabr_points(), _heston_points())

    assert list(comparison.columns) == MODEL_COMPARISON_BUCKET_COLUMNS
    assert comparison["bucket"].tolist() == [
        "left_wing",
        "left_mid",
        "atm",
        "right_mid",
        "right_wing",
    ]
    assert comparison["n_points"].sum() == 5


def test_underperformance_warnings_are_generated() -> None:
    by_expiry = build_model_comparison_by_expiry(_sabr_points(), _heston_points())
    by_bucket = build_model_comparison_by_moneyness(_sabr_points(), _heston_points())

    warnings = build_underperformance_warnings(by_expiry, by_bucket, threshold=0.001)

    assert not warnings.empty
    assert set(warnings["category"]) == {"heston_underperforms_sabr"}
