from __future__ import annotations

import numpy as np
import pandas as pd

from volsurface.calibration import compute_sabr_weights
from volsurface.robustness import (
    ROBUSTNESS_GRID_COLUMNS,
    ROBUSTNESS_SUMMARY_COLUMNS,
    WARNING_COLUMNS,
    apply_expiry_filter,
    apply_liquidity_filter,
    build_warning_summary,
    run_sabr_robustness,
    select_recommended_universe,
)
from volsurface.sabr import hagan_sabr_iv


def _surface(time_to_expiry: float = 0.02) -> pd.DataFrame:
    forward = 100.0
    strikes = [85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0]
    rows = []
    for index, strike in enumerate(strikes):
        option_type = "put" if strike < forward else "call"
        side = "atm" if strike == forward else f"otm_{option_type}"
        rows.append(
            {
                "underlying": "TEST",
                "quote_datetime": "2026-01-01T15:00:00+00:00",
                "expiry": "2026-02-01",
                "option_type": option_type,
                "strike": strike,
                "bid": 1.0 + index,
                "ask": 1.1 + index,
                "mid": 1.05 + index,
                "spot": 100.0,
                "forward": forward,
                "time_to_expiry": time_to_expiry,
                "log_moneyness": np.log(strike / forward),
                "option_side_for_surface": side,
                "implied_vol": hagan_sabr_iv(
                    forward,
                    strike,
                    time_to_expiry,
                    alpha=2.0,
                    beta=0.5,
                    rho=-0.2,
                    nu=0.7,
                ),
                "bid_ask_spread": 0.1,
                "relative_spread": 0.05,
                "volume": 10,
                "open_interest": 100,
            }
        )
    return pd.DataFrame(rows)


def test_expiry_filter_thresholds() -> None:
    surface = pd.concat([_surface(0.003), _surface(0.02)], ignore_index=True)

    filtered = apply_expiry_filter(surface, "exclude_T_lt_2_trading_days")

    assert filtered["time_to_expiry"].min() >= 2.0 / 252.0
    assert len(filtered) == len(_surface(0.02))


def test_liquidity_filters() -> None:
    surface = _surface()
    surface.loc[0, "relative_spread"] = 0.4
    surface.loc[1, "volume"] = 0
    surface.loc[2, "open_interest"] = 0

    spread_filtered = apply_liquidity_filter(surface, "relative_spread_lte_0_25")
    combined = apply_liquidity_filter(
        surface,
        "relative_spread_lte_0_25_and_volume_gt_0_and_open_interest_gt_0",
    )

    assert len(spread_filtered) == len(surface) - 1
    assert len(combined) == len(surface) - 3


def test_warning_classifier_catches_bound_hits() -> None:
    grid = pd.DataFrame(
        {
            "scenario_group": ["unit"],
            "scenario": ["bounds"],
            "beta": [0.5],
            "weighting": ["spread_weighted"],
            **{column: [False] for column in WARNING_COLUMNS},
        }
    )
    grid["alpha_near_bound"] = True
    grid["nu_near_bound"] = True

    summary = build_warning_summary(grid)

    counts = dict(zip(summary["warning"], summary["count"], strict=True))
    assert counts["alpha_near_bound"] == 1
    assert counts["nu_near_bound"] == 1


def test_beta_grid_runner_returns_expected_rows() -> None:
    outputs = run_sabr_robustness(_surface())

    beta_rows = outputs.beta_sensitivity

    assert set(beta_rows["beta"]) == {0.0, 0.5, 1.0}
    assert set(beta_rows["scenario"]) == {"base_stage4", "recommended"}


def test_weighting_sensitivity_runner_handles_expected_modes() -> None:
    outputs = run_sabr_robustness(_surface())

    assert set(outputs.weighting_sensitivity["weighting"]) == {
        "unweighted",
        "spread_weighted",
        "vega_weighted",
    }


def test_vega_weighting_works_with_required_columns() -> None:
    weights = compute_sabr_weights(_surface(), "vega_weighted")

    assert weights.notna().all()
    assert (weights > 0.0).all()
    assert weights.max() <= 20.0


def test_recommended_universe_selection_is_deterministic() -> None:
    surface = _surface()
    surface.loc[0, "relative_spread"] = 0.4
    surface.loc[1, "volume"] = 0
    surface.loc[2, "open_interest"] = 0

    first = select_recommended_universe(surface)
    second = select_recommended_universe(surface)

    assert first.equals(second)
    assert len(first) == len(surface) - 3


def test_robustness_outputs_have_required_columns() -> None:
    outputs = run_sabr_robustness(_surface())

    assert list(outputs.robustness_grid.columns) == ROBUSTNESS_GRID_COLUMNS
    for column in ROBUSTNESS_SUMMARY_COLUMNS:
        assert column in outputs.robustness_summary.columns
    assert not outputs.warning_summary.empty

