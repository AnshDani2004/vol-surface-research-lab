"""Forward extraction and forward-based moneyness."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pair_calls_and_puts_for_forward(
    df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    near_atm_log_moneyness: float | None = None,
    weight_by_spread: bool = True,
) -> pd.DataFrame:
    """Estimate one forward per expiry from same-strike call/put mid pairs."""

    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite")
    if near_atm_log_moneyness is not None and near_atm_log_moneyness <= 0.0:
        raise ValueError("near_atm_log_moneyness must be positive when supplied")

    required = {"expiry", "option_type", "strike", "mid", "time_to_expiry"}
    missing = required.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"missing columns for forward extraction: {missing_text}")

    grouped = (
        df.groupby(["expiry", "strike", "option_type"], as_index=False)
        .agg(
            mid=("mid", "mean"),
            bid_ask_spread=("bid_ask_spread", "mean"),
            time_to_expiry=("time_to_expiry", "mean"),
        )
        .sort_values(["expiry", "strike", "option_type"])
    )
    pairs = grouped.pivot(
        index=["expiry", "strike", "time_to_expiry"],
        columns="option_type",
        values=["mid", "bid_ask_spread"],
    )
    pairs.columns = [f"{metric}_{option_type}" for metric, option_type in pairs.columns]
    pairs = pairs.reset_index()
    pairs = pairs.dropna(subset=["mid_call", "mid_put"])
    if pairs.empty:
        raise ValueError(
            "no same-strike call/put pairs available for forward extraction"
        )

    pairs["pair_forward"] = pairs["strike"] + np.exp(
        risk_free_rate * pairs["time_to_expiry"]
    ) * (pairs["mid_call"] - pairs["mid_put"])

    if near_atm_log_moneyness is not None:
        initial_forward = pairs.groupby("expiry")["pair_forward"].transform("mean")
        pairs = pairs[
            np.log(pairs["strike"] / initial_forward).abs() <= near_atm_log_moneyness
        ].copy()
        if pairs.empty:
            raise ValueError("near-ATM filter removed all call/put pairs")

    if weight_by_spread:
        combined_spread = pairs["bid_ask_spread_call"] + pairs["bid_ask_spread_put"]
        pairs["weight"] = np.where(combined_spread > 0.0, 1.0 / combined_spread, 1.0)
    else:
        pairs["weight"] = 1.0

    rows = []
    for expiry, expiry_pairs in pairs.groupby("expiry", sort=True):
        weights = expiry_pairs["weight"].to_numpy(dtype=float)
        forwards = expiry_pairs["pair_forward"].to_numpy(dtype=float)
        rows.append(
            {
                "expiry": expiry,
                "time_to_expiry": float(expiry_pairs["time_to_expiry"].mean()),
                "risk_free_rate": float(risk_free_rate),
                "forward": float(np.average(forwards, weights=weights)),
                "paired_strike_count": int(len(expiry_pairs)),
                "min_strike": float(expiry_pairs["strike"].min()),
                "max_strike": float(expiry_pairs["strike"].max()),
            }
        )
    return pd.DataFrame(rows)


def add_forward_moneyness(
    df: pd.DataFrame,
    forward_by_expiry: pd.DataFrame,
    atm_relative_tolerance: float = 0.005,
) -> pd.DataFrame:
    """Add forward, log-moneyness, and OTM surface-side labels."""

    if atm_relative_tolerance < 0.0 or not np.isfinite(atm_relative_tolerance):
        raise ValueError("atm_relative_tolerance must be finite and non-negative")

    required = {"expiry", "strike", "option_type"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing columns for moneyness: {', '.join(sorted(missing))}")
    if "forward" not in forward_by_expiry:
        raise ValueError("forward_by_expiry must contain a forward column")

    forwards = forward_by_expiry.loc[:, ["expiry", "forward"]]
    result = df.merge(forwards, on="expiry", how="left", validate="many_to_one")
    if result["forward"].isna().any():
        raise ValueError("missing forward for one or more expiries")
    if (result["forward"] <= 0.0).any():
        raise ValueError("forward values must be positive")

    result["log_moneyness"] = np.log(result["strike"] / result["forward"])
    relative_distance = (result["strike"] / result["forward"] - 1.0).abs()
    result["option_side_for_surface"] = "not_surface_otm"
    result.loc[
        relative_distance <= atm_relative_tolerance,
        "option_side_for_surface",
    ] = "atm"
    result.loc[
        (relative_distance > atm_relative_tolerance)
        & (result["strike"] < result["forward"])
        & (result["option_type"] == "put"),
        "option_side_for_surface",
    ] = "otm_put"
    result.loc[
        (relative_distance > atm_relative_tolerance)
        & (result["strike"] > result["forward"])
        & (result["option_type"] == "call"),
        "option_side_for_surface",
    ] = "otm_call"
    return result
