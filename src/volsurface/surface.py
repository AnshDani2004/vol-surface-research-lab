"""Observed implied-volatility table construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from volsurface.black_scholes import implied_vol_bisection

IV_TABLE_COLUMNS = [
    "underlying",
    "quote_datetime",
    "expiry",
    "option_type",
    "strike",
    "mid",
    "bid",
    "ask",
    "spot",
    "forward",
    "time_to_expiry",
    "log_moneyness",
    "option_side_for_surface",
    "implied_vol",
    "iv_status",
    "iv_failure_reason",
]

SURFACE_SIDE_VALUES = {"otm_put", "atm", "otm_call"}


def compute_implied_vol_table(
    df: pd.DataFrame,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    """Compute Black-Scholes implied volatility for each cleaned option row."""

    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite")
    if not np.isfinite(dividend_yield):
        raise ValueError("dividend_yield must be finite")

    required = {
        "underlying",
        "quote_datetime",
        "expiry",
        "option_type",
        "strike",
        "mid",
        "bid",
        "ask",
        "spot",
        "forward",
        "time_to_expiry",
        "log_moneyness",
        "option_side_for_surface",
    }
    missing = required.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"missing columns for implied-vol table: {missing_text}")

    result = df.copy()
    implied_vols: list[float] = []
    statuses: list[str] = []
    failure_reasons: list[str] = []

    for row in result.itertuples(index=False):
        try:
            implied_vol = implied_vol_bisection(
                price=float(row.mid),
                spot=float(row.spot),
                strike=float(row.strike),
                time_to_expiry=float(row.time_to_expiry),
                rate=float(risk_free_rate),
                option_type=row.option_type,
                dividend_yield=float(dividend_yield),
            )
        except (RuntimeError, ValueError, OverflowError, FloatingPointError) as error:
            implied_vols.append(np.nan)
            statuses.append("failed")
            failure_reasons.append(str(error))
        else:
            implied_vols.append(float(implied_vol))
            statuses.append("success")
            failure_reasons.append("")

    result["implied_vol"] = implied_vols
    result["iv_status"] = statuses
    result["iv_failure_reason"] = failure_reasons
    return result.loc[:, IV_TABLE_COLUMNS]


def build_otm_surface_table(iv_df: pd.DataFrame) -> pd.DataFrame:
    """Keep successful OTM/ATM observed quotes for surface construction."""

    required = {
        "iv_status",
        "option_side_for_surface",
        "expiry",
        "strike",
        "option_type",
    }
    missing = required.difference(iv_df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"missing columns for OTM surface table: {missing_text}")

    surface = iv_df[
        (iv_df["iv_status"] == "success")
        & (iv_df["option_side_for_surface"].isin(SURFACE_SIDE_VALUES))
    ].copy()
    return surface.sort_values(["expiry", "strike", "option_type"]).reset_index(
        drop=True
    )


def summarize_surface(
    iv_df: pd.DataFrame,
    otm_surface_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize implied-vol coverage and OTM surface rows by expiry."""

    required = {
        "expiry",
        "strike",
        "log_moneyness",
        "implied_vol",
        "iv_status",
        "option_side_for_surface",
    }
    missing = required.difference(iv_df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"missing columns for surface summary: {missing_text}")

    otm_counts = otm_surface_df.groupby("expiry").size()
    rows = []
    for expiry, expiry_rows in iv_df.groupby("expiry", sort=True):
        successful = expiry_rows[expiry_rows["iv_status"] == "success"]
        failed = expiry_rows[expiry_rows["iv_status"] != "success"]
        atm = successful[successful["option_side_for_surface"] == "atm"]
        rows.append(
            {
                "expiry": expiry,
                "raw_rows": int(len(expiry_rows)),
                "successful_iv_rows": int(len(successful)),
                "failed_iv_rows": int(len(failed)),
                "otm_surface_rows": int(otm_counts.get(expiry, 0)),
                "min_strike": _safe_min(successful["strike"]),
                "max_strike": _safe_max(successful["strike"]),
                "min_log_moneyness": _safe_min(successful["log_moneyness"]),
                "max_log_moneyness": _safe_max(successful["log_moneyness"]),
                "min_iv": _safe_min(successful["implied_vol"]),
                "median_iv": _safe_median(successful["implied_vol"]),
                "max_iv": _safe_max(successful["implied_vol"]),
                "atm_iv_estimate": _safe_median(atm["implied_vol"]),
            }
        )
    return pd.DataFrame(rows)


def _safe_min(values: pd.Series) -> float:
    return float(values.min()) if not values.empty else np.nan


def _safe_max(values: pd.Series) -> float:
    return float(values.max()) if not values.empty else np.nan


def _safe_median(values: pd.Series) -> float:
    return float(values.median()) if not values.empty else np.nan
