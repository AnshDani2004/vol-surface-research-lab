"""Same-universe model comparison helpers for Stage 5."""

from __future__ import annotations

import numpy as np
import pandas as pd

from volsurface.calibration import calibrate_sabr_per_expiry

MODEL_COMPARISON_EXPIRY_COLUMNS = [
    "expiry",
    "n_points",
    "sabr_rmse_iv",
    "heston_rmse_iv",
    "sabr_mae_iv",
    "heston_mae_iv",
    "sabr_weighted_rmse_iv",
    "heston_weighted_rmse_iv",
    "sabr_max_abs_error",
    "heston_max_abs_error",
    "winner_by_rmse",
    "winner_by_mae",
    "notes",
]

MODEL_COMPARISON_BUCKET_COLUMNS = [
    "bucket",
    "n_points",
    "sabr_rmse_iv",
    "heston_rmse_iv",
    "sabr_mae_iv",
    "heston_mae_iv",
    "sabr_bias",
    "heston_bias",
    "winner_by_rmse",
    "notes",
]


def recompute_sabr_on_universe(
    surface: pd.DataFrame,
    beta: float = 0.5,
    weighting: str = "spread_weighted",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute SABR on the same universe used for Heston comparison."""

    result = calibrate_sabr_per_expiry(surface, beta=beta, weighting=weighting)
    point_rows = result.fit_points.copy()
    point_rows["row_type"] = "point"
    summary_rows = []
    for expiry, group in point_rows.groupby("expiry", sort=True):
        summary_rows.append(
            {
                "row_type": "expiry_summary",
                "expiry": expiry,
                "n_points": int(len(group)),
                "rmse_iv": _rmse(group["iv_error"]),
                "mae_iv": float(group["abs_iv_error"].mean()),
                "weighted_rmse_iv": _weighted_rmse(group),
                "max_abs_error": float(group["abs_iv_error"].max()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    combined = pd.concat([point_rows, summary], ignore_index=True, sort=False)
    return combined, point_rows


def build_model_comparison_by_expiry(
    sabr_points: pd.DataFrame,
    heston_points: pd.DataFrame,
) -> pd.DataFrame:
    """Compare SABR and Heston errors by expiry."""

    rows = []
    merged = _merge_model_points(sabr_points, heston_points)
    for expiry, group in merged.groupby("expiry", sort=True):
        rows.append(
            {
                "expiry": expiry,
                "n_points": int(len(group)),
                "sabr_rmse_iv": _rmse(group["sabr_error"]),
                "heston_rmse_iv": _rmse(group["heston_error"]),
                "sabr_mae_iv": float(group["sabr_error"].abs().mean()),
                "heston_mae_iv": float(group["heston_error"].abs().mean()),
                "sabr_weighted_rmse_iv": _weighted_rmse_from_arrays(
                    group["sabr_error"],
                    group["weight"],
                ),
                "heston_weighted_rmse_iv": _weighted_rmse_from_arrays(
                    group["heston_error"],
                    group["weight"],
                ),
                "sabr_max_abs_error": float(group["sabr_error"].abs().max()),
                "heston_max_abs_error": float(group["heston_error"].abs().max()),
                "winner_by_rmse": _winner(
                    _rmse(group["sabr_error"]),
                    _rmse(group["heston_error"]),
                ),
                "winner_by_mae": _winner(
                    float(group["sabr_error"].abs().mean()),
                    float(group["heston_error"].abs().mean()),
                ),
                "notes": "same filtered universe",
            }
        )
    return pd.DataFrame(rows, columns=MODEL_COMPARISON_EXPIRY_COLUMNS)


def build_model_comparison_by_moneyness(
    sabr_points: pd.DataFrame,
    heston_points: pd.DataFrame,
) -> pd.DataFrame:
    """Compare SABR and Heston errors by log-moneyness bucket."""

    merged = _merge_model_points(sabr_points, heston_points)
    merged["bucket"] = assign_moneyness_bucket(merged["log_moneyness"])
    rows = []
    bucket_order = ["left_wing", "left_mid", "atm", "right_mid", "right_wing"]
    for bucket in bucket_order:
        group = merged[merged["bucket"] == bucket]
        rows.append(
            {
                "bucket": bucket,
                "n_points": int(len(group)),
                "sabr_rmse_iv": _rmse(group["sabr_error"]),
                "heston_rmse_iv": _rmse(group["heston_error"]),
                "sabr_mae_iv": _mean_abs(group["sabr_error"]),
                "heston_mae_iv": _mean_abs(group["heston_error"]),
                "sabr_bias": _mean(group["sabr_error"]),
                "heston_bias": _mean(group["heston_error"]),
                "winner_by_rmse": _winner(
                    _rmse(group["sabr_error"]),
                    _rmse(group["heston_error"]),
                ),
                "notes": "same filtered universe" if len(group) else "empty bucket",
            }
        )
    return pd.DataFrame(rows, columns=MODEL_COMPARISON_BUCKET_COLUMNS)


def assign_moneyness_bucket(log_moneyness: pd.Series | np.ndarray) -> pd.Series:
    """Assign Stage 5 moneyness buckets."""

    values = pd.Series(log_moneyness, dtype=float)
    return pd.cut(
        values,
        bins=[-np.inf, -0.05, -0.01, 0.01, 0.05, np.inf],
        labels=["left_wing", "left_mid", "atm", "right_mid", "right_wing"],
        include_lowest=True,
        right=True,
    ).astype("string")


def build_underperformance_warnings(
    comparison_by_expiry: pd.DataFrame,
    comparison_by_moneyness: pd.DataFrame,
    threshold: float = 0.01,
) -> pd.DataFrame:
    """Report where Heston materially underperforms SABR by RMSE."""

    rows = []
    for row in comparison_by_expiry.itertuples(index=False):
        if row.heston_rmse_iv > row.sabr_rmse_iv + threshold:
            gap = row.heston_rmse_iv - row.sabr_rmse_iv
            rows.append(
                {
                    "category": "heston_underperforms_sabr",
                    "severity": "warning",
                    "expiry": row.expiry,
                    "detail": f"expiry RMSE gap={gap:.6g}",
                }
            )
    for row in comparison_by_moneyness.itertuples(index=False):
        if row.heston_rmse_iv > row.sabr_rmse_iv + threshold:
            gap = row.heston_rmse_iv - row.sabr_rmse_iv
            rows.append(
                {
                    "category": "heston_underperforms_sabr",
                    "severity": "warning",
                    "expiry": "",
                    "detail": f"{row.bucket} RMSE gap={gap:.6g}",
                }
            )
    return pd.DataFrame(rows, columns=["category", "severity", "expiry", "detail"])


def _merge_model_points(
    sabr_points: pd.DataFrame,
    heston_points: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["expiry", "option_type", "strike"]
    sabr = sabr_points.rename(
        columns={
            "iv_error": "sabr_error",
            "sabr_iv": "sabr_iv",
            "market_iv": "market_iv",
        }
    )
    heston = heston_points[heston_points["iv_status"] == "success"].rename(
        columns={"iv_error": "heston_error"}
    )
    columns = [
        *keys,
        "log_moneyness",
        "weight",
        "sabr_error",
        "heston_error",
    ]
    merged = sabr.merge(
        heston.loc[:, [*keys, "heston_error"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    return merged.loc[:, columns]


def _rmse(values: pd.Series) -> float:
    if len(values) == 0:
        return np.nan
    array = values.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(array**2)))


def _weighted_rmse(group: pd.DataFrame) -> float:
    return _weighted_rmse_from_arrays(group["iv_error"], group["weight"])


def _weighted_rmse_from_arrays(errors: pd.Series, weights: pd.Series) -> float:
    if len(errors) == 0:
        return np.nan
    squared_errors = errors.to_numpy(dtype=float) ** 2
    return float(np.sqrt(np.average(squared_errors, weights=weights)))


def _mean_abs(values: pd.Series) -> float:
    return float(values.abs().mean()) if len(values) else np.nan


def _mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else np.nan


def _winner(left: float, right: float) -> str:
    if not np.isfinite(left) or not np.isfinite(right):
        return "not_available"
    if abs(left - right) <= 1e-12:
        return "tie"
    return "sabr" if left < right else "heston"
