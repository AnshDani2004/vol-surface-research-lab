"""Audit sensitivity checks (exploratory; added during the resume audit).

Two questions about the headline single-snapshot results, answered without
changing any pipeline code:

1. Expiry clock. `volsurface.time.compute_time_to_expiry` originally
   measured time to 00:00 UTC on the expiry date; on branch
   audit-expiry-clock it measures to the 16:00 New York close. This reruns
   the Stage 3 to Stage 4.5 SABR path under both clocks and reports the
   per-expiry and median SABR RMSE.
2. Heston quadrature. Stage 5 calibrates global Heston with 32 Gauss-Laguerre
   nodes. This recalibrates on the recommended universe at 32, 96 and 192
   nodes.

Outputs go to outputs/audit/. These numbers are exploratory diagnostics, not
replacements for the Stage 4 to Stage 5 headline outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from volsurface.calibration import calibrate_sabr_per_expiry
from volsurface.filters import compute_mid_price, filter_option_quotes
from volsurface.forwards import add_forward_moneyness, pair_calls_and_puts_for_forward
from volsurface.heston_calibration import calibrate_heston_global
from volsurface.market_data import load_snapshot
from volsurface.robustness import DEFAULT_MIN_POINTS, select_recommended_universe
from volsurface.surface import build_otm_surface_table, compute_implied_vol_table
from volsurface.time import compute_time_to_expiry

SNAPSHOT_PATH = Path("data/snapshots/spy_option_chain_snapshot.csv")
UNIVERSE_PATH = Path(
    "outputs/stage4_5_sabr_robustness/recommended_calibration_universe.csv"
)
OUTPUT_DIR = Path("outputs/audit")
RISK_FREE_RATE = 0.05
MERGE_KEYS = ["underlying", "quote_datetime", "expiry", "option_type", "strike"]
LIQUIDITY_COLUMNS = ["bid_ask_spread", "relative_spread", "volume", "open_interest"]
SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0


def _original_clock(df: pd.DataFrame) -> pd.DataFrame:
    """The pre-fix convention: time to 00:00 UTC on the expiry date."""
    result = df.copy()
    quote_datetime = pd.to_datetime(
        result["quote_datetime"], errors="coerce", utc=True
    )
    expiry = pd.to_datetime(result["expiry"], errors="coerce", utc=True)
    seconds = (expiry - quote_datetime).dt.total_seconds()
    result["time_to_expiry"] = seconds / SECONDS_PER_YEAR
    return result


def expiry_clock_check() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_snapshot(SNAPSHOT_PATH)
    per_expiry = []
    summary = []
    for label, clock in (
        ("expiry_00_utc_original", _original_clock),
        ("expiry_1600_new_york", compute_time_to_expiry),
    ):
        frame = clock(compute_mid_price(raw))
        clean, _ = filter_option_quotes(frame)
        forwards = pair_calls_and_puts_for_forward(clean, risk_free_rate=RISK_FREE_RATE)
        clean = add_forward_moneyness(clean, forwards)
        iv_table = compute_implied_vol_table(clean, risk_free_rate=RISK_FREE_RATE)
        surface = build_otm_surface_table(iv_table)
        extra = [c for c in LIQUIDITY_COLUMNS if c not in surface.columns]
        if extra:
            surface = surface.merge(
                clean[MERGE_KEYS + extra], on=MERGE_KEYS, how="left", validate="1:1"
            )
        recommended = select_recommended_universe(surface)
        for universe, rows in (("otm_surface", surface), ("recommended", recommended)):
            params = calibrate_sabr_per_expiry(
                rows,
                beta=0.5,
                weighting="spread_weighted",
                min_points=DEFAULT_MIN_POINTS,
            ).params
            for _, row in params.iterrows():
                per_expiry.append(
                    {
                        "clock": label,
                        "universe": universe,
                        "expiry": row["expiry"],
                        "T": row["T"],
                        "n_points": row["n_points"],
                        "rmse_iv": row["rmse_iv"],
                    }
                )
            summary.append(
                {
                    "clock": label,
                    "universe": universe,
                    "rows": len(rows),
                    "expiries": int(rows["expiry"].nunique()),
                    "median_rmse_iv": float(np.nanmedian(params["rmse_iv"])),
                }
            )
    return pd.DataFrame(per_expiry), pd.DataFrame(summary)


def heston_quadrature_check() -> pd.DataFrame:
    universe = pd.read_csv(UNIVERSE_PATH)
    rows = []
    for nodes in (32, 96, 192):
        result = calibrate_heston_global(
            universe, weighting="spread_weighted", quadrature_order=nodes
        )
        params = result.params.iloc[0]
        for _, err in result.fit_errors.iterrows():
            rows.append(
                {
                    "quadrature_nodes": nodes,
                    "scope": err["scope"],
                    "expiry": err["expiry"],
                    "points": err["total_points"],
                    "rmse_iv": err["rmse_iv"],
                    **{k: params[k] for k in ("v0", "theta", "kappa", "sigma", "rho")},
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_expiry, summary = expiry_clock_check()
    per_expiry.to_csv(OUTPUT_DIR / "expiry_clock_sabr_by_expiry.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "expiry_clock_sabr_summary.csv", index=False)
    heston_quadrature_check().to_csv(
        OUTPUT_DIR / "heston_quadrature_global_fit.csv", index=False
    )


if __name__ == "__main__":
    main()
