"""Generate deterministic Stage 4.5 SABR robustness outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from volsurface.plotting import save_sabr_robustness_plots
from volsurface.robustness import run_sabr_robustness

OTM_SURFACE_PATH = Path("outputs/stage3_real_snapshot/otm_surface_table.csv")
CLEAN_CHAIN_PATH = Path("outputs/stage3_real_snapshot/clean_chain.csv")
OUTPUT_DIR = Path("outputs/stage4_5_sabr_robustness")
MERGE_KEYS = ["underlying", "quote_datetime", "expiry", "option_type", "strike"]
OPTIONAL_COLUMNS = [
    "bid_ask_spread",
    "relative_spread",
    "volume",
    "open_interest",
    "last_trade_datetime",
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    surface = _load_enriched_surface()
    outputs = run_sabr_robustness(surface)

    outputs.robustness_grid.to_csv(OUTPUT_DIR / "robustness_grid.csv", index=False)
    outputs.robustness_summary.to_csv(
        OUTPUT_DIR / "robustness_summary.csv",
        index=False,
    )
    outputs.filtered_universe_summary.to_csv(
        OUTPUT_DIR / "filtered_universe_summary.csv",
        index=False,
    )
    outputs.beta_sensitivity.to_csv(OUTPUT_DIR / "beta_sensitivity.csv", index=False)
    outputs.weighting_sensitivity.to_csv(
        OUTPUT_DIR / "weighting_sensitivity.csv",
        index=False,
    )
    outputs.expiry_filter_sensitivity.to_csv(
        OUTPUT_DIR / "expiry_filter_sensitivity.csv",
        index=False,
    )
    outputs.liquidity_filter_sensitivity.to_csv(
        OUTPUT_DIR / "liquidity_filter_sensitivity.csv",
        index=False,
    )
    outputs.recommended_universe.to_csv(
        OUTPUT_DIR / "recommended_calibration_universe.csv",
        index=False,
    )
    outputs.warning_summary.to_csv(OUTPUT_DIR / "warning_summary.csv", index=False)
    save_sabr_robustness_plots(
        outputs.robustness_grid,
        outputs.warning_summary,
        outputs.beta_sensitivity,
        outputs.weighting_sensitivity,
        outputs.base_fit_points,
        outputs.recommended_fit_points,
        OUTPUT_DIR / "plots",
    )


def _load_enriched_surface() -> pd.DataFrame:
    surface = pd.read_csv(OTM_SURFACE_PATH)
    clean = pd.read_csv(CLEAN_CHAIN_PATH)
    available = [column for column in OPTIONAL_COLUMNS if column in clean]
    enrich = clean.loc[:, [*MERGE_KEYS, *available]].drop_duplicates(MERGE_KEYS)
    merged = surface.merge(enrich, on=MERGE_KEYS, how="left", validate="one_to_one")
    for column in ("relative_spread", "volume", "open_interest"):
        if column not in merged:
            raise ValueError(
                f"missing required robustness column after merge: {column}"
            )
    return merged


if __name__ == "__main__":
    main()
