"""Generate deterministic Stage 3.5 outputs from the saved SPY snapshot."""

from __future__ import annotations

import json
from pathlib import Path

from volsurface.arbitrage import run_static_arbitrage_diagnostics
from volsurface.filters import compute_mid_price, filter_option_quotes
from volsurface.forwards import add_forward_moneyness, pair_calls_and_puts_for_forward
from volsurface.market_data import load_snapshot, load_snapshot_metadata
from volsurface.surface import (
    build_otm_surface_table,
    compute_implied_vol_table,
    summarize_surface,
)
from volsurface.time import compute_time_to_expiry

SNAPSHOT_PATH = Path("data/snapshots/spy_option_chain_snapshot.csv")
METADATA_PATH = Path("data/snapshots/spy_option_chain_snapshot_metadata.json")
OUTPUT_DIR = Path("outputs/stage3_real_snapshot")
RISK_FREE_RATE = 0.05


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_snapshot(SNAPSHOT_PATH)
    with_mid = compute_mid_price(raw)
    with_time = compute_time_to_expiry(with_mid)
    clean, filter_report = filter_option_quotes(with_time)
    forward_by_expiry = pair_calls_and_puts_for_forward(
        clean,
        risk_free_rate=RISK_FREE_RATE,
    )
    clean_with_moneyness = add_forward_moneyness(clean, forward_by_expiry)
    iv_table = compute_implied_vol_table(
        clean_with_moneyness,
        risk_free_rate=RISK_FREE_RATE,
    )
    otm_surface_table = build_otm_surface_table(iv_table)
    arbitrage_report = run_static_arbitrage_diagnostics(clean_with_moneyness)
    surface_summary = summarize_surface(iv_table, otm_surface_table)

    metadata = load_snapshot_metadata(METADATA_PATH)
    metadata = {
        **metadata,
        "deterministic_generation_input": str(SNAPSHOT_PATH),
        "risk_free_rate": RISK_FREE_RATE,
    }

    clean_with_moneyness.to_csv(OUTPUT_DIR / "clean_chain.csv", index=False)
    filter_report.to_csv(OUTPUT_DIR / "filter_report.csv", index=False)
    forward_by_expiry.to_csv(OUTPUT_DIR / "forward_by_expiry.csv", index=False)
    iv_table.to_csv(OUTPUT_DIR / "iv_table.csv", index=False)
    otm_surface_table.to_csv(OUTPUT_DIR / "otm_surface_table.csv", index=False)
    arbitrage_report.to_csv(OUTPUT_DIR / "arbitrage_report.csv", index=False)
    surface_summary.to_csv(OUTPUT_DIR / "surface_summary.csv", index=False)
    (OUTPUT_DIR / "snapshot_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
