"""Generate deterministic Stage 3 surface and diagnostics outputs."""

from __future__ import annotations

from pathlib import Path

from volsurface.arbitrage import run_static_arbitrage_diagnostics
from volsurface.data import load_option_chain
from volsurface.filters import compute_mid_price, filter_option_quotes
from volsurface.forwards import add_forward_moneyness, pair_calls_and_puts_for_forward
from volsurface.surface import (
    build_otm_surface_table,
    compute_implied_vol_table,
    summarize_surface,
)
from volsurface.time import compute_time_to_expiry

INPUT_PATH = Path("data/samples/sample_option_chain.csv")
OUTPUT_DIR = Path("outputs/stage3_surface")
RISK_FREE_RATE = 0.05


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_option_chain(INPUT_PATH)
    clean, _ = filter_option_quotes(compute_time_to_expiry(compute_mid_price(raw)))
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

    iv_table.to_csv(OUTPUT_DIR / "iv_table.csv", index=False)
    otm_surface_table.to_csv(OUTPUT_DIR / "otm_surface_table.csv", index=False)
    arbitrage_report.to_csv(OUTPUT_DIR / "arbitrage_report.csv", index=False)
    surface_summary.to_csv(OUTPUT_DIR / "surface_summary.csv", index=False)


if __name__ == "__main__":
    main()

