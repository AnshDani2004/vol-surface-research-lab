"""Generate deterministic Stage 2 chain-cleaning outputs."""

from __future__ import annotations

from pathlib import Path

from volsurface.data import load_option_chain
from volsurface.filters import compute_mid_price, filter_option_quotes
from volsurface.forwards import add_forward_moneyness, pair_calls_and_puts_for_forward
from volsurface.time import compute_time_to_expiry

INPUT_PATH = Path("data/samples/sample_option_chain.csv")
OUTPUT_DIR = Path("outputs/stage2_chain_cleaning")
RISK_FREE_RATE = 0.05


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_option_chain(INPUT_PATH)
    with_mid = compute_mid_price(raw)
    with_time = compute_time_to_expiry(with_mid)
    clean, report = filter_option_quotes(with_time)
    forward_by_expiry = pair_calls_and_puts_for_forward(
        clean,
        risk_free_rate=RISK_FREE_RATE,
    )
    clean_with_moneyness = add_forward_moneyness(clean, forward_by_expiry)

    clean_with_moneyness.to_csv(OUTPUT_DIR / "clean_chain.csv", index=False)
    report.to_csv(OUTPUT_DIR / "filter_report.csv", index=False)
    forward_by_expiry.to_csv(OUTPUT_DIR / "forward_by_expiry.csv", index=False)


if __name__ == "__main__":
    main()

