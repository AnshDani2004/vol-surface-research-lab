"""Download and save one SPY option-chain snapshot with yfinance.

This script is intentionally separate from deterministic regeneration. The
checked CSV snapshot is the reproducible input for Stage 3.5 outputs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from volsurface.market_data import combine_calls_puts, save_snapshot

OUTPUT_DIR = Path("data/snapshots")
SNAPSHOT_PATH = OUTPUT_DIR / "spy_option_chain_snapshot.csv"
METADATA_PATH = OUTPUT_DIR / "spy_option_chain_snapshot_metadata.json"
UNDERLYING = "SPY"
MAX_EXPIRIES = 6


def main() -> None:
    try:
        import yfinance as yf
    except ImportError as error:
        raise SystemExit(
            "yfinance is required for live snapshot downloads. "
            "Install it with `python -m pip install yfinance`."
        ) from error

    ticker = yf.Ticker(UNDERLYING)
    spot = _get_spot(ticker)
    expiries = list(ticker.options[:MAX_EXPIRIES])
    if not expiries:
        raise SystemExit("No SPY option expiries returned by yfinance.")

    retrieval_datetime = datetime.now(timezone.utc).isoformat()
    snapshots = []
    for expiry in expiries:
        chain = ticker.option_chain(expiry)
        metadata = {
            "underlying": UNDERLYING,
            "source": "yfinance",
            "retrieval_datetime": retrieval_datetime,
            "quote_datetime": retrieval_datetime,
            "spot": spot,
            "expiry": expiry,
        }
        snapshots.append(combine_calls_puts(chain.calls, chain.puts, metadata))

    snapshot = pd.concat(snapshots, ignore_index=True)
    metadata = {
        "underlying": UNDERLYING,
        "source": "yfinance",
        "market_data_timing": "not certified live; yfinance data may be delayed",
        "retrieval_datetime": retrieval_datetime,
        "spot": spot,
        "expiries": expiries,
        "notes": (
            "Saved yfinance SPY option-chain snapshot. Deterministic output "
            "generation uses this checked CSV, not a live download."
        ),
    }
    save_snapshot(snapshot, SNAPSHOT_PATH, METADATA_PATH, metadata)
    print(f"saved {len(snapshot)} contracts to {SNAPSHOT_PATH}")
    print(f"saved metadata to {METADATA_PATH}")


def _get_spot(ticker: object) -> float:
    fast_info = getattr(ticker, "fast_info", {})
    for key in ("last_price", "lastPrice", "regular_market_price"):
        try:
            value = fast_info[key]
        except (KeyError, TypeError):
            continue
        if value is not None:
            return float(value)

    history = ticker.history(period="1d")
    if history.empty:
        raise SystemExit("Could not determine SPY spot from yfinance.")
    return float(history["Close"].iloc[-1])


if __name__ == "__main__":
    main()
