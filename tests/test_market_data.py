from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from volsurface.filters import compute_mid_price, filter_option_quotes
from volsurface.forwards import add_forward_moneyness, pair_calls_and_puts_for_forward
from volsurface.market_data import (
    combine_calls_puts,
    load_snapshot,
    normalize_market_snapshot,
    save_snapshot,
)
from volsurface.surface import build_otm_surface_table, compute_implied_vol_table
from volsurface.time import compute_time_to_expiry


def test_normalizing_calls_and_puts_produces_required_schema() -> None:
    calls = pd.DataFrame(
        {
            "strike": [100.0],
            "bid": [6.0],
            "ask": [6.2],
            "lastPrice": [6.1],
            "volume": [10],
            "openInterest": [100],
        }
    )
    puts = pd.DataFrame(
        {
            "strike": [100.0],
            "bid": [4.8],
            "ask": [5.0],
            "lastPrice": [4.9],
            "volume": [8],
            "openInterest": [90],
        }
    )
    metadata = {
        "underlying": "SPY",
        "retrieval_datetime": "2024-03-15T20:00:00+00:00",
        "expiry": "2024-04-19",
        "spot": 101.0,
    }

    snapshot = combine_calls_puts(calls, puts, metadata)

    assert list(snapshot.columns[:11]) == [
        "underlying",
        "quote_datetime",
        "expiry",
        "option_type",
        "strike",
        "bid",
        "ask",
        "last",
        "volume",
        "open_interest",
        "spot",
    ]
    assert list(snapshot["option_type"]) == ["call", "put"]


def test_missing_market_fields_are_handled_deterministically() -> None:
    snapshot = normalize_market_snapshot(
        pd.DataFrame(
            {
                "underlying": ["SPY"],
                "quote_datetime": ["2024-03-15T20:00:00+00:00"],
                "expiry": ["2024-04-19"],
                "option_type": ["C"],
                "strike": [100.0],
                "spot": [101.0],
            }
        )
    )

    assert pd.isna(snapshot.loc[0, "bid"])
    assert pd.isna(snapshot.loc[0, "ask"])
    assert pd.isna(snapshot.loc[0, "last"])
    assert snapshot.loc[0, "volume"] == 0
    assert snapshot.loc[0, "open_interest"] == 0
    assert snapshot.loc[0, "option_type"] == "call"


def test_save_and_load_snapshot_preserves_required_columns(tmp_path: Path) -> None:
    snapshot = normalize_market_snapshot(
        pd.DataFrame(
            {
                "underlying": ["SPY"],
                "quote_datetime": ["2024-03-15T20:00:00+00:00"],
                "expiry": ["2024-04-19"],
                "option_type": ["put"],
                "strike": [100.0],
                "bid": [4.8],
                "ask": [5.0],
                "last": [4.9],
                "volume": [8],
                "open_interest": [90],
                "spot": [101.0],
            }
        )
    )
    path = tmp_path / "snapshot.csv"
    metadata_path = tmp_path / "metadata.json"
    metadata = {
        "underlying": "SPY",
        "source": "unit-test",
        "retrieval_datetime": "2024-03-15T20:00:00+00:00",
        "spot": 101.0,
    }

    save_snapshot(snapshot, path, metadata_path, metadata)
    loaded = load_snapshot(path)
    loaded_metadata = json.loads(metadata_path.read_text())

    assert list(loaded.columns) == list(snapshot.columns)
    assert loaded_metadata["source"] == "unit-test"
    assert loaded_metadata["retrieval_datetime"] == "2024-03-15T20:00:00+00:00"
    assert loaded_metadata["spot"] == 101.0
    assert loaded_metadata["expiries"] == ["2024-04-19"]
    assert loaded_metadata["raw_contract_count"] == 1


def test_generate_pipeline_runs_on_small_snapshot_fixture_without_internet() -> None:
    calls = pd.DataFrame(
        {
            "strike": [95.0, 100.0, 105.0],
            "bid": [8.9, 6.0, 3.4],
            "ask": [9.1, 6.2, 3.6],
            "lastPrice": [9.0, 6.1, 3.5],
            "volume": [10, 11, 12],
            "openInterest": [100, 110, 120],
        }
    )
    puts = pd.DataFrame(
        {
            "strike": [95.0, 100.0, 105.0],
            "bid": [2.3, 4.8, 7.4],
            "ask": [2.5, 5.0, 7.6],
            "lastPrice": [2.4, 4.9, 7.5],
            "volume": [8, 9, 10],
            "openInterest": [80, 90, 100],
        }
    )
    metadata = {
        "underlying": "SPY",
        "source": "unit-test",
        "retrieval_datetime": "2024-03-15T20:00:00+00:00",
        "expiry": "2024-04-19",
        "spot": 101.0,
    }

    raw = combine_calls_puts(calls, puts, metadata)
    clean, filter_report = filter_option_quotes(
        compute_time_to_expiry(compute_mid_price(raw))
    )
    forward_by_expiry = pair_calls_and_puts_for_forward(clean, risk_free_rate=0.05)
    clean = add_forward_moneyness(clean, forward_by_expiry)
    iv_table = compute_implied_vol_table(clean, risk_free_rate=0.05)
    otm_surface = build_otm_surface_table(iv_table)

    assert len(clean) == 6
    assert filter_report.loc[filter_report["reason"] == "kept", "count"].item() == 6
    assert len(forward_by_expiry) == 1
    assert (iv_table["iv_status"] == "success").all()
    assert not otm_surface.empty

