from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from volsurface.data import REQUIRED_COLUMNS, load_option_chain


def test_load_option_chain_normalizes_columns_and_order(tmp_path: Path) -> None:
    path = tmp_path / "chain.csv"
    pd.DataFrame(
        {
            "Underlying Symbol": ["SPY"],
            "Quote Date": ["2024-03-15 15:45:00"],
            "Expiration": ["2024-04-19 16:00:00"],
            "Type": ["C"],
            "Strike": [500],
            "Bid": [18.3],
            "Ask": [18.5],
            "Last": [18.4],
            "Volume": [100],
            "Open Interest": [200],
            "Underlying Price": [515.2],
            "Vendor Field": ["kept"],
        }
    ).to_csv(path, index=False)

    chain = load_option_chain(path)

    assert list(chain.columns) == [*REQUIRED_COLUMNS, "vendor_field"]
    assert chain.loc[0, "option_type"] == "call"
    assert pd.api.types.is_datetime64_any_dtype(chain["quote_datetime"])
    assert pd.api.types.is_datetime64_any_dtype(chain["expiry"])


def test_load_option_chain_required_column_validation(tmp_path: Path) -> None:
    path = tmp_path / "missing.csv"
    pd.DataFrame({"underlying": ["SPY"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required option-chain columns"):
        load_option_chain(path)


def test_load_option_chain_rejects_invalid_option_type(tmp_path: Path) -> None:
    path = tmp_path / "invalid_type.csv"
    pd.DataFrame(
        {
            "underlying": ["SPY"],
            "quote_datetime": ["2024-03-15 15:45:00"],
            "expiry": ["2024-04-19 16:00:00"],
            "option_type": ["straddle"],
            "strike": [500],
            "bid": [1.0],
            "ask": [1.2],
            "last": [1.1],
            "volume": [1],
            "open_interest": [1],
            "spot": [515.2],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="option_type"):
        load_option_chain(path)

