from __future__ import annotations

import numpy as np
import pandas as pd

from volsurface.heston_diagnostics import (
    calibrate_per_expiry_heston,
    compact_variant_guesses,
    default_multistart_guesses,
)


def test_per_expiry_calibration_handles_one_expiry() -> None:
    surface = _surface("2026-09-18", 7)

    result = calibrate_per_expiry_heston(
        surface,
        max_nfev=1,
        quadrature_order=16,
    )

    assert not result.params.empty
    assert set(result.params["variant_name"]) == {"per_expiry_heston"}
    assert not result.fit_points.empty


def test_per_expiry_calibration_skips_too_few_points_expiry() -> None:
    surface = pd.concat(
        [_surface("2026-09-18", 7), _surface("2026-10-16", 3)],
        ignore_index=True,
    )

    result = calibrate_per_expiry_heston(
        surface,
        max_nfev=1,
        quadrature_order=16,
        min_points=6,
    )

    assert "too_few_points" in set(result.failure_report["category"])
    assert "2026-10-16" in set(result.failure_report["expiry"])


def test_multistart_guess_builders_are_deterministic() -> None:
    surface = _surface("2026-09-18", 7)

    first = default_multistart_guesses(surface)
    second = default_multistart_guesses(surface)
    compact = compact_variant_guesses(surface)

    assert len(first) == 12
    assert len(compact) == 4
    assert [label for label, _ in first] == [label for label, _ in second]
    assert np.array_equal(first[0][1], second[0][1])


def _surface(expiry: str, n_points: int) -> pd.DataFrame:
    forward = 100.0
    strikes = np.linspace(88.0, 112.0, n_points)
    rows = []
    for strike in strikes:
        option_type = "put" if strike < forward else "call"
        vol = 0.22 + 0.001 * abs(strike - forward)
        rows.append(
            {
                "expiry": expiry,
                "option_type": option_type,
                "strike": strike,
                "forward": forward,
                "time_to_expiry": 0.25,
                "log_moneyness": np.log(strike / forward),
                "option_side_for_surface": "atm" if strike == forward else option_type,
                "implied_vol": vol,
                "mid": 1.0,
                "bid": 0.95,
                "ask": 1.05,
            }
        )
    return pd.DataFrame(rows)
