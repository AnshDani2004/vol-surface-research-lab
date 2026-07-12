"""Convenience exports for Stage 3 diagnostics."""

from volsurface.arbitrage import (
    check_calendar_monotonicity,
    check_call_convexity,
    check_call_monotonicity,
    check_put_monotonicity,
    run_static_arbitrage_diagnostics,
)
from volsurface.surface import (
    build_otm_surface_table,
    compute_implied_vol_table,
    summarize_surface,
)

__all__ = [
    "build_otm_surface_table",
    "check_calendar_monotonicity",
    "check_call_convexity",
    "check_call_monotonicity",
    "check_put_monotonicity",
    "compute_implied_vol_table",
    "run_static_arbitrage_diagnostics",
    "summarize_surface",
]

