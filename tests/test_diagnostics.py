from __future__ import annotations

import volsurface.diagnostics as diagnostics


def test_diagnostics_module_exports_stage3_helpers() -> None:
    assert callable(diagnostics.compute_implied_vol_table)
    assert callable(diagnostics.build_otm_surface_table)
    assert callable(diagnostics.summarize_surface)
    assert callable(diagnostics.run_static_arbitrage_diagnostics)

