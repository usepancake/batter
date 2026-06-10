"""Feature 3 — sensitivity heatmap: total_return_grid on SensitivityResult.

Additive: total_return_grid shape is GRID_N × GRID_N; base-cell value matches
a direct run_backtest of the same spec/dataset.
"""

from __future__ import annotations

import pytest

from pancake_engine import BacktestConfig, run_backtest, run_sensitivity_analysis
from pancake_engine.sensitivity import GRID_N

from ._runner_helpers import make_dataset, make_spec, row


DAY = 86_400


def _spec_and_dataset():
    spec = make_spec(side="YES", sizing_value=0.1, entry_when={"feature": "alpha", "gte": 2.5})
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=500, price=0.40, outcome=1, alpha=3.0, target=1),
        row(mkt="m/B", dec_ts=600, res_ts=1000, price=0.55, outcome=0, alpha=3.5, target=0),
        row(mkt="m/C", dec_ts=1100, res_ts=1500, price=0.30, outcome=1, alpha=2.6, target=1),
        row(mkt="m/D", dec_ts=1600, res_ts=2000, price=0.62, outcome=0, alpha=4.0, target=0),
        row(mkt="m/E", dec_ts=2100, res_ts=2500, price=0.25, outcome=1, alpha=2.8, target=1),
        row(mkt="m/F", dec_ts=2600, res_ts=3000, price=0.70, outcome=0, alpha=3.2, target=0),
    ])
    return spec, dataset


def test_total_return_grid_shape() -> None:
    """total_return_grid has the same 7×7 shape as sharpe_grid."""
    spec, dataset = _spec_and_dataset()
    res = run_sensitivity_analysis(spec, dataset, n_mc=50, mc_seed=0)

    assert hasattr(res, "total_return_grid"), "SensitivityResult missing total_return_grid"
    grid = res.total_return_grid
    assert len(grid) == GRID_N
    assert all(len(r) == GRID_N for r in grid), "Each row must have GRID_N entries"


def test_total_return_grid_none_only_when_no_trades() -> None:
    """total_return_grid cells are None only when no trades fired (zero-trade cell).
    total_return can be defined even when Sharpe is None (degenerate daily returns)."""
    spec, dataset = _spec_and_dataset()
    res = run_sensitivity_analysis(spec, dataset, n_mc=50, mc_seed=0)

    # A cell with a defined Sharpe must also have a defined total_return.
    for ri, (tr_row, sh_row) in enumerate(zip(res.total_return_grid, res.sharpe_grid)):
        for ci, (tr_val, sh_val) in enumerate(zip(tr_row, sh_row)):
            if sh_val is not None:
                assert tr_val is not None, (
                    f"cell [{ri}][{ci}]: sharpe defined but total_return is None"
                )


def test_base_cell_total_return_matches_direct_run() -> None:
    """Base cell total_return matches a direct run_backtest of the same spec."""
    spec, dataset = _spec_and_dataset()
    res = run_sensitivity_analysis(spec, dataset, n_mc=50, mc_seed=0)

    base_val = res.total_return_grid[res.base_entry_idx][res.base_sizing_idx]
    assert base_val is not None, "Base cell must not be None"

    direct = run_backtest(spec, dataset, with_inference=False)
    assert base_val == pytest.approx(direct.metrics.standard.total_return, rel=1e-9)


def test_total_return_grid_in_to_dict() -> None:
    """to_dict() includes total_return_grid."""
    spec, dataset = _spec_and_dataset()
    res = run_sensitivity_analysis(spec, dataset, n_mc=50, mc_seed=0)

    d = res.to_dict()
    assert "total_return_grid" in d
    assert d["total_return_grid"] == res.total_return_grid


def test_total_return_grid_deterministic() -> None:
    """Same seed → identical total_return_grid."""
    spec, dataset = _spec_and_dataset()
    a = run_sensitivity_analysis(spec, dataset, n_mc=50, mc_seed=42)
    b = run_sensitivity_analysis(spec, dataset, n_mc=50, mc_seed=42)
    assert a.total_return_grid == b.total_return_grid


def test_explicit_grid_total_return_shape() -> None:
    """Explicit entry/sizing grids → total_return_grid has matching shape."""
    spec, dataset = _spec_and_dataset()
    entries = [1.0, 2.0, 2.5, 3.0, 4.0]
    sizings = [0.05, 0.1, 0.2]
    res = run_sensitivity_analysis(
        spec, dataset,
        entry_thresholds=entries,
        sizing_fractions=sizings,
        n_mc=10,
    )
    assert len(res.total_return_grid) == len(entries)
    assert all(len(r) == len(sizings) for r in res.total_return_grid)
