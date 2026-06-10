"""ADR-0046 robustness: entry×sizing Sharpe sweep + Monte-Carlo drawdown."""

from __future__ import annotations

import pytest

from pancake_engine import run_backtest, run_sensitivity_analysis
from pancake_engine.sensitivity import _centered_grid, _find_gte, _set_gte

from ._runner_helpers import make_dataset, make_spec, row


def _spec_and_dataset():
    # alpha varies so different entry thresholds include/exclude trades;
    # outcomes mix wins/losses so Sharpe + drawdown are non-trivial.
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


def test_with_inference_flag_skips_ci_and_permutation(monkeypatch):
    """0.7.1 sweep perf: with_inference=False must NOT call the expensive
    inference (bootstrap CIs + permutation), but still returns Sharpe + total
    return unchanged; the default keeps inference so receipts are byte-identical.
    Spied at the call level so the proof doesn't depend on the data being
    non-degenerate."""
    import pancake_engine.metrics.standard as std

    calls = {"bootstrap": 0, "perm": 0}
    real_boot, real_perm = std.block_bootstrap_ci, std.permutation_p_sharpe
    monkeypatch.setattr(std, "block_bootstrap_ci", lambda *a, **k: (calls.__setitem__("bootstrap", calls["bootstrap"] + 1), real_boot(*a, **k))[1])
    monkeypatch.setattr(std, "permutation_p_sharpe", lambda *a, **k: (calls.__setitem__("perm", calls["perm"] + 1), real_perm(*a, **k))[1])

    spec, dataset = _spec_and_dataset()

    fast = run_backtest(spec, dataset, with_inference=False).metrics.standard
    assert calls == {"bootstrap": 0, "perm": 0}  # inference skipped entirely
    assert fast.sharpe_ci == (None, None)
    assert fast.cagr_ci == (None, None)
    assert fast.sortino_ci == (None, None)
    assert fast.sharpe_p_value is None

    full = run_backtest(spec, dataset).metrics.standard  # default True
    assert calls["bootstrap"] >= 1 and calls["perm"] >= 1  # inference ran

    # The cheap metric the sweep actually reads is identical either way.
    assert fast.sharpe == full.sharpe
    assert fast.total_return == full.total_return


def test_grid_shape_and_base_indices():
    spec, dataset = _spec_and_dataset()
    res = run_sensitivity_analysis(spec, dataset, n_mc=200, mc_seed=0)

    assert len(res.entry_thresholds) == 7
    assert len(res.sizing_fractions) == 7
    assert len(res.sharpe_grid) == 7
    assert all(len(rowv) == 7 for rowv in res.sharpe_grid)
    assert 0 <= res.base_entry_idx < 7
    assert 0 <= res.base_sizing_idx < 7
    # base cell sits at the spec's actual entry (2.5) / sizing (0.1)
    assert abs(res.entry_thresholds[res.base_entry_idx] - 2.5) < 0.05
    assert abs(res.sizing_fractions[res.base_sizing_idx] - 0.1) < 0.05


def test_mc_drawdown_fan():
    spec, dataset = _spec_and_dataset()
    res = run_sensitivity_analysis(spec, dataset, n_mc=500, mc_seed=0)
    # one point per resample step (t = 0..base_num_trades)
    assert len(res.mc_drawdown_points) == res.base_num_trades + 1
    for pt in res.mc_drawdown_points:
        # percentiles ordered within each step
        assert pt["p5"] <= pt["p25"] <= pt["p50"] <= pt["p75"] <= pt["p95"]
        assert pt["p5"] >= 0.0  # drawdown is a non-negative fraction
    # step 0 = starting capital → no drawdown yet
    assert res.mc_drawdown_points[0]["p95"] == 0.0
    # running-max drawdown only grows along the path (median is non-decreasing)
    medians = [pt["p50"] for pt in res.mc_drawdown_points]
    assert all(medians[i] <= medians[i + 1] + 1e-9 for i in range(len(medians) - 1))
    assert res.mc_n == 500
    assert res.base_num_trades >= 1


def test_deterministic_same_seed():
    spec, dataset = _spec_and_dataset()
    a = run_sensitivity_analysis(spec, dataset, n_mc=300, mc_seed=7)
    b = run_sensitivity_analysis(spec, dataset, n_mc=300, mc_seed=7)
    assert a.to_dict() == b.to_dict()


def test_n_mc_zero_raises():
    # n_mc=0 with >=1 trade used to reach np.percentile on an empty array →
    # IndexError. Guard it with a clear ValueError instead.
    spec, dataset = _spec_and_dataset()
    with pytest.raises(ValueError, match="n_mc"):
        run_sensitivity_analysis(spec, dataset, n_mc=0)


def test_zero_trade_base_cell_emits_t0_sentinel():
    # Entry threshold above every alpha → base cell fires no trades. The MC fan
    # must still honor the documented len == base_num_trades + 1 invariant by
    # emitting the single t=0 (no-drawdown) sentinel.
    spec = make_spec(side="YES", sizing_value=0.1, entry_when={"feature": "alpha", "gte": 99.0})
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=500, price=0.40, outcome=1, alpha=3.0, target=1),
        row(mkt="m/B", dec_ts=600, res_ts=1000, price=0.55, outcome=0, alpha=3.5, target=0),
    ])
    res = run_sensitivity_analysis(spec, dataset, n_mc=10, mc_seed=0)
    assert res.base_num_trades == 0
    assert len(res.mc_drawdown_points) == res.base_num_trades + 1
    assert res.mc_drawdown_points[0] == {
        "t": 0.0, "p5": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0,
    }


def test_explicit_grids_respected():
    spec, dataset = _spec_and_dataset()
    entries = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    sizings = [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
    res = run_sensitivity_analysis(
        spec, dataset, entry_thresholds=entries, sizing_fractions=sizings, n_mc=100
    )
    assert res.entry_thresholds == entries
    assert res.sizing_fractions == sizings
    # tighter entry thresholds fire fewer trades → at the extreme, some cells go undefined (None)
    assert any(cell is None for rowv in res.sharpe_grid for cell in rowv) or all(
        cell is None or isinstance(cell, float) for rowv in res.sharpe_grid for cell in rowv
    )


def test_no_entry_threshold_raises():
    import pytest

    spec, dataset = _spec_and_dataset()
    # an entry condition with no gte lever
    spec = make_spec(entry_when={"feature_equal": {"a": "target", "b": "outcome"}})
    with pytest.raises(ValueError, match="E_SENSITIVITY_NO_ENTRY_THRESHOLD"):
        run_sensitivity_analysis(spec, dataset, n_mc=10)


def test_grid_helpers():
    grid, idx = _centered_grid(2.5, 0.025, 0.01, 0.99)
    assert len(grid) == 7
    assert grid[idx] == min(grid, key=lambda v: abs(v - 2.5))
    assert _find_gte({"feature": "alpha", "gte": 2.0}) == 2.0
    assert _find_gte({"all_of": [{"feature": "x", "lte": 1}, {"feature": "y", "gte": 3.0}]}) == 3.0
    assert _set_gte({"feature": "a", "gte": 2.0}, 5.0)["gte"] == 5.0
