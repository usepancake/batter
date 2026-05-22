"""WF runner: per-fold backtest + schedule wiring."""

from __future__ import annotations

import pytest

from pancake_engine import (
    BacktestConfig,
    WalkforwardConfig,
    run_walkforward,
)
from pancake_engine.warnings import WarningCode

from ._runner_helpers import make_spec
from ._wf_helpers import make_wf_dataset


DAY = 86_400


def _ds_three_folds():
    """90-row dataset spanning 90 days, evenly spaced."""
    return make_wf_dataset([
        (i * DAY, i * DAY + 100, {})
        for i in range(90)
    ])


def test_walkforward_basic_three_fold_expanding() -> None:
    spec = make_spec(side="YES", sizing_value=0.05, starting_capital=10000.0)
    dataset = _ds_three_folds()
    config = WalkforwardConfig(
        window_type="expanding", test_horizon=30 * DAY, step=30 * DAY,
    )
    backtest_config = BacktestConfig(observation_time=200 * DAY)
    result = run_walkforward(spec, dataset, config, backtest_config)
    assert result.validation.ok, [e.code for e in result.validation.errors]
    assert len(result.folds) == 3
    for fold in result.folds:
        # Each fold gets 30 trades since rows are 1-per-day
        assert fold.result.metrics.standard.num_trades == 30


def test_walkforward_truncate_policy_raises() -> None:
    spec = make_spec()
    dataset = _ds_three_folds()
    config = WalkforwardConfig(
        window_type="expanding", test_horizon=30 * DAY, step=30 * DAY,
        resolution_policy="truncate_at_window_end",
    )
    with pytest.raises(NotImplementedError, match="E_OVERHANG_TRUNCATION_UNSUPPORTED"):
        run_walkforward(spec, dataset, config)


def test_walkforward_skip_overhang_drops_overhanging_rows() -> None:
    """Rows whose resolution_time > test_end are dropped under skip_overhang."""
    spec = make_spec(side="YES", sizing_value=0.05, starting_capital=10000.0)
    # Build dataset spanning 90+ days so default min_fold_count=3 is satisfied.
    dataset = make_wf_dataset([
        (1 * DAY, 5 * DAY, {}),       # fold 0: decision + resolution inside fold 0
        (10 * DAY, 35 * DAY, {}),     # fold 0 decision, resolution overhangs into fold 1
        (40 * DAY, 50 * DAY, {}),     # fold 1: clean
        (70 * DAY, 80 * DAY, {}),     # fold 2: clean
    ])
    config = WalkforwardConfig(
        window_type="expanding", test_horizon=30 * DAY, step=30 * DAY,
        resolution_policy="skip_overhang",
    )
    backtest_config = BacktestConfig(observation_time=200 * DAY)
    result = run_walkforward(spec, dataset, config, backtest_config)
    # Fold 0 had 2 decisions; one overhangs → 1 trade
    fold0_trades = result.folds[0].result.metrics.standard.num_trades
    fold1_trades = result.folds[1].result.metrics.standard.num_trades
    fold2_trades = result.folds[2].result.metrics.standard.num_trades
    assert fold0_trades == 1
    assert fold1_trades == 1
    assert fold2_trades == 1
    assert any(w.code == WarningCode.OVERHANG_SKIPPED for w in result.warnings)


def test_walkforward_allow_overhang_keeps_all() -> None:
    """Under allow_overhang (default), no rows are skipped."""
    spec = make_spec(side="YES", sizing_value=0.05, starting_capital=10000.0)
    dataset = make_wf_dataset([
        (1 * DAY, 5 * DAY, {}),
        (10 * DAY, 35 * DAY, {}),
        (40 * DAY, 50 * DAY, {}),
        (70 * DAY, 80 * DAY, {}),
    ])
    config = WalkforwardConfig(
        window_type="expanding", test_horizon=30 * DAY, step=30 * DAY,
        resolution_policy="allow_overhang",
    )
    backtest_config = BacktestConfig(observation_time=200 * DAY)
    result = run_walkforward(spec, dataset, config, backtest_config)
    assert not any(w.code == WarningCode.OVERHANG_SKIPPED for w in result.warnings)
    assert result.folds[0].result.metrics.standard.num_trades == 2  # both fold-0 decisions
    assert result.folds[1].result.metrics.standard.num_trades == 1
    assert result.folds[2].result.metrics.standard.num_trades == 1


def test_walkforward_empty_fold_warning() -> None:
    """A fold with no decisions in its test_window emits EMPTY_FOLD."""
    spec = make_spec(side="YES", sizing_value=0.05, starting_capital=10000.0)
    # 3 folds of 30 days over a 90-day dataset, but rows ONLY in fold 0.
    # Need at least one row in the last day so schedule produces 3 folds.
    dataset = make_wf_dataset(
        [(i * DAY, i * DAY + 100, {}) for i in range(15)]    # fold 0: 15 rows
        + [(89 * DAY, 89 * DAY + 100, {})]                    # fold 2: 1 row (force span)
    )
    config = WalkforwardConfig(
        window_type="expanding", test_horizon=30 * DAY, step=30 * DAY,
    )
    backtest_config = BacktestConfig(observation_time=200 * DAY)
    result = run_walkforward(spec, dataset, config, backtest_config)
    # Fold 1 (days 30-60) has no decisions → EMPTY_FOLD
    empty_folds = [f for f in result.folds if any(
        w.code == WarningCode.EMPTY_FOLD for w in f.result.warnings
    )]
    assert len(empty_folds) >= 1


def test_walkforward_low_trades_in_fold_warning() -> None:
    """Fold with 1-9 trades emits LOW_TRADES_IN_FOLD."""
    spec = make_spec(side="YES", sizing_value=0.01, starting_capital=10000.0)
    dataset = make_wf_dataset(
        [(i * DAY, i * DAY + 100, {}) for i in range(30)]                         # fold 0: 30
        + [(31 * DAY + i * DAY, 31 * DAY + i * DAY + 100, {}) for i in range(5)]  # fold 1: 5
        + [(61 * DAY + i * DAY, 61 * DAY + i * DAY + 100, {}) for i in range(20)] # fold 2: 20
    )
    config = WalkforwardConfig(
        window_type="expanding", test_horizon=30 * DAY, step=30 * DAY,
    )
    backtest_config = BacktestConfig(observation_time=200 * DAY)
    result = run_walkforward(spec, dataset, config, backtest_config)
    # Fold 1 should have LOW_TRADES_IN_FOLD
    low_fold = [f for f in result.folds if any(
        w.code == WarningCode.LOW_TRADES_IN_FOLD for w in f.result.warnings
    )]
    assert len(low_fold) >= 1


def test_walkforward_insufficient_folds_errors() -> None:
    """Dataset span too short to produce min_fold_count folds → error."""
    spec = make_spec()
    dataset = make_wf_dataset([(i * DAY, i * DAY + 100, {}) for i in range(10)])
    # 10 days; can't make 3 folds of 30 days
    config = WalkforwardConfig(
        window_type="expanding", test_horizon=30 * DAY, step=30 * DAY,
        min_fold_count=3,
    )
    result = run_walkforward(spec, dataset, config, BacktestConfig(observation_time=100 * DAY))
    assert not result.validation.ok
    codes = {e.code for e in result.validation.errors}
    assert "E_WALKFORWARD_INSUFFICIENT_FOLDS" in codes


def test_walkforward_two_folds_override_warning() -> None:
    """min_fold_count=2 with exactly 2 folds → OVERRIDE_MIN_FOLD_COUNT info."""
    spec = make_spec(side="YES", sizing_value=0.05, starting_capital=10000.0)
    # 60 days, 30-day folds → 2 folds
    dataset = make_wf_dataset([(i * DAY, i * DAY + 100, {}) for i in range(60)])
    config = WalkforwardConfig(
        window_type="expanding", test_horizon=30 * DAY, step=30 * DAY,
        min_fold_count=2,
    )
    result = run_walkforward(spec, dataset, config, BacktestConfig(observation_time=100 * DAY))
    assert result.validation.ok
    assert len(result.folds) == 2
    assert any(w.code == WarningCode.OVERRIDE_MIN_FOLD_COUNT for w in result.warnings)


def test_walkforward_min_fold_count_below_two_errors() -> None:
    spec = make_spec()
    dataset = make_wf_dataset([(i * DAY, i * DAY + 100, {}) for i in range(90)])
    config = WalkforwardConfig(
        window_type="expanding", test_horizon=30 * DAY, step=30 * DAY,
        min_fold_count=1,
    )
    result = run_walkforward(spec, dataset, config)
    assert not result.validation.ok
    codes = {e.code for e in result.validation.errors}
    assert "E_WALKFORWARD_CONFIG_INVALID" in codes


def test_walkforward_feature_lookahead_unchecked_warning() -> None:
    """Dataset without provenance flag → FEATURE_LOOKAHEAD_UNCHECKED info."""
    spec = make_spec(side="YES", sizing_value=0.05, starting_capital=10000.0)
    dataset = make_wf_dataset([(i * DAY, i * DAY + 100, {}) for i in range(90)])
    config = WalkforwardConfig(
        window_type="expanding", test_horizon=30 * DAY, step=30 * DAY,
    )
    result = run_walkforward(spec, dataset, config, BacktestConfig(observation_time=200 * DAY))
    assert any(w.code == WarningCode.FEATURE_LOOKAHEAD_UNCHECKED for w in result.warnings)


def test_walkforward_feature_lookahead_verified_suppresses_warning() -> None:
    spec = make_spec(side="YES", sizing_value=0.05, starting_capital=10000.0)
    dataset = make_wf_dataset(
        [(i * DAY, i * DAY + 100, {}) for i in range(90)],
        provenance={"feature_construction_verified_no_lookahead": True},
    )
    config = WalkforwardConfig(
        window_type="expanding", test_horizon=30 * DAY, step=30 * DAY,
    )
    result = run_walkforward(spec, dataset, config, BacktestConfig(observation_time=200 * DAY))
    assert not any(w.code == WarningCode.FEATURE_LOOKAHEAD_UNCHECKED for w in result.warnings)
