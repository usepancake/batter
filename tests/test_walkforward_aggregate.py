"""Aggregate metrics + dispersion warnings."""

from __future__ import annotations

from pancake_engine import BacktestConfig, WalkforwardConfig, run_walkforward
from pancake_engine.warnings import WarningCode

from ._runner_helpers import make_spec
from ._wf_helpers import make_wf_dataset


DAY = 86_400


def test_aggregate_pooled_metrics_combine_trades() -> None:
    """Pooled num_trades = sum of fold num_trades."""
    spec = make_spec(side="YES", sizing_value=0.05, starting_capital=10000.0)
    dataset = make_wf_dataset([(i * DAY, i * DAY + 100, {}) for i in range(90)])
    config = WalkforwardConfig(window_type="expanding", test_horizon=30 * DAY, step=30 * DAY)
    result = run_walkforward(spec, dataset, config, BacktestConfig(observation_time=200 * DAY))
    pooled = result.aggregate.pooled
    fold_sum = sum(f.result.metrics.standard.num_trades for f in result.folds)
    assert pooled.num_trades == fold_sum


def test_aggregate_fold_mean_total_return_is_mean() -> None:
    spec = make_spec(side="YES", sizing_value=0.05, starting_capital=10000.0)
    dataset = make_wf_dataset([(i * DAY, i * DAY + 100, {}) for i in range(90)])
    config = WalkforwardConfig(window_type="expanding", test_horizon=30 * DAY, step=30 * DAY)
    result = run_walkforward(spec, dataset, config, BacktestConfig(observation_time=200 * DAY))
    fold_returns = [f.result.metrics.standard.total_return for f in result.folds]
    mean_expected = sum(fold_returns) / len(fold_returns)
    assert abs(result.aggregate.fold_mean.total_return - mean_expected) < 1e-9


def test_walkforward_unequal_fold_size_warning() -> None:
    """One fold > 3× another → UNEQUAL_FOLD_SIZE."""
    spec = make_spec(side="YES", sizing_value=0.01, starting_capital=10000.0)
    dataset = make_wf_dataset(
        [(i * DAY, i * DAY + 100, {}) for i in range(30)]            # fold 0: 30 trades
        + [(31 * DAY + i * DAY, 31 * DAY + i * DAY + 100, {})
           for i in range(20)]                                          # fold 1: 20 trades
        + [(61 * DAY + i * DAY, 61 * DAY + i * DAY + 100, {})
           for i in range(2)]                                           # fold 2: 2 trades
    )
    config = WalkforwardConfig(window_type="expanding", test_horizon=30 * DAY, step=30 * DAY)
    result = run_walkforward(spec, dataset, config, BacktestConfig(observation_time=200 * DAY))
    assert any(w.code == WarningCode.UNEQUAL_FOLD_SIZE for w in result.warnings)


def test_walkforward_single_fold_carries_warning() -> None:
    """One fold > 70% of total |pnl| → SINGLE_FOLD_CARRIES."""
    spec = make_spec(side="YES", sizing_value=0.05, starting_capital=10000.0)
    # Engineer: 3 folds. Fold 0 has many big winning trades; folds 1+2 have a couple each.
    rows = []
    # Fold 0: many wins
    for i in range(25):
        rows.append((i * DAY, i * DAY + 100, {"price": 0.2, "outcome": 1, "target": 1}))
    # Fold 1: few trades
    for i in range(2):
        rows.append(((30 + i) * DAY, (30 + i) * DAY + 100, {"price": 0.5, "outcome": 1, "target": 1}))
    # Fold 2: few trades
    for i in range(2):
        rows.append(((60 + i) * DAY, (60 + i) * DAY + 100, {"price": 0.5, "outcome": 1, "target": 1}))
    dataset = make_wf_dataset(rows)
    config = WalkforwardConfig(window_type="expanding", test_horizon=30 * DAY, step=30 * DAY)
    result = run_walkforward(spec, dataset, config, BacktestConfig(observation_time=200 * DAY))
    assert any(w.code == WarningCode.WALKFORWARD_SINGLE_FOLD_CARRIES for w in result.warnings)


def test_walkforward_sign_flip_warning() -> None:
    """Fold with opposite-sign return vs pooled → WALKFORWARD_SIGN_FLIP."""
    spec = make_spec(side="YES", sizing_value=0.05, starting_capital=10000.0)
    # Fold 0: many wins (positive); Fold 1: many losses (negative); Fold 2: many wins (positive)
    rows = []
    for i in range(20):
        rows.append((i * DAY, i * DAY + 100, {"price": 0.2, "outcome": 1, "target": 1}))
    for i in range(15):
        rows.append(((30 + i) * DAY, (30 + i) * DAY + 100,
                     {"price": 0.99, "outcome": 0, "target": 1}))   # YES loses
    for i in range(20):
        rows.append(((60 + i) * DAY, (60 + i) * DAY + 100, {"price": 0.2, "outcome": 1, "target": 1}))
    dataset = make_wf_dataset(rows)
    config = WalkforwardConfig(window_type="expanding", test_horizon=30 * DAY, step=30 * DAY)
    result = run_walkforward(spec, dataset, config, BacktestConfig(observation_time=200 * DAY))
    assert any(w.code == WarningCode.WALKFORWARD_SIGN_FLIP for w in result.warnings)
