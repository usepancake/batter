"""Walk-forward idempotency: same inputs → same aggregate_result_hash."""

from __future__ import annotations

from pancake_engine import BacktestConfig, WalkforwardConfig, run_walkforward

from ._runner_helpers import make_spec
from ._wf_helpers import make_wf_dataset


DAY = 86_400


def test_walkforward_aggregate_hash_byte_equal_across_reruns() -> None:
    spec = make_spec(side="YES", sizing_value=0.05, starting_capital=10000.0)
    dataset = make_wf_dataset([(i * DAY, i * DAY + 100, {}) for i in range(90)])
    config = WalkforwardConfig(window_type="expanding", test_horizon=30 * DAY, step=30 * DAY)
    backtest_config = BacktestConfig(observation_time=200 * DAY)
    r1 = run_walkforward(spec, dataset, config, backtest_config)
    r2 = run_walkforward(spec, dataset, config, backtest_config)
    r3 = run_walkforward(spec, dataset, config, backtest_config)
    assert r1.aggregate_result_hash == r2.aggregate_result_hash == r3.aggregate_result_hash
    assert r1.aggregate_result_hash != ""


def test_walkforward_per_fold_result_hash_deterministic() -> None:
    spec = make_spec(side="YES", sizing_value=0.05, starting_capital=10000.0)
    dataset = make_wf_dataset([(i * DAY, i * DAY + 100, {}) for i in range(90)])
    config = WalkforwardConfig(window_type="expanding", test_horizon=30 * DAY, step=30 * DAY)
    backtest_config = BacktestConfig(observation_time=200 * DAY)
    r1 = run_walkforward(spec, dataset, config, backtest_config)
    r2 = run_walkforward(spec, dataset, config, backtest_config)
    for f1, f2 in zip(r1.folds, r2.folds):
        assert f1.result.result_hash == f2.result.result_hash


def test_walkforward_aggregate_hash_does_not_collide_with_backtest_hash() -> None:
    """aggregate_result_hash includes 'walkforward_version' + 'result_kind' so it cannot
    collide with a plain BacktestResult.result_hash."""
    from pancake_engine import run_backtest
    spec = make_spec(side="YES", sizing_value=0.05, starting_capital=10000.0)
    # Single-fold-style WF
    dataset = make_wf_dataset([(i * DAY, i * DAY + 100, {}) for i in range(60)])
    bt_config = BacktestConfig(observation_time=200 * DAY)
    bt_result = run_backtest(spec, dataset, bt_config)

    config = WalkforwardConfig(
        window_type="expanding", test_horizon=30 * DAY, step=30 * DAY, min_fold_count=2,
    )
    wf_result = run_walkforward(spec, dataset, config, bt_config)
    # Hash domains separated by walkforward_version + result_kind in payload
    assert wf_result.aggregate_result_hash != bt_result.result_hash


def test_walkforward_different_step_different_hash() -> None:
    spec = make_spec(side="YES", sizing_value=0.05, starting_capital=10000.0)
    dataset = make_wf_dataset([(i * DAY, i * DAY + 100, {}) for i in range(90)])
    backtest_config = BacktestConfig(observation_time=200 * DAY)
    r1 = run_walkforward(spec, dataset,
                          WalkforwardConfig(window_type="expanding", test_horizon=30 * DAY, step=30 * DAY),
                          backtest_config)
    r2 = run_walkforward(spec, dataset,
                          WalkforwardConfig(window_type="expanding", test_horizon=30 * DAY, step=15 * DAY),
                          backtest_config)
    assert r1.aggregate_result_hash != r2.aggregate_result_hash
