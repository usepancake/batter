"""Prediction-market metrics: brier_crowd computed, brier_strategy null, Wilson CI."""

from __future__ import annotations

import math

from pancake_engine import BacktestConfig, WarningCode, run_backtest
from pancake_engine.metrics.pm import wilson_ci95

from ._runner_helpers import make_dataset, make_spec, row


def test_metrics_pm_brier_crowd_computed() -> None:
    """brier_crowd is always computable when at least one trade exists.

    For YES trade at price 0.6 with outcome=1: (0.6-1)^2 = 0.16.
    """
    spec = make_spec(side="YES", sizing_value=0.1, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.6, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=300)
    result = run_backtest(spec, dataset, config)
    pm = result.metrics.pm
    assert pm.brier_crowd is not None
    assert abs(pm.brier_crowd - 0.16) < 1e-9


def test_metrics_pm_brier_strategy_null_in_pr1() -> None:
    """Rule-based spec → brier_strategy is None + BRIER_NOT_APPLICABLE warning."""
    spec = make_spec(side="YES", sizing_value=0.1, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.5, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=300)
    result = run_backtest(spec, dataset, config)
    assert result.metrics.pm.brier_strategy is None
    assert result.metrics.pm.brier_skill_score is None
    assert any(w.code == WarningCode.BRIER_NOT_APPLICABLE for w in result.warnings)


def test_metrics_pm_wilson_ci_null_for_zero_trades() -> None:
    """num_trades = 0 → Wilson CI bounds = None (architecture M-1)."""
    spec = make_spec(
        side="YES",
        starting_capital=1000.0,
        entry_when={"feature": "alpha", "gte": 999.0},  # never fires
    )
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.5, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=300)
    result = run_backtest(spec, dataset, config)
    assert result.metrics.pm.win_rate_ci95_low is None
    assert result.metrics.pm.win_rate_ci95_high is None
    assert any(w.code == WarningCode.NO_TRADES_NO_CI for w in result.warnings)


def test_wilson_ci_basic_values() -> None:
    """Wilson CI smoke values."""
    # n=0 → (None, None)
    assert wilson_ci95(0, 0) == (None, None)
    # n=100, p=0.5 → roughly (0.40, 0.60)
    low, high = wilson_ci95(50, 100)
    assert low is not None and high is not None
    assert abs(low - 0.4038) < 1e-3
    assert abs(high - 0.5962) < 1e-3
    # n=1, p=1 → upper bound clamped to 1.0
    low, high = wilson_ci95(1, 1)
    assert low is not None and high is not None
    assert 0.0 <= low <= 1.0
    assert high == 1.0
