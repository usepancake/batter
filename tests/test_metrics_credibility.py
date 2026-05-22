"""Credibility warnings: IMPLAUSIBLY_HIGH_SHARPE, LOW_SAMPLE_SIZE, MICRO_SAMPLE_SIZE,
MARK_AT_COST_DRAWDOWN_MUTED, SINGLE_MARKET_RESULT.
"""

from __future__ import annotations

from pancake_engine import BacktestConfig, WarningCode, run_backtest

from ._runner_helpers import make_dataset, make_spec, row


def test_low_sample_size_warning_fires_at_n_under_30() -> None:
    spec = make_spec(side="YES", sizing_value=0.01, starting_capital=10000.0)
    dataset = make_dataset([
        row(mkt=f"m/{i}", dec_ts=100 + i * 1000, res_ts=500 + i * 1000,
            price=0.5, outcome=1, alpha=3.0, target=1)
        for i in range(12)
    ])
    config = BacktestConfig(observation_time=200_000)
    result = run_backtest(spec, dataset, config)
    codes = {w.code for w in result.warnings}
    assert WarningCode.LOW_SAMPLE_SIZE in codes or WarningCode.MICRO_SAMPLE_SIZE in codes


def test_micro_sample_size_warning_fires_at_n_under_10() -> None:
    spec = make_spec(side="YES", sizing_value=0.01, starting_capital=10000.0)
    dataset = make_dataset([
        row(mkt=f"m/{i}", dec_ts=100 + i * 1000, res_ts=500 + i * 1000,
            price=0.5, outcome=1, alpha=3.0, target=1)
        for i in range(5)
    ])
    config = BacktestConfig(observation_time=200_000)
    result = run_backtest(spec, dataset, config)
    codes = {w.code for w in result.warnings}
    assert WarningCode.MICRO_SAMPLE_SIZE in codes


def test_single_market_result_warning_fires() -> None:
    """All trades on same market_link → SINGLE_MARKET_RESULT."""
    spec = make_spec(side="YES", sizing_value=0.05, starting_capital=10000.0)
    dataset = make_dataset([
        row(mkt="m/SAME", dec_ts=100, res_ts=200, price=0.5, outcome=1, alpha=3.0, target=1),
        row(mkt="m/SAME", dec_ts=300, res_ts=400, price=0.5, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=500)
    result = run_backtest(spec, dataset, config)
    assert any(w.code == WarningCode.SINGLE_MARKET_RESULT for w in result.warnings)


def test_mark_at_cost_drawdown_muted_warning_fires() -> None:
    """All winning trades under mark_at_cost → max_drawdown < 1% AND n > 10 → muted warning."""
    spec = make_spec(side="YES", sizing_value=0.01, starting_capital=100_000.0)
    dataset = make_dataset([
        row(mkt=f"m/{i}", dec_ts=100 + i * 1000, res_ts=500 + i * 1000,
            price=0.5, outcome=1, alpha=3.0, target=1)
        for i in range(15)
    ])
    config = BacktestConfig(observation_time=200_000)
    result = run_backtest(spec, dataset, config)
    # All wins, no losses → max_drawdown ≈ 0
    assert result.metrics.standard.max_drawdown < 0.01
    assert any(w.code == WarningCode.MARK_AT_COST_DRAWDOWN_MUTED for w in result.warnings)
