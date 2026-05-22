"""Standard metrics: CAGR ruined case, zero-trade defaults."""

from __future__ import annotations

from pancake_engine import BacktestConfig, WarningCode, run_backtest

from ._runner_helpers import make_dataset, make_spec, row


def test_cagr_ruined_case() -> None:
    """Ending equity ≤ 0 → cagr = -1.0 and RUINED warning, no 0**fractional exception."""
    # Engineer total loss: bet 100% on a loser at high entry price (so we lose all)
    # sizing.value = 1.0, price 0.99, outcome means YES loses → all in, lose everything
    spec = make_spec(side="YES", sizing_value=1.0, starting_capital=1000.0)
    dataset = make_dataset([
        # YES at 0.99, target=1, outcome=0 → yes_payoff=false → YES loses → proceeds=0
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.99, outcome=0, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=300)
    result = run_backtest(spec, dataset, config)

    assert result.validation.ok
    assert result.metrics.standard.ending_capital == 0.0
    assert result.metrics.standard.cagr == -1.0
    assert any(w.code == WarningCode.RUINED for w in result.warnings)


def test_zero_trades_metrics_defaults() -> None:
    """Strategy that never fires → total_return=0, cagr=0, sharpe=None, sortino=None,
    max_drawdown=0, win_rate=None, NO_TRADES_GENERATED + NO_TRADES_NO_CI warnings."""
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

    s = result.metrics.standard
    assert s.num_trades == 0
    assert s.total_return == 0.0
    assert s.cagr == 0.0
    assert s.sharpe is None
    assert s.sortino is None
    assert s.max_drawdown == 0.0
    assert s.win_rate is None

    codes = {w.code for w in result.warnings}
    assert WarningCode.NO_TRADES_GENERATED in codes
    assert WarningCode.NO_TRADES_NO_CI in codes
