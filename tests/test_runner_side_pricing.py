"""TS regression test: NO side at price 0.96 → ~104.166 shares → +4.17% pnl.

Ported verbatim from pancake-production/tests/evidence-runner/runner.test.ts L127-156.
The 0.96 NO-side case caught a hidden ``1 - entry_price`` inversion in an earlier
TS revision (total_return blew up to 2.84e+40). Engine 0.3 must NOT invert.
"""

from __future__ import annotations

from pancake_engine import BacktestConfig, run_backtest

from ._runner_helpers import make_dataset, make_spec, row


def test_no_side_at_0_96_books_correct_win() -> None:
    """NO at 0.96 with target != outcome (NO wins) → +4.17% pnl on a $100 sized trade."""
    spec = make_spec(
        side="NO",
        sizing_value=0.1,
        slip_bps=0,
        fee_bps=0,
        starting_capital=1000.0,
    )
    # NO wins when target != outcome (yes_payoff false → strategy_wins for NO)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.96, outcome=0, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=300)
    result = run_backtest(spec, dataset, config)

    assert result.validation.ok
    assert result.metrics.standard.num_trades == 1
    trade = result.trades[0]

    # entry_price stored as literal NO price (no inversion)
    assert trade.entry_price_quote == 0.96
    assert trade.entry_price == 0.96  # slip = 0
    # shares = $100 / 0.96 ≈ 104.1667
    assert abs(trade.shares - 100.0 / 0.96) < 1e-9
    # proceeds = shares × $1 (NO wins → settle_value = 1)
    assert trade.proceeds == trade.shares
    # pnl = proceeds − cost ≈ +4.1667
    assert abs(trade.pnl - (100.0 / 0.96 - 100.0)) < 1e-9
    # return_pct ≈ 4.17%
    assert abs(trade.return_pct - (1.0 / 0.96 - 1.0)) < 1e-9


def test_yes_side_at_0_20_books_correct_win() -> None:
    """YES at 0.20 wins → shares = 500, proceeds $500, pnl +$400 (+400% on $100)."""
    spec = make_spec(side="YES", sizing_value=0.1, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.20, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=300)
    result = run_backtest(spec, dataset, config)

    assert result.validation.ok
    assert result.metrics.standard.num_trades == 1
    trade = result.trades[0]
    assert trade.entry_price_quote == 0.20
    assert abs(trade.shares - 500.0) < 1e-9
    assert abs(trade.proceeds - 500.0) < 1e-9
    assert abs(trade.pnl - 400.0) < 1e-9
    assert abs(trade.return_pct - 4.0) < 1e-9


def test_yes_side_at_0_20_loses() -> None:
    """YES at 0.20 loses → proceeds 0, pnl -$100 (-100% on $100)."""
    spec = make_spec(side="YES", sizing_value=0.1, starting_capital=1000.0)
    dataset = make_dataset([
        # target != outcome → yes_payoff false → YES loses
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.20, outcome=0, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=300)
    result = run_backtest(spec, dataset, config)

    trade = result.trades[0]
    assert trade.proceeds == 0.0
    assert abs(trade.pnl + 100.0) < 1e-9
    assert abs(trade.return_pct + 1.0) < 1e-9


def test_breakeven_pnl_zero_is_not_a_win() -> None:
    """At entry_price post-slip exactly 1 − fee_frac, a winning trade has pnl = 0.

    Engine 0.3 uses STRICT pnl > 0 for win_rate. Breakeven is not a win.
    """
    # Engineer pnl = 0: cost = $100, fee = $0.50 (50bps), fill = entry × 1.005,
    # shares = 99.50 / fill, proceeds_on_win = shares × $1 = 99.50 / fill.
    # pnl_on_win = 99.50 / fill - 100. Set fill = 0.995 → shares = 100 → pnl = 0.
    # → entry = 0.995 / 1.005 ≈ 0.99005 (no slip needed; just set fee).
    # Simpler: slip=0, fee_bps=50, entry=0.995 → fill=0.995, investable=99.50,
    # shares=99.50/0.995=100, proceeds=100, pnl=100-100=0.
    spec = make_spec(side="YES", sizing_value=0.1, slip_bps=0, fee_bps=50, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.995, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=300)
    result = run_backtest(spec, dataset, config)

    trade = result.trades[0]
    assert abs(trade.pnl) < 1e-9
    # win_rate should be 0/1 = 0 (strict): pnl = 0 is not a win
    assert result.metrics.standard.win_rate == 0.0
