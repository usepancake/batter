"""Mark-at-cost fee realization (B-1 of math bash).

Under mark_at_cost: mark_value = shares × entry_fill_price = notional − fee.
Equity drops by ``fee`` at the decision-time event point — fees realized at entry,
not at resolution.
"""

from __future__ import annotations

from pancake_engine import BacktestConfig, run_backtest

from ._runner_helpers import make_dataset, make_spec, row


def test_mark_at_cost_realizes_fee_at_entry() -> None:
    """$1000 starting / $100 sized / 50 bps fee → equity = $999.50 at decision-time event.

    Math: notional = 100, fee = 100 × 0.005 = 0.50, investable = 99.50,
    shares = 99.50 / fill_price, mark_value (mark_at_cost) = shares × entry_price = 99.50.
    cash = 1000 - 100 = 900. equity = cash + mark_value = 900 + 99.50 = 999.50.
    """
    spec = make_spec(
        side="YES", sizing_value=0.1, slip_bps=0, fee_bps=50, starting_capital=1000.0,
    )
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=500, price=0.5, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=1000)
    result = run_backtest(spec, dataset, config)

    assert result.validation.ok
    # The first equity_curve sample is at the decision-time event (T=100)
    eq_at_decision = next(p for p in result.equity_curve if p.t == 100)
    assert abs(eq_at_decision.equity - 999.50) < 1e-9, (
        f"expected $999.50 at decision (fee realized at entry); got {eq_at_decision.equity}"
    )


def test_zero_fee_equity_constant_until_resolution() -> None:
    """With fee = 0: equity at decision-time event == starting_capital (no fee to realize)."""
    spec = make_spec(
        side="YES", sizing_value=0.1, slip_bps=0, fee_bps=0, starting_capital=1000.0,
    )
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=500, price=0.5, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=1000)
    result = run_backtest(spec, dataset, config)
    eq_at_decision = next(p for p in result.equity_curve if p.t == 100)
    assert abs(eq_at_decision.equity - 1000.0) < 1e-9


def test_mark_at_cost_two_open_fees_compound() -> None:
    """Two open positions with fees: equity at decision-2-time == starting − fee1 − fee2."""
    spec = make_spec(
        side="YES", sizing_value=0.1, slip_bps=0, fee_bps=50, starting_capital=1000.0,
    )
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=500, price=0.5, outcome=1, alpha=3.0, target=1),
        row(mkt="m/B", dec_ts=200, res_ts=600, price=0.5, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=1000)
    result = run_backtest(spec, dataset, config)

    # At T=100: cost=100, fee=0.50, equity = 999.50
    # At T=200: cost = 0.1 × cash($900) = 90, fee = 0.45, equity = 999.50 - 0.45 = 999.05
    eq_at_200 = next(p for p in result.equity_curve if p.t == 200)
    assert abs(eq_at_200.equity - 999.05) < 1e-9, (
        f"expected $999.05 at second decision (compounded fees); got {eq_at_200.equity}"
    )
