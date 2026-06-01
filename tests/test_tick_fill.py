"""SimFillRouter (rule 145) — fill math + parity with the backtest fill."""

from __future__ import annotations

import math

from pancake_engine import run_backtest
from pancake_engine.runner.fill import Fill, FillRejection, SimFillRouter

from ._runner_helpers import make_dataset, make_spec, row


def test_basic_yes_fill_no_costs() -> None:
    r = SimFillRouter(slippage_bps=0.0, fee_bps=0.0)
    f = r.fill(side="YES", yes_close=0.4, available_cash=1000.0, sizing_value=0.1)
    assert isinstance(f, Fill)
    assert f.quote_price == 0.4
    assert f.fill_price == 0.4
    assert f.cost == 100.0
    assert f.fee == 0.0
    assert math.isclose(f.shares, 250.0)


def test_no_side_quote_is_one_minus_close() -> None:
    r = SimFillRouter(slippage_bps=0.0, fee_bps=0.0)
    f = r.fill(side="NO", yes_close=0.4, available_cash=1000.0, sizing_value=0.1)
    assert isinstance(f, Fill)
    assert math.isclose(f.quote_price, 0.6)
    assert math.isclose(f.fill_price, 0.6)
    assert math.isclose(f.shares, 100.0 / 0.6)


def test_slippage_raises_fill_price() -> None:
    r = SimFillRouter(slippage_bps=100.0, fee_bps=0.0)  # 1%
    f = r.fill(side="YES", yes_close=0.4, available_cash=1000.0, sizing_value=0.1)
    assert isinstance(f, Fill)
    assert math.isclose(f.fill_price, 0.4 * 1.01)
    assert math.isclose(f.shares, 100.0 / (0.4 * 1.01))


def test_fee_reduces_investable() -> None:
    r = SimFillRouter(slippage_bps=0.0, fee_bps=100.0)  # 1%
    f = r.fill(side="YES", yes_close=0.4, available_cash=1000.0, sizing_value=0.1)
    assert isinstance(f, Fill)
    assert math.isclose(f.fee, 1.0)
    assert math.isclose(f.shares, 99.0 / 0.4)
    assert f.cost == 100.0  # cost is the full notional (incl. fee)


def test_sizing_zero_rejects() -> None:
    r = SimFillRouter(slippage_bps=0.0, fee_bps=0.0)
    f = r.fill(side="YES", yes_close=0.4, available_cash=0.0, sizing_value=0.1)
    assert isinstance(f, FillRejection)
    assert f.reason == "sizing_zero"


def test_quote_out_of_range_rejects() -> None:
    r = SimFillRouter(slippage_bps=0.0, fee_bps=0.0)

    def f(side: str, yes_close: float) -> Fill | FillRejection:
        return r.fill(side=side, yes_close=yes_close, available_cash=1000.0, sizing_value=0.1)

    assert isinstance(f("YES", 0.0), FillRejection)
    assert isinstance(f("YES", 1.0), FillRejection)
    assert isinstance(f("NO", 1.0), FillRejection)  # quote 1 - 1 = 0.0 → rejected


def test_fill_price_pushed_out_of_range_rejects() -> None:
    r = SimFillRouter(slippage_bps=5000.0, fee_bps=0.0)  # +50% slip
    f = r.fill(side="YES", yes_close=0.8, available_cash=1000.0, sizing_value=0.1)  # 0.8*1.5 = 1.2
    assert isinstance(f, FillRejection)
    assert f.reason == "fill_price_out_of_range"


def test_parity_with_backtest_fill() -> None:
    """SimFillRouter reproduces the backtest's open-fill (same rule-145 math)."""
    spec = make_spec(
        side="YES", sizing_value=0.1, slip_bps=50.0, fee_bps=25.0, starting_capital=1000.0
    )
    ds = make_dataset([row(mkt="m", dec_ts=100, res_ts=200, price=0.4, outcome=1, alpha=3.0)])
    result = run_backtest(spec, ds)
    assert len(result.trades) == 1
    trade = result.trades[0]

    r = SimFillRouter(slippage_bps=50.0, fee_bps=25.0)
    f = r.fill(side="YES", yes_close=0.4, available_cash=1000.0, sizing_value=0.1)
    assert isinstance(f, Fill)
    assert math.isclose(f.fill_price, trade.entry_price, rel_tol=1e-12)
    assert math.isclose(f.shares, trade.shares, rel_tol=1e-12)
    assert math.isclose(f.cost, trade.cost, rel_tol=1e-12)
