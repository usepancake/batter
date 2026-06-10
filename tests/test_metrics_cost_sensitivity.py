"""Transaction-cost sensitivity + break-even multiplier."""

from __future__ import annotations

import pytest

from pancake_engine.metrics.cost_sensitivity import cost_sensitivity
from pancake_engine.runner.trade import Trade


def _mk(quote: float, settle: float, *, cost: float = 100.0, slip_bps: float = 50.0, fee_bps: float = 10.0) -> Trade:
    """A self-consistent Trade, built with the engine's own cost math (so k=1 round-trips)."""
    fill = quote * (1 + slip_bps / 10_000)
    fee = cost * fee_bps / 10_000
    shares = (cost - fee) / fill
    proceeds = shares * settle
    pnl = proceeds - cost
    return Trade(
        market_slug="m", outcome="YES", entry_t=0, entry_price=fill, entry_price_quote=quote,
        exit_t=1, exit_price=settle, exit_price_quote=settle, exit_reason="hold_to_resolution",
        shares=shares, cost=cost, proceeds=proceeds, pnl=pnl, return_pct=pnl / cost,
        days_held=1, resolved_outcome=None,
    )


def _by_mult(res):
    return {round(p.multiplier, 6): p for p in res.points}


def test_k1_reproduces_stored_trades() -> None:
    trades = [_mk(0.8, 1.0), _mk(0.8, 1.0), _mk(0.8, 0.0)]
    res = cost_sensitivity(trades)
    p1 = _by_mult(res)[1.0]
    assert p1.total_pnl == pytest.approx(sum(t.pnl for t in trades))
    assert p1.mean_return == pytest.approx(sum(t.return_pct for t in trades) / len(trades))
    assert p1.n_trades == 3


def test_zero_cost_dominates_net_cost() -> None:
    trades = [_mk(0.8, 1.0), _mk(0.8, 1.0), _mk(0.8, 0.0)]
    pts = _by_mult(cost_sensitivity(trades))
    assert pts[0.0].mean_return >= pts[1.0].mean_return >= pts[5.0].mean_return
    assert pts[0.0].total_pnl >= pts[5.0].total_pnl


def test_mean_return_monotone_non_increasing_in_cost() -> None:
    trades = [_mk(0.75, 1.0) for _ in range(4)] + [_mk(0.75, 0.0)]
    res = cost_sensitivity(trades, multipliers=(0.0, 0.5, 1.0, 2.0, 5.0, 10.0))
    means = [p.mean_return for p in res.points]
    assert all(means[i] >= means[i + 1] - 1e-12 for i in range(len(means) - 1))


def test_break_even_multiplier_is_a_root() -> None:
    # 5 moderate-edge winners (quote 0.8) + 1 loser → profitable at 1×, unprofitable at high cost.
    trades = [_mk(0.8, 1.0) for _ in range(5)] + [_mk(0.8, 0.0)]
    res = cost_sensitivity(trades)
    be = res.break_even_multiplier
    assert be is not None and 1.0 < be < 50.0
    # mean return at the break-even multiplier is ~0
    at_be = cost_sensitivity(trades, multipliers=(be,)).points[0].mean_return
    assert at_be == pytest.approx(0.0, abs=1e-6)


def test_unprofitable_gross_gives_break_even_zero() -> None:
    # net-negative even with zero costs (one small winner can't cover three losers)
    trades = [_mk(0.9, 1.0)] + [_mk(0.9, 0.0) for _ in range(3)]
    res = cost_sensitivity(trades)
    assert res.break_even_multiplier == 0.0


def test_always_profitable_gives_break_even_none() -> None:
    # deep-edge winners (quote 0.1), no losers → profitable even at 50× cost
    trades = [_mk(0.1, 1.0) for _ in range(4)]
    res = cost_sensitivity(trades)
    assert res.break_even_multiplier is None


def test_empty_trade_log() -> None:
    res = cost_sensitivity([])
    assert res.break_even_multiplier is None
    assert all(p.n_trades == 0 for p in res.points)


def test_smoke_over_real_backtest_trades() -> None:
    import sys
    sys.path.insert(0, "tests")
    from _runner_helpers import make_dataset, make_spec, row  # noqa: E402

    from pancake_engine import BacktestConfig, run_backtest

    spec = make_spec(side="YES", sizing_value=0.1, slip_bps=50, fee_bps=10, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.5, outcome=1, alpha=3, target=1),
        row(mkt="m/B", dec_ts=300, res_ts=400, price=0.6, outcome=0, alpha=3, target=1),
        row(mkt="m/C", dec_ts=500, res_ts=600, price=0.4, outcome=1, alpha=3, target=1),
    ])
    r = run_backtest(spec, dataset, BacktestConfig(observation_time=700))
    res = cost_sensitivity(r.trades)
    p1 = _by_mult(res)[1.0]
    # net pnl at the realised cost level == ending - starting
    assert p1.total_pnl == pytest.approx(
        r.metrics.standard.ending_capital - r.metrics.standard.starting_capital, rel=1e-9
    )
