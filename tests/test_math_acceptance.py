"""Engine 0.3 math acceptance suite.

Hand-calculated fixtures that prove the calculation machine behaves correctly
on core and edge cases. Each case ships with its arithmetic in a comment block;
the test asserts trades, ending_capital, total_return, cagr, and warnings.

Every test also asserts ``result_hash`` stability across two reruns —
nondeterminism would surface here even if individual values matched.

Engine 0.3 is correctness-first, not TS parity. Known TS divergences live in
pancake-production/docs/research/pancake-engine-0.3-ts-divergences.md.

Doctrine reminders:
- ``yes_payoff`` is true when ``target == outcome`` (the default condition).
- For ``side = YES``: strategy_wins = yes_payoff   → wins when target == outcome.
- For ``side = NO`` : strategy_wins = !yes_payoff  → wins when target != outcome.
- ``mark_at_cost``: mark_value = shares × entry_fill_price = notional − fee.
- Fees realized at entry (B-1 of math bash).
- ``win_rate`` uses strict ``pnl > 0``.
"""

from __future__ import annotations

import pytest

from pancake_engine import BacktestConfig, run_backtest
from pancake_engine.warnings import WarningCode

from ._runner_helpers import make_dataset, make_spec, row

EPS = 1e-9
DAY = 86_400


# -----------------------------------------------------------------------------
# Helper: assert result_hash is stable across an independent rerun.
# -----------------------------------------------------------------------------


def _assert_hash_stable(spec, dataset, config) -> str:
    r1 = run_backtest(spec, dataset, config)
    r2 = run_backtest(spec, dataset, config)
    assert r1.result_hash != "", "empty result_hash"
    assert r1.result_hash == r2.result_hash, (
        f"non-deterministic hash: {r1.result_hash} != {r2.result_hash}"
    )
    return r1.result_hash


# -----------------------------------------------------------------------------
# Case 1 — YES @ 0.50 wins, $100 notional → proceeds $200, pnl +$100
# -----------------------------------------------------------------------------
#
# Hand calculation:
#   starting_capital = 1000
#   sizing.value     = 0.1  → notional = 0.1 × 1000 = 100
#   side             = YES
#   price            = 0.5, no slip, no fee
#   shares           = 100 / 0.5                    = 200
#   target=1, outcome=1 → yes_payoff = True → YES wins → settle = $1
#   proceeds         = 200 × 1                      = 200
#   pnl              = 200 − 100                    = +100
#   return_pct       = 100 / 100                    = +1.0
#   ending           = 1000 − 100 + 200             = 1100
#   total_return     = 1100 / 1000 − 1              = +0.10
# -----------------------------------------------------------------------------


def test_case_01_yes_at_050_wins() -> None:
    spec = make_spec(side="YES", sizing_value=0.1, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.50, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=300)
    h = _assert_hash_stable(spec, dataset, config)

    r = run_backtest(spec, dataset, config)
    assert r.result_hash == h
    assert r.metrics.standard.num_trades == 1
    t = r.trades[0]
    assert abs(t.shares - 200.0) < EPS
    assert abs(t.proceeds - 200.0) < EPS
    assert abs(t.pnl - 100.0) < EPS
    assert abs(t.return_pct - 1.0) < EPS
    assert abs(r.metrics.standard.ending_capital - 1100.0) < EPS
    assert abs(r.metrics.standard.total_return - 0.10) < EPS
    assert r.metrics.standard.win_rate == 1.0


# -----------------------------------------------------------------------------
# Case 2 — YES @ 0.50 loses → proceeds $0, pnl -$100
# -----------------------------------------------------------------------------
#
# Hand calculation:
#   target=1, outcome=0 → yes_payoff = False → YES loses → settle = $0
#   proceeds         = 200 × 0                      = 0
#   pnl              = 0 − 100                      = −100
#   return_pct       = −100 / 100                   = −1.0
#   ending           = 1000 − 100 + 0               = 900
#   total_return     = −0.10
# -----------------------------------------------------------------------------


def test_case_02_yes_at_050_loses() -> None:
    spec = make_spec(side="YES", sizing_value=0.1, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.50, outcome=0, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=300)
    _assert_hash_stable(spec, dataset, config)
    r = run_backtest(spec, dataset, config)

    assert r.metrics.standard.num_trades == 1
    t = r.trades[0]
    assert t.proceeds == 0.0
    assert abs(t.pnl + 100.0) < EPS
    assert abs(t.return_pct + 1.0) < EPS
    assert abs(r.metrics.standard.ending_capital - 900.0) < EPS
    assert abs(r.metrics.standard.total_return + 0.10) < EPS
    assert r.metrics.standard.win_rate == 0.0


# -----------------------------------------------------------------------------
# Case 3 — NO @ 0.96 wins, $100 notional → shares 104.166…, pnl +4.166…
#         (TS regression case from runner.test.ts L127–156)
# -----------------------------------------------------------------------------
#
# Hand calculation:
#   side             = NO,  price = 0.96  (literal NO price, NO inversion)
#   shares           = 100 / 0.96                    = 104.166666666…
#   target=1, outcome=0 → yes_payoff = False → NO wins → settle = $1
#   proceeds         = shares × 1                    = 104.166666666…
#   pnl              = 104.166666… − 100             = +4.166666666…
#   return_pct       = (1/0.96) − 1                  = +0.0416666…
#   ending           = 1000 − 100 + 104.166666…      = 1004.166666…
#   total_return     = +0.00416666…
# -----------------------------------------------------------------------------


def test_case_03_no_at_096_wins() -> None:
    spec = make_spec(side="NO", sizing_value=0.1, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.96, outcome=0, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=300)
    _assert_hash_stable(spec, dataset, config)
    r = run_backtest(spec, dataset, config)

    assert r.metrics.standard.num_trades == 1
    t = r.trades[0]
    expected_shares = 100.0 / 0.96
    assert abs(t.shares - expected_shares) < EPS
    assert abs(t.proceeds - expected_shares) < EPS
    assert abs(t.pnl - (expected_shares - 100.0)) < EPS
    assert abs(t.return_pct - (1.0 / 0.96 - 1.0)) < EPS
    assert abs(r.metrics.standard.ending_capital - (1000.0 - 100.0 + expected_shares)) < EPS
    # No-inversion lock: entry_price_quote retained as the literal NO price
    assert t.entry_price_quote == 0.96


# -----------------------------------------------------------------------------
# Case 4 — NO @ 0.96 loses → pnl −$100
# -----------------------------------------------------------------------------
#
# Hand calculation:
#   target=1, outcome=1 → yes_payoff = True → NO loses → settle = $0
#   proceeds         = shares × 0   = 0
#   pnl              = 0 − 100      = −100
#   ending           = 900
# -----------------------------------------------------------------------------


def test_case_04_no_at_096_loses() -> None:
    spec = make_spec(side="NO", sizing_value=0.1, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.96, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=300)
    _assert_hash_stable(spec, dataset, config)
    r = run_backtest(spec, dataset, config)

    t = r.trades[0]
    assert t.proceeds == 0.0
    assert abs(t.pnl + 100.0) < EPS
    assert abs(r.metrics.standard.ending_capital - 900.0) < EPS
    assert r.metrics.standard.win_rate == 0.0


# -----------------------------------------------------------------------------
# Case 5 — YES @ 0.05 wins, sizing=1.0, 1-day hold
#         → ending $20,000, total_return 19.0, cagr=None,
#           CAGR_EXTRAPOLATION_OVERFLOW warning (HIGH-1 patch landed 2026-05-23)
# -----------------------------------------------------------------------------
#
# Hand calculation:
#   notional   = 1.0 × 1000 = 1000
#   shares     = 1000 / 0.05 = 20_000
#   target=1, outcome=1 → YES wins → proceeds = 20_000 × 1 = 20_000
#   ending     = 1000 − 1000 + 20_000 = 20_000
#   total_return = 20_000 / 1000 − 1 = 19.0
#
#   year_fraction = max(86_400 / 31_557_600, 1/365) = 1/365
#   (ending/start)^(1/year) = 20^365 ≈ 1.8e475 → OverflowError → cagr = None
# -----------------------------------------------------------------------------


def test_case_05_cagr_overflow_high1() -> None:
    spec = make_spec(side="YES", sizing_value=1.0, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=0, res_ts=DAY, price=0.05, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=2 * DAY)

    # Must NOT raise OverflowError (HIGH-1 patch).
    _assert_hash_stable(spec, dataset, config)
    r = run_backtest(spec, dataset, config)

    assert r.metrics.standard.num_trades == 1
    assert abs(r.metrics.standard.ending_capital - 20_000.0) < EPS
    assert abs(r.metrics.standard.total_return - 19.0) < EPS
    assert r.metrics.standard.cagr is None
    assert any(w.code == WarningCode.CAGR_EXTRAPOLATION_OVERFLOW for w in r.warnings)


# -----------------------------------------------------------------------------
# Case 6 — fee-at-entry: $100 notional, 50 bps fee, mark_at_cost equity
#         drops to $999.50 at the decision-time event (B-1 of math bash)
# -----------------------------------------------------------------------------
#
# Hand calculation:
#   notional        = 100
#   fee             = 100 × 50/10_000              = 0.50
#   investable      = 100 − 0.50                   = 99.50
#   shares          = 99.50 / 0.50                 = 199
#   At decision T=100:
#     cash          = 1000 − 100                   = 900
#     mark_value    = shares × entry_price = 199 × 0.50 = 99.50
#     equity        = 900 + 99.50                  = 999.50
#   At resolution (YES wins):
#     proceeds      = 199 × 1                      = 199
#     cash          = 900 + 199                    = 1099
#     ending        = 1099   (mark_value = 0 after close)
# -----------------------------------------------------------------------------


def test_case_06_fee_realized_at_entry() -> None:
    spec = make_spec(side="YES", sizing_value=0.1, fee_bps=50, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=500, price=0.50, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=1000)
    _assert_hash_stable(spec, dataset, config)
    r = run_backtest(spec, dataset, config)

    decision_point = next(p for p in r.equity_curve if p.t == 100)
    assert abs(decision_point.equity - 999.50) < EPS, (
        f"fee-at-entry: decision-time equity should be $999.50, got {decision_point.equity}"
    )
    assert abs(r.metrics.standard.ending_capital - 1099.0) < EPS


# -----------------------------------------------------------------------------
# Case 7 — overlapping trades: trade B cannot spend trade A's future proceeds
# -----------------------------------------------------------------------------
#
# Hand calculation:
#   sizing.value = 0.9
#   A: dec_ts=100, res_ts=1000, price=0.50, outcome=1 (YES wins)
#   B: dec_ts=200, res_ts=2000, price=0.50, outcome=1 (YES wins)
#
#   At T=100 (A.DECISION):  A.cost = 0.9 × 1000 = $900.  cash = $100.
#   At T=200 (B.DECISION):  cash (pre-A-settlement) = $100.
#                            B.cost = 0.9 × $100 = $90.    cash = $10.
#     ↑ If future-cash leak were present, B would see $1900 (post-A's $1800
#       proceeds) and B.cost would be $1710. The event-time ledger prevents this.
#   At T=1000 (A.RESOLUTION): A.proceeds = 1800.  cash = $1810.
#   At T=2000 (B.RESOLUTION): B.proceeds = 180.   cash = $1990.
#   ending = $1990.  total_return = 0.99.
# -----------------------------------------------------------------------------


def test_case_07_overlapping_no_future_cash_leak() -> None:
    spec = make_spec(side="YES", sizing_value=0.9, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=1000, price=0.50, outcome=1, alpha=3.0, target=1),
        row(mkt="m/B", dec_ts=200, res_ts=2000, price=0.50, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=3000)
    _assert_hash_stable(spec, dataset, config)
    r = run_backtest(spec, dataset, config)

    assert r.metrics.standard.num_trades == 2
    a, b = sorted(r.trades, key=lambda t: t.entry_t)
    assert abs(a.cost - 900.0) < EPS
    assert abs(b.cost - 90.0) < EPS, (
        f"B.cost = {b.cost}: future cash leak — B should see pre-A-settlement cash $100, "
        "sized to $90; NOT post-A $1900 sized to $1710."
    )
    assert abs(r.metrics.standard.ending_capital - 1990.0) < EPS
    assert abs(r.metrics.standard.total_return - 0.99) < EPS


# -----------------------------------------------------------------------------
# Case 8 — same timestamp: DECISION processes before RESOLUTION
# -----------------------------------------------------------------------------
#
# Hand calculation:
#   sizing.value = 0.5
#   A: dec_ts=100, res_ts=500, price=0.50, outcome=1 (YES wins)
#   B: dec_ts=500, res_ts=900, price=0.50, outcome=1 (YES wins)
#                ↑ same T=500 as A's resolution
#
#   Event ordering: (time, kind) with DECISION=0 < RESOLUTION=1.
#   At T=500: B.DECISION processes BEFORE A.RESOLUTION.
#
#   A: cost = 0.5 × 1000 = $500.  cash after A.open = $500.
#   B at T=500 sees PRE-A-settlement cash = $500.
#     B.cost = 0.5 × $500 = $250.  cash after B.open = $250.
#   Then A.RESOLUTION at T=500: A.proceeds = $1000.  cash = $1250.
#   B.RESOLUTION at T=900: B.proceeds = $500.        cash = $1750.
#
#   ending = $1750.
# -----------------------------------------------------------------------------


def test_case_08_same_timestamp_decision_before_resolution() -> None:
    spec = make_spec(side="YES", sizing_value=0.5, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=500, price=0.50, outcome=1, alpha=3.0, target=1),
        row(mkt="m/B", dec_ts=500, res_ts=900, price=0.50, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=1000)
    _assert_hash_stable(spec, dataset, config)
    r = run_backtest(spec, dataset, config)

    a, b = sorted(r.trades, key=lambda t: t.entry_t)
    assert abs(a.cost - 500.0) < EPS
    assert abs(b.cost - 250.0) < EPS, (
        f"B.cost = {b.cost}: DECISION-before-RESOLUTION ordering broken. "
        "B should see pre-A-settlement cash $500 → cost $250, NOT post-A $1000 → cost $500."
    )
    assert abs(r.metrics.standard.ending_capital - 1750.0) < EPS

    # Exactly one equity sample per unique timestamp (no duplicates at T=500)
    times = [p.t for p in r.equity_curve]
    assert len(times) == len(set(times)), f"duplicate equity timestamps: {times}"


# -----------------------------------------------------------------------------
# Case 9 — zero trades → cagr 0, sharpe None, win_rate None,
#         NO_TRADES_GENERATED warning
# -----------------------------------------------------------------------------
#
# Hand calculation:
#   entry condition alpha >= 999.0 NEVER fires on the supplied alpha=3.0 row.
#   All metrics fall back to zero-trade defaults:
#     total_return = 0.0    (no equity change)
#     cagr         = 0.0    (piecewise: num_trades == 0 → 0.0)
#     sharpe       = None   (n < 2)
#     sortino      = None
#     win_rate     = None   (no trades, not zero)
#     max_drawdown = 0.0
# -----------------------------------------------------------------------------


def test_case_09_zero_trades() -> None:
    spec = make_spec(
        side="YES",
        starting_capital=1000.0,
        entry_when={"feature": "alpha", "gte": 999.0},   # never fires
    )
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.50, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=300)
    _assert_hash_stable(spec, dataset, config)
    r = run_backtest(spec, dataset, config)

    s = r.metrics.standard
    assert s.num_trades == 0
    assert s.total_return == 0.0
    assert s.cagr == 0.0
    assert s.sharpe is None
    assert s.sortino is None
    assert s.win_rate is None
    assert s.max_drawdown == 0.0

    codes = {w.code for w in r.warnings}
    assert WarningCode.NO_TRADES_GENERATED in codes
    assert WarningCode.NO_TRADES_NO_CI in codes


# -----------------------------------------------------------------------------
# Case 10 — account ruined → cagr -1.0, RUINED warning
# -----------------------------------------------------------------------------
#
# Hand calculation:
#   sizing.value = 1.0 → notional = $1000 (the full bank)
#   price = 0.5 → shares = 2000
#   target=1, outcome=0 → YES loses → proceeds = 0
#   ending = 1000 − 1000 + 0 = $0
#   cagr piecewise: ending_equity ≤ 0 → cagr = −1.0, ruined = True
# -----------------------------------------------------------------------------


def test_case_10_account_ruined() -> None:
    spec = make_spec(side="YES", sizing_value=1.0, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.50, outcome=0, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=300)
    _assert_hash_stable(spec, dataset, config)
    r = run_backtest(spec, dataset, config)

    assert r.metrics.standard.num_trades == 1
    assert r.metrics.standard.ending_capital == 0.0
    assert r.metrics.standard.cagr == -1.0
    assert any(w.code == WarningCode.RUINED for w in r.warnings)
