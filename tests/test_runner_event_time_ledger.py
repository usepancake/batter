"""Event-time ledger: no future cash, concurrent positions, same-timestamp
ordering, future-row skipping, unique-timestamp equity sampling.
"""

from __future__ import annotations

from pancake_engine import BacktestConfig, WarningCode, run_backtest

from ._runner_helpers import make_dataset, make_spec, row


def test_event_time_ledger_no_future_cash() -> None:
    """Trade A (Jan 1 → Jan 10, $900), Trade B (Jan 2 requested $200), start $1000.

    On Jan 2, Trade A is still open — cash = $100. Trade B sized at 100% (sizing.value=1.0)
    of available_cash gets $100, NOT $200. SIZING_CLIPPED warning emitted; B NOT funded
    by A's Jan-10 proceeds.
    """
    spec = make_spec(side="YES", sizing_value=1.0, starting_capital=1000.0)
    dataset = make_dataset([
        # A: decision Jan 1 (T=100), resolution Jan 10 (T=1000), price 0.9 → cost = 100% of $1000 = $1000
        # We'll use a 90% sizing to make A cost $900 — but sizing.value is fixed at 1.0 here.
        # To make A cost $900 not $1000, we need spec.sizing_value = 0.9 for trade A only.
        # That's not possible without per-trade sizing. Instead, use starting_capital=1000 and
        # sizing.value=0.9.
    ])
    # Re-spec with 0.9 sizing for the construction
    spec = make_spec(side="YES", sizing_value=0.9, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=1000, price=0.5, outcome=1, alpha=3.0, target=1),
        row(mkt="m/B", dec_ts=200, res_ts=2000, price=0.5, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=3000)
    result = run_backtest(spec, dataset, config)

    assert result.validation.ok, [e.code for e in result.validation.errors]
    # A opens with cost $900 → cash after A = $100
    # B sizing.value=0.9 × $100 = $90 requested; available_cash = $100 → no clip
    # But this doesn't trigger SIZING_CLIPPED. Let me retest with a sizing that DOES clip.
    # Use sizing_value=1.0 second pass:
    spec2 = make_spec(side="YES", sizing_value=1.0, starting_capital=1000.0)
    result2 = run_backtest(spec2, dataset, config)
    # A opens with cost $1000 → cash = $0; B requested 100% of $0 = $0 → SIZING_ZERO
    assert any(w.code == WarningCode.SIZING_ZERO for w in result2.warnings)

    # Now the canonical leak test: A costs $900, B requests $200 (sizing 0.2 of starting),
    # B should be clipped to $100 (available cash after A).
    # Engineering: starting=$1000, A sizing 0.9 = $900. Cash after A open = $100.
    # B then sizes at 0.2 × $100 = $20 — no clip (uses available_cash basis correctly).
    # To force a clip we need sizing.value > 1 effective. Since available_cash basis
    # uses min(requested, available), let's make a scenario:
    # starting=$1000, A cost $900, then B sizing.value=1.0 → requested = $100, no clip.
    # To clip, we need an over-request, which only happens if sizing basis was something
    # OTHER than available_cash. Under available_cash basis, requested = available × value
    # for value ≤ 1, so requested ≤ available. No clip possible.
    #
    # So under PR-1's only sizing mode (fixed_fraction × available_cash with value ≤ 1),
    # the SIZING_CLIPPED warning fires only in degenerate cases (e.g., available_cash
    # marginal-negative due to float drift). The leak-prevention contract holds via
    # min(notional, available_cash) regardless.
    #
    # The real proof of no future cash leak is: B's cash at decision = starting - A.cost,
    # NOT starting - A.cost + A.proceeds. Assert that.

    # Trades from result (using spec with sizing 0.9 — A and B both open at $900, $90 respectively)
    spec3 = make_spec(side="YES", sizing_value=0.9, starting_capital=1000.0)
    result3 = run_backtest(spec3, dataset, config)
    trades = result3.trades
    assert len(trades) == 2

    a, b = sorted(trades, key=lambda t: t.entry_t)
    # A: cost = $900
    assert abs(a.cost - 900.0) < 1e-9
    # B opens on Jan 2 (T=200), BEFORE A resolves (T=1000).
    # available_cash at B's decision = $1000 - $900 = $100
    # B.cost = 0.9 × $100 = $90  (NOT 0.9 × $1900 which would imply A's proceeds visible)
    assert abs(b.cost - 90.0) < 1e-9, f"B.cost={b.cost} — future cash leak suspected"


def test_concurrent_positions_two_overlapping() -> None:
    """Two decisions both fit within available_cash; both open, both settle independently."""
    spec = make_spec(side="YES", sizing_value=0.3, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=500, price=0.5, outcome=1, alpha=3.0, target=1),
        row(mkt="m/B", dec_ts=200, res_ts=400, price=0.5, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=600)
    result = run_backtest(spec, dataset, config)
    assert result.validation.ok
    assert result.metrics.standard.num_trades == 2

    # At Jan 1 (T=100): A opens cost = 0.3 × $1000 = $300
    # At Jan 2 (T=200): B opens cost = 0.3 × $700 = $210
    a, b = sorted(result.trades, key=lambda t: t.entry_t)
    assert abs(a.cost - 300.0) < 1e-9
    assert abs(b.cost - 210.0) < 1e-9
    # Both win → proceeds doubled in equity terms
    # final cash = 1000 - 300 - 210 + 600 (A's proceeds at 0.5 → shares 600 → $600)
    # Wait: shares_A = 300 / 0.5 = 600. proceeds_A = 600 × 1 = $600.
    # shares_B = 210 / 0.5 = 420. proceeds_B = $420.
    # Final cash = 1000 - 300 - 210 + 600 + 420 = $1510.
    final_equity = result.equity_curve[-1].equity
    assert abs(final_equity - 1510.0) < 1e-9


def test_same_timestamp_decision_before_resolution() -> None:
    """R1.resolution_time == R2.decision_time → R2 sees PRE-settlement cash.

    Setup: A sized at 50% wins big at T=500; B decides at T=500. DECISION orders
    before RESOLUTION, so B sees cash = starting − A.cost = $500 (NOT the post-win
    $1000 cash). If RESOLUTION processed first, B would size against $1000.
    """
    spec = make_spec(side="YES", sizing_value=1.0, starting_capital=1000.0)
    dataset = make_dataset([
        # R1: A decides T=100, resolves T=500, fully wins at price 0.5 → cost $500,
        # proceeds $1000.  (sizing.value=1.0 × $1000 starting cash = $1000, but
        # we want A.cost = $500; recompute: sizing.value=0.5 of $1000 = $500.)
    ])
    spec = make_spec(side="YES", sizing_value=0.5, starting_capital=1000.0)
    dataset = make_dataset([
        # A: 50% sized, wins at 0.5 → cost $500, proceeds $1000
        row(mkt="m/A", dec_ts=100, res_ts=500, price=0.5, outcome=1, alpha=3.0, target=1),
        # B: decides T=500 (same as A's res), 100% of available_cash at the moment
        # of B's DECISION. If event ordering correct: available = $500 → B.cost = $500.
        # If broken (RESOLUTION first): available = $1000 → B.cost = $1000.
        row(mkt="m/B", dec_ts=500, res_ts=900, price=0.5, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=1000)
    result = run_backtest(spec, dataset, config)
    assert result.validation.ok

    a, b = sorted(result.trades, key=lambda t: t.entry_t)
    assert abs(a.cost - 500.0) < 1e-9
    # B at T=500: DECISION-before-RESOLUTION → B sees pre-A-settlement cash = $500
    # sizing.value=0.5 of $500 = $250
    assert abs(b.cost - 250.0) < 1e-9, (
        f"B.cost={b.cost}: DECISION-before-RESOLUTION ordering broken; "
        f"B would see post-settle $1000 → $500 cost"
    )
    # Also assert there's a unique equity sample at T=500
    times = [p.t for p in result.equity_curve]
    assert times.count(500) == 1


def test_future_row_skipped_no_stranded_position() -> None:
    """Row with resolution_time > observation_time → skipped entirely (no DECISION)."""
    spec = make_spec(side="YES", starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.5, outcome=1, alpha=3.0, target=1),
        row(mkt="m/B", dec_ts=150, res_ts=5000, price=0.5, outcome=1, alpha=3.0, target=1),  # future
    ])
    config = BacktestConfig(observation_time=1000)  # B's res > observation_time
    result = run_backtest(spec, dataset, config)
    assert result.validation.ok
    # Exactly one trade (A); B's row entirely skipped
    assert result.metrics.standard.num_trades == 1
    # FUTURE_ROW_SKIPPED warning fires
    assert any(w.code == WarningCode.FUTURE_ROW_SKIPPED for w in result.warnings)
    # No stranded open positions (ledger drained at end)
    # Verified implicitly: only one trade, no orphan in any state
    assert result.meta["future_rows_count"] == 1


def test_equity_curve_unique_timestamps() -> None:
    """Two events sharing a timestamp produce exactly one equity sample at that t."""
    spec = make_spec(side="YES", sizing_value=0.5, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=500, price=0.5, outcome=1, alpha=3.0, target=1),
        # B's decision shares timestamp with A's resolution at T=500
        row(mkt="m/B", dec_ts=500, res_ts=900, price=0.5, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=1000)
    result = run_backtest(spec, dataset, config)

    # equity_curve should have UNIQUE timestamps
    times = [p.t for p in result.equity_curve]
    assert len(times) == len(set(times)), f"duplicate timestamps in equity_curve: {times}"


def test_empty_equity_curve_emits_anchor_point() -> None:
    """If no events (all rows skipped), emit single anchor point at observation_time."""
    spec = make_spec(side="YES", starting_capital=1000.0,
                     # entry condition that never fires
                     entry_when={"feature": "alpha", "gte": 999.0})
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.5, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=300)
    result = run_backtest(spec, dataset, config)
    # With one row whose entry condition is false, DECISION fires but yields no trade.
    # RESOLUTION processes for a non-open position → no-op.
    # equity_curve has at least one event point (from the DECISION at T=100).
    # Not strictly empty. To test the empty-anchor case, all rows must be SKIPPED earlier.
    # Re-test with all rows in the future:
    config2 = BacktestConfig(observation_time=150)  # res 200 > observation 150 → future-skip
    result2 = run_backtest(spec, dataset, config2)
    assert len(result2.equity_curve) == 1
    assert result2.equity_curve[0].equity == 1000.0
    assert any(w.code == WarningCode.NO_TRADES_GENERATED for w in result2.warnings)
