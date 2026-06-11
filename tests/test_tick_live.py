"""Live-mode tick tests (ADR-0050 L1, Path A).

Contract under test:
- mode="live" runs the SAME decision path as paper up to the fill boundary.
- Live mode does NOT mutate new_positions / new_cash from entries.
- target_positions carries one TargetPosition per entered instrument.
- Paper mode keeps target_positions=None (backward compat).
- Guards gate live intents exactly as paper fills.
- Provenance (source_manifest_id, instrument_id) round-trips from bar to TargetPosition.
- Determinism: same request ⇒ identical response across 3 runs.
"""

from __future__ import annotations

import math

import pytest

from pancake_engine import (
    MarketBar,
    ResolutionMarker,
    TargetPosition,
    TickError,
    TickPosition,
    TickRequest,
    TickResponse,
    tick,
)

from ._runner_helpers import make_spec


def _req(
    *,
    bars: list[MarketBar],
    positions: dict[str, TickPosition] | None = None,
    cash: float = 1000.0,
    cursor: int = 1000,
    side: str = "YES",
    slip_bps: float = 0.0,
    fee_bps: float = 0.0,
    sizing_value: float = 0.1,
    entry_when: dict | None = None,
    paper_guard: dict | None = None,
    mode: str = "live",
) -> TickRequest:
    spec = make_spec(
        side=side,
        sizing_value=sizing_value,
        slip_bps=slip_bps,
        fee_bps=fee_bps,
        entry_when=entry_when,
        paper_guard=paper_guard,
    )
    return TickRequest(
        deployment_id="dep-live",
        mode=mode,
        strategy_spec_ir=spec,
        tick_cursor=cursor,
        market_snapshot=bars,
        current_cash=cash,
        current_positions=positions or {},
    )


def _paper_req(
    *,
    bars: list[MarketBar],
    positions: dict[str, TickPosition] | None = None,
    cash: float = 1000.0,
    cursor: int = 1000,
    side: str = "YES",
    slip_bps: float = 0.0,
    fee_bps: float = 0.0,
    sizing_value: float = 0.1,
    entry_when: dict | None = None,
    paper_guard: dict | None = None,
) -> TickRequest:
    return _req(
        bars=bars,
        positions=positions,
        cash=cash,
        cursor=cursor,
        side=side,
        slip_bps=slip_bps,
        fee_bps=fee_bps,
        sizing_value=sizing_value,
        entry_when=entry_when,
        paper_guard=paper_guard,
        mode="paper",
    )


# ---------------------------------------------------------------------------
# (a) PARITY — same instruments entered, same signal_price == paper fill close
# ---------------------------------------------------------------------------


def test_live_parity_entered_instruments_match_paper() -> None:
    """Same request → same set of entered instruments in live and paper."""
    bar_a = MarketBar(instrument_id="A", observed_at=990, close=0.4, alpha=3.0)
    bar_b = MarketBar(instrument_id="B", observed_at=990, close=0.6, alpha=3.0)
    bar_skip = MarketBar(instrument_id="C", observed_at=990, close=0.3, alpha=0.5)  # alpha < 2.0

    paper = tick(_paper_req(bars=[bar_a, bar_b, bar_skip]))
    live = tick(_req(bars=[bar_a, bar_b, bar_skip]))

    # Paper opened positions on A and B.
    paper_entered = set(paper.new_positions)
    assert paper_entered == {"A", "B"}

    # Live target_positions must have the same keys.
    assert live.target_positions is not None
    assert set(live.target_positions) == paper_entered


def test_live_parity_signal_price_equals_paper_fill_close() -> None:
    """Live signal_price must equal bar.close (== paper fill's quote_price for YES)."""
    bar = MarketBar(instrument_id="X", observed_at=990, close=0.4, alpha=3.0)
    paper = tick(_paper_req(bars=[bar]))
    live = tick(_req(bars=[bar]))

    assert live.target_positions is not None
    assert "X" in live.target_positions
    tp: TargetPosition = live.target_positions["X"]

    # For a YES-side spec, signal_price = bar.close.
    assert math.isclose(tp.signal_price, bar.close)
    # Paper fill's quote_price is bar.close (no slippage).
    paper_pos = paper.new_positions["X"]
    assert math.isclose(tp.signal_price, paper_pos.entry_price)  # zero slip → fill == close


def test_live_parity_no_side_signal_price() -> None:
    """For NO-side specs, signal_price = 1 - bar.close."""
    bar = MarketBar(instrument_id="X", observed_at=990, close=0.08, alpha=3.0)
    live = tick(_req(
        bars=[bar],
        side="NO",
        entry_when={"feature": "price", "gte": 0.90},
    ))
    assert live.target_positions is not None
    assert "X" in live.target_positions
    tp = live.target_positions["X"]
    assert math.isclose(tp.signal_price, 1.0 - 0.08)


# ---------------------------------------------------------------------------
# (b) Live mode mutates nothing — new_positions / new_cash unchanged by entries
# ---------------------------------------------------------------------------


def test_live_no_cash_mutation_from_entries() -> None:
    """Live mode must NOT deduct cash for entered instruments."""
    bar = MarketBar(instrument_id="X", observed_at=990, close=0.4, alpha=3.0)
    live = tick(_req(bars=[bar], cash=1000.0))

    # Cash must remain 1000.0 (no fill deducted).
    assert math.isclose(live.new_cash, 1000.0)
    # No new positions opened.
    assert live.new_positions == {}


def test_live_no_position_mutation_from_entries() -> None:
    """Live target_positions keys must NOT appear in new_positions."""
    bar_a = MarketBar(instrument_id="A", observed_at=990, close=0.4, alpha=3.0)
    bar_b = MarketBar(instrument_id="B", observed_at=990, close=0.6, alpha=3.0)
    live = tick(_req(bars=[bar_a, bar_b]))

    assert "A" not in live.new_positions
    assert "B" not in live.new_positions
    assert live.target_positions is not None
    assert set(live.target_positions) == {"A", "B"}


def test_live_settlement_cash_effect_identical_to_paper() -> None:
    """Settlement (step 1) mutates cash identically in live and paper."""
    pos = TickPosition(
        instrument_id="X", side="YES", shares=250.0, entry_price=0.4,
        cost=100.0, fee=0.0, opened_at=900, last_mark=0.4,
    )
    bar = MarketBar(
        instrument_id="X", observed_at=990, close=0.4, alpha=0.0,
        resolution=ResolutionMarker(resolved_at=980, resolved_outcome=1),
    )
    paper = tick(_paper_req(bars=[bar], positions={"X": pos}, cash=900.0))
    live = tick(_req(bars=[bar], positions={"X": pos}, cash=900.0))

    assert math.isclose(paper.new_cash, live.new_cash)
    assert math.isclose(paper.new_equity, live.new_equity)
    assert live.target_positions == {}  # no new entries (only settlement)


# ---------------------------------------------------------------------------
# (c) Backward compat — request without mode behaves exactly as before
# ---------------------------------------------------------------------------


def test_backward_compat_default_mode_is_paper() -> None:
    """Omitting mode (default 'paper') behaves byte-identically to explicit paper."""
    bar = MarketBar(instrument_id="X", observed_at=990, close=0.4, alpha=3.0)
    spec = make_spec()
    req_default = TickRequest(
        deployment_id="dep-compat",
        strategy_spec_ir=spec,
        tick_cursor=1000,
        market_snapshot=[bar],
        current_cash=1000.0,
        # mode omitted — should default to "paper"
    )
    req_explicit = TickRequest(
        deployment_id="dep-compat",
        mode="paper",
        strategy_spec_ir=spec,
        tick_cursor=1000,
        market_snapshot=[bar],
        current_cash=1000.0,
    )
    resp_default = tick(req_default)
    resp_explicit = tick(req_explicit)

    assert resp_default.model_dump() == resp_explicit.model_dump()
    # Paper mode: target_positions must be None.
    assert resp_default.target_positions is None


# ---------------------------------------------------------------------------
# (d) Bogus mode still rejected
# ---------------------------------------------------------------------------


def test_bogus_mode_rejected() -> None:
    with pytest.raises(TickError) as ei:
        tick(_req(bars=[], mode="bogus"))
    assert ei.value.code == "UNSUPPORTED_MODE"
    assert ei.value.retryable is False


def test_unknown_mode_rejected() -> None:
    with pytest.raises(TickError) as ei:
        tick(_req(bars=[], mode="sim"))
    assert ei.value.code == "UNSUPPORTED_MODE"


# ---------------------------------------------------------------------------
# (e) Provenance echo — source_manifest_id + instrument_id round-trip
# ---------------------------------------------------------------------------


def test_provenance_source_manifest_id_round_trip() -> None:
    """source_manifest_id on the bar must appear on the TargetPosition."""
    bar = MarketBar(
        instrument_id="X", observed_at=990, close=0.4, alpha=3.0,
        source_manifest_id="manifest-abc123",
    )
    live = tick(_req(bars=[bar]))

    assert live.target_positions is not None
    tp = live.target_positions["X"]
    assert tp.source_manifest_id == "manifest-abc123"


def test_provenance_instrument_id_round_trip() -> None:
    """instrument_id from the bar must appear on the TargetPosition."""
    bar = MarketBar(instrument_id="poly-mkt-999", observed_at=990, close=0.4, alpha=3.0)
    live = tick(_req(bars=[bar]))

    assert live.target_positions is not None
    tp = live.target_positions["poly-mkt-999"]
    assert tp.instrument_id == "poly-mkt-999"


def test_provenance_absent_manifest_id_is_none() -> None:
    """Bar without source_manifest_id → TargetPosition.source_manifest_id is None."""
    bar = MarketBar(instrument_id="X", observed_at=990, close=0.4, alpha=3.0)
    live = tick(_req(bars=[bar]))

    assert live.target_positions is not None
    assert live.target_positions["X"].source_manifest_id is None


# ---------------------------------------------------------------------------
# (f) Guard-suspended live tick emits zero targets
# ---------------------------------------------------------------------------


def test_guard_suspended_live_emits_zero_targets() -> None:
    """A tripped paper_guard must suppress live targets exactly as paper fills."""
    # consecutive_losses=3 >= max_consecutive_losses=2 → guard trips immediately.
    bar = MarketBar(instrument_id="X", observed_at=990, close=0.4, alpha=3.0)
    live = tick(_req(
        bars=[bar],
        paper_guard={"max_consecutive_losses": 2, "cooldown_bars": 3},
        mode="live",
    ) if False else _req(  # reconstruct with consecutive_losses threaded in
        bars=[bar],
        paper_guard={"max_consecutive_losses": 2, "cooldown_bars": 3},
    ))
    # With default consecutive_losses=0 guard won't trip — need to thread state.
    # Use cooldown_remaining=1 to simulate an already-tripped guard.
    spec = make_spec(paper_guard={"max_consecutive_losses": 2, "cooldown_bars": 3})
    req_suspended = TickRequest(
        deployment_id="dep-guard",
        mode="live",
        strategy_spec_ir=spec,
        tick_cursor=1000,
        market_snapshot=[bar],
        current_cash=1000.0,
        current_positions={},
        cooldown_remaining=1,  # already in cooldown
    )
    resp = tick(req_suspended)
    assert resp.target_positions == {}  # zero targets while suspended


def test_guard_trips_live_emits_zero_targets_and_event() -> None:
    """Guard trips this tick → guard_suspended event emitted, zero targets."""
    bar = MarketBar(instrument_id="X", observed_at=990, close=0.4, alpha=3.0)
    spec = make_spec(paper_guard={"max_consecutive_losses": 1, "cooldown_bars": 2})
    req = TickRequest(
        deployment_id="dep-guard2",
        mode="live",
        strategy_spec_ir=spec,
        tick_cursor=1000,
        market_snapshot=[bar],
        current_cash=1000.0,
        current_positions={},
        consecutive_losses=1,  # at threshold → trips this tick
    )
    resp = tick(req)
    kinds = [e.event_kind for e in resp.events]
    assert "guard_suspended" in kinds
    assert resp.target_positions == {}


def test_guard_paper_and_live_suppress_same_entries() -> None:
    """Guard suppresses the SAME entries in both modes."""
    bar = MarketBar(instrument_id="X", observed_at=990, close=0.4, alpha=3.0)
    spec = make_spec(paper_guard={"max_consecutive_losses": 1, "cooldown_bars": 2})

    paper_req = TickRequest(
        deployment_id="dep-g", mode="paper", strategy_spec_ir=spec,
        tick_cursor=1000, market_snapshot=[bar], current_cash=1000.0,
        consecutive_losses=1,
    )
    live_req = TickRequest(
        deployment_id="dep-g", mode="live", strategy_spec_ir=spec,
        tick_cursor=1000, market_snapshot=[bar], current_cash=1000.0,
        consecutive_losses=1,
    )
    paper_resp = tick(paper_req)
    live_resp = tick(live_req)

    # Paper: no new positions (guard blocked).
    assert paper_resp.new_positions == {}
    # Live: no targets (guard blocked).
    assert live_resp.target_positions == {}


# ---------------------------------------------------------------------------
# (g) Determinism — same request 3× ⇒ identical response
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("run_i", [0, 1, 2])
def test_live_determinism(run_i: int) -> None:
    """Live tick is deterministic across calls."""
    bars = [
        MarketBar(instrument_id="A", observed_at=990, close=0.4, alpha=3.0,
                  source_manifest_id="m1"),
        MarketBar(instrument_id="B", observed_at=990, close=0.7, alpha=3.0),
    ]
    req = _req(bars=bars, cash=500.0)
    results = [tick(req).model_dump() for _ in range(3)]
    assert results[0] == results[1] == results[2]


def test_live_determinism_cross_mode_decision_set() -> None:
    """The set of entry decisions is identical across 3 paper/live pairs."""
    bars = [
        MarketBar(instrument_id="A", observed_at=990, close=0.3, alpha=3.0),
        MarketBar(instrument_id="B", observed_at=990, close=0.6, alpha=3.0),
        MarketBar(instrument_id="C", observed_at=990, close=0.5, alpha=0.5),  # no entry
    ]
    for _ in range(3):
        paper = tick(_paper_req(bars=bars))
        live = tick(_req(bars=bars))
        assert set(paper.new_positions) == set(live.target_positions or {})


# ---------------------------------------------------------------------------
# Additional: target_shares math
# ---------------------------------------------------------------------------


def test_target_shares_math() -> None:
    """target_shares = sizing_notional / signal_price (cost-free)."""
    # sizing_value=0.1, cash=1000 → notional=100; signal_price=0.4 → shares=250
    bar = MarketBar(instrument_id="X", observed_at=990, close=0.4, alpha=3.0)
    live = tick(_req(bars=[bar], cash=1000.0, sizing_value=0.1))

    assert live.target_positions is not None
    tp = live.target_positions["X"]
    expected_shares = (1000.0 * 0.1) / 0.4
    assert math.isclose(tp.target_shares, expected_shares)


def test_paper_target_positions_is_none() -> None:
    """Paper mode must always return target_positions=None."""
    bar = MarketBar(instrument_id="X", observed_at=990, close=0.4, alpha=3.0)
    resp = tick(_paper_req(bars=[bar]))
    assert resp.target_positions is None


def test_live_no_entries_empty_dict() -> None:
    """When no entry condition fires, live returns empty dict (not None)."""
    bar = MarketBar(instrument_id="X", observed_at=990, close=0.4, alpha=0.5)
    live = tick(_req(bars=[bar]))
    assert live.target_positions == {}
