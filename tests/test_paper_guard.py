"""Feature 1 — strategy.paper_guard (tick() / paper lane only).

Guards stop NEW risk only; settlements always proceed.
State threads through TickRequest / TickResponse.
"""

from __future__ import annotations

import math

import pytest

from pancake_engine import (
    MarketBar,
    ResolutionMarker,
    TickPosition,
    TickRequest,
    TickResponse,
    tick,
)
from pancake_engine.validate import validate_spec

from ._runner_helpers import make_spec


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _req(
    *,
    bars: list[MarketBar],
    positions: dict[str, TickPosition] | None = None,
    cash: float = 1000.0,
    cursor: int = 1000,
    side: str = "YES",
    sizing_value: float = 0.1,
    paper_guard: dict | None = None,
    peak_equity: float | None = None,
    consecutive_losses: int = 0,
    cooldown_remaining: int = 0,
) -> TickRequest:
    spec = make_spec(side=side, sizing_value=sizing_value, paper_guard=paper_guard)
    return TickRequest(
        deployment_id="dep-1",
        mode="paper",
        strategy_spec_ir=spec,
        tick_cursor=cursor,
        market_snapshot=bars,
        current_cash=cash,
        current_positions=positions or {},
        peak_equity=peak_equity,
        consecutive_losses=consecutive_losses,
        cooldown_remaining=cooldown_remaining,
    )


def _entry_bar(iid: str = "X", close: float = 0.4) -> MarketBar:
    """A bar that fires the default entry condition (alpha=3.0 >= 2.0)."""
    return MarketBar(instrument_id=iid, observed_at=990, close=close, alpha=3.0)


def _resolved_bar(
    iid: str, *, close: float, outcome: int, at: int = 980
) -> MarketBar:
    return MarketBar(
        instrument_id=iid,
        observed_at=990,
        close=close,
        alpha=0.0,
        resolution=ResolutionMarker(resolved_at=at, resolved_outcome=outcome),
    )


def _held_pos(iid: str = "X", *, shares: float = 250.0, cost: float = 100.0) -> TickPosition:
    return TickPosition(
        instrument_id=iid,
        side="YES",
        shares=shares,
        entry_price=0.4,
        cost=cost,
        fee=0.0,
        opened_at=900,
        last_mark=0.4,
    )


# ---------------------------------------------------------------------------
# Spec without paper_guard → byte-identical tick behaviour
# ---------------------------------------------------------------------------


def test_no_guard_tick_identical() -> None:
    """A spec without paper_guard must produce identical TickResponse (byte-for-byte)."""
    bars = [_entry_bar()]
    req_base = _req(bars=bars)
    req_guard_none = TickRequest(
        deployment_id="dep-1",
        mode="paper",
        strategy_spec_ir=make_spec(paper_guard=None),
        tick_cursor=1000,
        market_snapshot=bars,
        current_cash=1000.0,
        current_positions={},
        peak_equity=None,
        consecutive_losses=0,
        cooldown_remaining=0,
    )
    resp_base = tick(req_base)
    resp_guard = tick(req_guard_none)
    # All fields that come from existing logic must match
    assert resp_base.new_cash == resp_guard.new_cash
    assert resp_base.new_equity == resp_guard.new_equity
    assert resp_base.events == resp_guard.events
    # New guard fields default to passthrough
    assert resp_guard.peak_equity is not None
    assert resp_guard.consecutive_losses == 0
    assert resp_guard.cooldown_remaining == 0


# ---------------------------------------------------------------------------
# TickRequest / TickResponse new fields have safe defaults
# ---------------------------------------------------------------------------


def test_tick_request_guard_fields_default() -> None:
    req = _req(bars=[])
    assert req.peak_equity is None
    assert req.consecutive_losses == 0
    assert req.cooldown_remaining == 0


def test_tick_response_guard_fields_present() -> None:
    resp = tick(_req(bars=[]))
    assert hasattr(resp, "peak_equity")
    assert hasattr(resp, "consecutive_losses")
    assert hasattr(resp, "cooldown_remaining")


# ---------------------------------------------------------------------------
# Equity peak tracking
# ---------------------------------------------------------------------------


def test_peak_equity_initialises_on_first_tick() -> None:
    resp = tick(_req(bars=[], cash=1000.0, peak_equity=None))
    # No positions → equity == cash == 1000; peak must be set to 1000
    assert math.isclose(resp.peak_equity, 1000.0)


def test_peak_equity_carries_forward_when_higher() -> None:
    resp = tick(_req(bars=[], cash=800.0, peak_equity=1000.0))
    assert math.isclose(resp.peak_equity, 1000.0)


def test_peak_equity_updates_when_equity_rises() -> None:
    resp = tick(_req(bars=[], cash=1200.0, peak_equity=1000.0))
    assert math.isclose(resp.peak_equity, 1200.0)


# ---------------------------------------------------------------------------
# max_drawdown_pct guard
# ---------------------------------------------------------------------------


def test_drawdown_guard_trips_and_suppresses_entries() -> None:
    """equity has dropped from peak 1000 → 700 (30% drawdown).
    Guard threshold is 20% → should trip."""
    # No open positions: equity == cash == 700
    resp = tick(_req(
        bars=[_entry_bar()],
        cash=700.0,
        peak_equity=1000.0,
        paper_guard={"max_drawdown_pct": 0.20},
    ))
    # Entry suppressed
    assert resp.new_positions == {}
    # guard_suspended event emitted
    kinds = [e.event_kind for e in resp.events]
    assert "guard_suspended" in kinds
    ev = next(e for e in resp.events if e.event_kind == "guard_suspended")
    assert ev.payload["guard"] == "max_drawdown_pct"
    assert ev.payload["threshold"] == pytest.approx(0.20)
    assert ev.payload["observed"] > 0.20
    # cooldown_remaining set to 1 when cooldown_bars absent
    assert resp.cooldown_remaining == 1


def test_drawdown_guard_trips_with_cooldown_bars() -> None:
    resp = tick(_req(
        bars=[_entry_bar()],
        cash=700.0,
        peak_equity=1000.0,
        paper_guard={"max_drawdown_pct": 0.20, "cooldown_bars": 3},
    ))
    assert resp.cooldown_remaining == 3
    assert resp.new_positions == {}


def test_drawdown_guard_does_not_trip_below_threshold() -> None:
    # 5% drawdown, 20% threshold → no trip
    resp = tick(_req(
        bars=[_entry_bar()],
        cash=950.0,
        peak_equity=1000.0,
        paper_guard={"max_drawdown_pct": 0.20},
    ))
    # Entry should fire normally
    assert "X" in resp.new_positions


def test_settlements_proceed_while_guard_tripped() -> None:
    """A resolution bar closes while guard is active; settlement must still happen.

    Scenario: guard trips this tick (drawdown >=5%); simultaneously Y resolves.
    We prove settlement of Y still processes even though entries are suppressed.
    The drawdown is computed from the start-of-tick equity (cash + marks of open positions).
    With cash=900, peak=1000, and Y held at mark=0.0 (close=1.0, NO-side: 1-1.0=0 → stale),
    start_equity = 900. drawdown = (1000-900)/1000 = 10% > 5% → trips.
    """
    # Y is held NO-side, yes_close=1.0 → mark per share = 1-1.0 = 0 → Y worth 0.
    # start_equity = cash(900) + mark(Y: 250*0 = 0) = 900. peak=1000 → 10% DD.
    pos = TickPosition(
        instrument_id="Y", side="NO", shares=250.0, entry_price=0.1,
        cost=100.0, fee=0.0, opened_at=900, last_mark=0.1,
    )
    bar_y = MarketBar(
        instrument_id="Y", observed_at=990, close=1.0, alpha=0.0,
        resolution=ResolutionMarker(resolved_at=980, resolved_outcome=0),
    )
    resp = tick(_req(
        bars=[
            _entry_bar("X"),  # entry candidate → suppressed by guard
            bar_y,            # held position → must settle (NO wins when outcome=0)
        ],
        positions={"Y": pos},
        cash=900.0,
        peak_equity=1000.0,
        paper_guard={"max_drawdown_pct": 0.05},  # 10% drawdown > 5% threshold
        side="NO",
    ))
    # Settlement happened (NO-side wins on outcome=0)
    assert "Y" not in resp.new_positions
    settled = next(e for e in resp.events if e.event_kind == "position_closed")
    assert settled.payload["instrument_id"] == "Y"
    # Entry was suppressed
    assert "X" not in resp.new_positions


# ---------------------------------------------------------------------------
# consecutive_losses guard
# ---------------------------------------------------------------------------


def test_consecutive_losses_count_threads_on_win() -> None:
    """A winning settlement resets consecutive_losses to 0."""
    pos = _held_pos("Y", cost=100.0)
    resp = tick(_req(
        bars=[_resolved_bar("Y", close=0.5, outcome=1)],
        positions={"Y": pos},
        cash=900.0,
        consecutive_losses=2,
        paper_guard={"max_consecutive_losses": 5},
    ))
    assert resp.consecutive_losses == 0


def test_consecutive_losses_count_increments_on_loss() -> None:
    """A losing settlement increments consecutive_losses."""
    pos = _held_pos("Y", cost=100.0, shares=250.0)
    resp = tick(_req(
        bars=[_resolved_bar("Y", close=0.5, outcome=0)],
        positions={"Y": pos},
        cash=900.0,
        consecutive_losses=1,
        paper_guard={"max_consecutive_losses": 5},
    ))
    assert resp.consecutive_losses == 2


def test_consecutive_losses_guard_trips() -> None:
    """After n consecutive full-loss settlements the guard trips on the NEXT entry tick."""
    # 3 losses already in state; threshold = 3 → should trip immediately
    resp = tick(_req(
        bars=[_entry_bar()],
        cash=1000.0,
        consecutive_losses=3,
        paper_guard={"max_consecutive_losses": 3},
    ))
    assert resp.new_positions == {}
    kinds = [e.event_kind for e in resp.events]
    assert "guard_suspended" in kinds
    ev = next(e for e in resp.events if e.event_kind == "guard_suspended")
    assert ev.payload["guard"] == "max_consecutive_losses"
    assert ev.payload["observed"] == 3
    assert ev.payload["threshold"] == 3


def test_consecutive_losses_guard_does_not_trip_below_threshold() -> None:
    resp = tick(_req(
        bars=[_entry_bar()],
        cash=1000.0,
        consecutive_losses=2,
        paper_guard={"max_consecutive_losses": 3},
    ))
    assert "X" in resp.new_positions


def test_consecutive_losses_no_settlement_no_change() -> None:
    """A tick with no settlements: consecutive_losses carries through."""
    resp = tick(_req(
        bars=[_entry_bar("Z")],
        cash=1000.0,
        consecutive_losses=1,
        # no guard → entries proceed; counter threads unchanged (no settlement)
        paper_guard={"max_consecutive_losses": 10},
    ))
    # Entry fired; no settlement → counter stays at 1
    assert resp.consecutive_losses == 1


# ---------------------------------------------------------------------------
# cooldown mechanics
# ---------------------------------------------------------------------------


def test_cooldown_suppresses_entries_no_extra_event() -> None:
    """While cooldown_remaining > 0 the engine skips entries silently (no extra event)."""
    resp = tick(_req(
        bars=[_entry_bar()],
        cash=1000.0,
        cooldown_remaining=2,
        paper_guard={"max_drawdown_pct": 0.20},  # guard not re-tripped (equity fine)
    ))
    # Entry suppressed
    assert resp.new_positions == {}
    # No guard_suspended event (cooldown already active)
    kinds = [e.event_kind for e in resp.events]
    assert "guard_suspended" not in kinds
    # Decremented
    assert resp.cooldown_remaining == 1


def test_cooldown_reaches_zero_then_entries_resume() -> None:
    """cooldown_remaining=1 → this tick skips entries; next tick (=0) enters again."""
    # Tick 1: cooldown_remaining=1 → suppresses, decrements to 0
    resp1 = tick(_req(
        bars=[_entry_bar()],
        cash=1000.0,
        cooldown_remaining=1,
        paper_guard={"max_drawdown_pct": 0.99},  # guard can't trip again
    ))
    assert resp1.new_positions == {}
    assert resp1.cooldown_remaining == 0

    # Tick 2: cooldown_remaining=0, guard can't trip (equity fine) → entry fires
    resp2 = tick(_req(
        bars=[_entry_bar()],
        cash=resp1.new_cash,
        cooldown_remaining=resp1.cooldown_remaining,
        paper_guard={"max_drawdown_pct": 0.99},
    ))
    assert "X" in resp2.new_positions


def test_settlements_proceed_during_cooldown() -> None:
    pos = _held_pos("Y", cost=100.0)
    resp = tick(_req(
        bars=[
            _entry_bar("X"),
            _resolved_bar("Y", close=0.5, outcome=0),
        ],
        positions={"Y": pos},
        cash=900.0,
        cooldown_remaining=2,
        paper_guard={"max_consecutive_losses": 10},
    ))
    assert "Y" not in resp.new_positions
    closed = next(e for e in resp.events if e.event_kind == "position_closed")
    assert closed.payload["instrument_id"] == "Y"
    assert "X" not in resp.new_positions
    assert resp.cooldown_remaining == 1


# ---------------------------------------------------------------------------
# Validation: paper_guard shape checks
# ---------------------------------------------------------------------------


def test_guard_empty_dict_rejected() -> None:
    spec = make_spec(paper_guard={})
    v = validate_spec(spec)
    assert not v.ok
    assert "E_EVIDENCE_SPEC_INVALID" in {e.code for e in v.errors}


def test_guard_unknown_key_rejected() -> None:
    spec = make_spec(paper_guard={"max_drawdown_pct": 0.2, "unknown_key": 1})
    v = validate_spec(spec)
    assert not v.ok
    assert "E_EVIDENCE_SPEC_INVALID" in {e.code for e in v.errors}


def test_guard_bad_drawdown_value_rejected() -> None:
    for bad in (0.0, 1.1, -0.1):
        spec = make_spec(paper_guard={"max_drawdown_pct": bad})
        v = validate_spec(spec)
        assert not v.ok, f"expected rejection for max_drawdown_pct={bad}"


def test_guard_bad_consecutive_losses_rejected() -> None:
    for bad in (0, -1):
        spec = make_spec(paper_guard={"max_consecutive_losses": bad})
        v = validate_spec(spec)
        assert not v.ok, f"expected rejection for max_consecutive_losses={bad}"


def test_guard_bad_cooldown_bars_rejected() -> None:
    spec = make_spec(paper_guard={"max_drawdown_pct": 0.2, "cooldown_bars": 0})
    v = validate_spec(spec)
    assert not v.ok


def test_guard_valid_single_key_ok() -> None:
    assert validate_spec(make_spec(paper_guard={"max_drawdown_pct": 0.25})).ok
    assert validate_spec(make_spec(paper_guard={"max_consecutive_losses": 3})).ok
    assert validate_spec(make_spec(paper_guard={"max_drawdown_pct": 0.1, "cooldown_bars": 5})).ok


def test_guard_all_keys_ok() -> None:
    v = validate_spec(make_spec(paper_guard={
        "max_drawdown_pct": 0.15,
        "max_consecutive_losses": 4,
        "cooldown_bars": 2,
    }))
    assert v.ok
