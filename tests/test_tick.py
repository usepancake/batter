"""tick() — single-bar paper step (ADR-0035 as amended)."""

from __future__ import annotations

import math

import pytest

from pancake_engine import (
    ENGINE_VERIFICATION_GRADE,
    MarketBar,
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
    mode: str = "paper",
) -> TickRequest:
    spec = make_spec(
        side=side, sizing_value=sizing_value, slip_bps=slip_bps, fee_bps=fee_bps,
        entry_when=entry_when,
    )
    return TickRequest(
        deployment_id="dep-1",
        mode=mode,
        strategy_spec_ir=spec,
        tick_cursor=cursor,
        market_snapshot=bars,
        current_cash=cash,
        current_positions=positions or {},
    )


def _kinds(resp: TickResponse) -> list[str]:
    return [e.event_kind for e in resp.events]


# --- entries ---------------------------------------------------------------


def test_entry_opens_position() -> None:
    resp = tick(_req(bars=[MarketBar(instrument_id="X", observed_at=990, close=0.4, alpha=3.0)]))
    assert resp.new_cash == 900.0
    assert set(resp.new_positions) == {"X"}
    pos = resp.new_positions["X"]
    assert pos.side == "YES"
    assert math.isclose(pos.shares, 250.0)
    assert pos.opened_at == 1000
    assert _kinds(resp) == ["order_placed", "order_filled", "position_opened"]
    # equity = cash 900 + mark 250*0.4 = 1000 (zero costs)
    assert math.isclose(resp.new_equity, 1000.0)


def test_no_entry_when_condition_false() -> None:
    resp = tick(_req(bars=[MarketBar(instrument_id="X", observed_at=990, close=0.4, alpha=1.0)]))
    assert resp.new_cash == 1000.0
    assert resp.new_positions == {}
    assert resp.events == []
    assert math.isclose(resp.new_equity, 1000.0)


def test_no_side_entry_condition_uses_no_price() -> None:
    # For a NO-side spec, the entry condition references the entry_price column,
    # which run_backtest populates with the literal NO price (see
    # test_case_03_no_at_096_wins). bar.close is the YES price, so tick() must map
    # the entry_price column to 1 - bar.close — otherwise live/paper diverges.
    # YES close 0.08 → NO price 0.92 ≥ 0.90 → enters.
    resp = tick(_req(
        side="NO",
        entry_when={"feature": "price", "gte": 0.90},
        bars=[MarketBar(instrument_id="X", observed_at=990, close=0.08, alpha=3.0)],
    ))
    assert set(resp.new_positions) == {"X"}
    # Negative control: YES close 0.15 → NO price 0.85 < 0.90 → no entry.
    resp2 = tick(_req(
        side="NO",
        entry_when={"feature": "price", "gte": 0.90},
        bars=[MarketBar(instrument_id="Y", observed_at=990, close=0.15, alpha=3.0)],
    ))
    assert resp2.new_positions == {}


def test_resolved_instrument_not_entered() -> None:
    from pancake_engine import ResolutionMarker
    bar = MarketBar(
        instrument_id="X", observed_at=990, close=0.4, alpha=3.0,
        resolution=ResolutionMarker(resolved_at=980, resolved_outcome=1),
    )
    resp = tick(_req(bars=[bar]))
    assert resp.new_positions == {}
    assert resp.events == []


def test_entry_fires_on_gte_price_condition() -> None:
    # DF-1 regression: `gte` (price_above) entries must fire identically to `lte`.
    # The engine evaluates the compiled condition against bar.close (mapped onto
    # the entry_price column); a bar whose close satisfies `price gte 0.7` opens.
    resp = tick(_req(
        bars=[MarketBar(instrument_id="X", observed_at=990, close=0.8)],
        entry_when={"feature": "price", "gte": 0.7},
    ))
    assert set(resp.new_positions) == {"X"}
    assert _kinds(resp) == ["order_placed", "order_filled", "position_opened"]


def test_entry_fires_on_band_condition() -> None:
    # DF-1 regression: a band (all_of[gte, lte]) entry fires when close is in range.
    resp = tick(_req(
        bars=[MarketBar(instrument_id="X", observed_at=990, close=0.90)],
        entry_when={"all_of": [
            {"feature": "price", "gte": 0.85},
            {"feature": "price", "lte": 0.97},
        ]},
    ))
    assert set(resp.new_positions) == {"X"}
    assert _kinds(resp) == ["order_placed", "order_filled", "position_opened"]


def test_no_entry_when_band_upper_bound_exceeded() -> None:
    # Negative control: close above the band's upper bound must NOT enter.
    resp = tick(_req(
        bars=[MarketBar(instrument_id="X", observed_at=990, close=0.99)],
        entry_when={"all_of": [
            {"feature": "price", "gte": 0.85},
            {"feature": "price", "lte": 0.97},
        ]},
    ))
    assert resp.new_positions == {}
    assert resp.events == []


# --- hold / mark-to-market -------------------------------------------------


def test_hold_marks_at_market() -> None:
    pos = TickPosition(instrument_id="X", side="YES", shares=250.0, entry_price=0.4,
                       cost=100.0, fee=0.0, opened_at=900, last_mark=0.4)
    # price moved up to 0.6; alpha below entry threshold so no new entry
    resp = tick(_req(
        bars=[MarketBar(instrument_id="X", observed_at=990, close=0.6, alpha=0.0)],
        positions={"X": pos}, cash=900.0,
    ))
    assert resp.new_cash == 900.0  # unchanged on a hold
    assert _kinds(resp) == ["mark_to_market"]
    mtm = resp.events[0]
    assert math.isclose(mtm.payload["mark_price"], 0.6)
    assert mtm.payload["stale_mark"] is False
    # equity = 900 + 250*0.6 = 1050
    assert math.isclose(resp.new_equity, 1050.0)
    assert math.isclose(resp.new_positions["X"].last_mark, 0.6)


def test_no_side_marks_one_minus_close() -> None:
    pos = TickPosition(instrument_id="X", side="NO", shares=100.0, entry_price=0.6,
                       cost=60.0, fee=0.0, opened_at=900, last_mark=0.6)
    resp = tick(_req(
        bars=[MarketBar(instrument_id="X", observed_at=990, close=0.3, alpha=0.0)],
        positions={"X": pos}, cash=940.0, side="NO",
    ))
    mtm = resp.events[0]
    assert math.isclose(mtm.payload["mark_price"], 0.7)  # 1 - 0.3
    assert math.isclose(resp.new_equity, 940.0 + 100.0 * 0.7)


def test_stale_mark_carries_forward_when_absent() -> None:
    pos = TickPosition(instrument_id="X", side="YES", shares=250.0, entry_price=0.4,
                       cost=100.0, fee=0.0, opened_at=900, last_mark=0.55)
    # snapshot does NOT contain X
    resp = tick(_req(bars=[], positions={"X": pos}, cash=900.0))
    mtm = resp.events[0]
    assert mtm.payload["stale_mark"] is True
    assert math.isclose(mtm.payload["mark_price"], 0.55)
    assert math.isclose(resp.new_equity, 900.0 + 250.0 * 0.55)


# --- settlement ------------------------------------------------------------


def _resolved_bar(close: float, outcome: int, *, iid: str = "X", at: int = 980) -> MarketBar:
    from pancake_engine import ResolutionMarker
    return MarketBar(
        instrument_id=iid, observed_at=990, close=close, alpha=0.0,
        resolution=ResolutionMarker(resolved_at=at, resolved_outcome=outcome),
    )


@pytest.mark.parametrize(
    "side,outcome,wins",
    [("YES", 1, True), ("YES", 0, False), ("NO", 0, True), ("NO", 1, False)],
)
def test_settlement(side: str, outcome: int, wins: bool) -> None:
    cost = 100.0
    pos = TickPosition(instrument_id="X", side=side, shares=250.0, entry_price=0.4,
                       cost=cost, fee=0.0, opened_at=900, last_mark=0.4)
    resp = tick(_req(bars=[_resolved_bar(0.5, outcome)], positions={"X": pos},
                     cash=900.0, side=side))
    assert "X" not in resp.new_positions  # closed
    assert _kinds(resp) == ["position_closed"]
    settle_value = 1.0 if wins else 0.0
    proceeds = 250.0 * settle_value
    assert math.isclose(resp.new_cash, 900.0 + proceeds)
    assert math.isclose(resp.new_equity, 900.0 + proceeds)  # no open positions left
    ev = resp.events[0].payload
    assert math.isclose(ev["settle_value"], settle_value)
    assert math.isclose(ev["pnl"], proceeds - cost)


# --- no look-ahead (rule 139) ----------------------------------------------


def test_lookahead_bar_rejected() -> None:
    bar = MarketBar(instrument_id="X", observed_at=1001, close=0.4, alpha=3.0)
    with pytest.raises(TickError) as ei:
        tick(_req(bars=[bar], cursor=1000))
    assert ei.value.code == "LOOKAHEAD"
    assert ei.value.envelope["retryable"] is False


def test_lookahead_resolution_rejected() -> None:
    from pancake_engine import ResolutionMarker
    bar = MarketBar(instrument_id="X", observed_at=990, close=0.4, alpha=3.0,
                    resolution=ResolutionMarker(resolved_at=1001, resolved_outcome=1))
    with pytest.raises(TickError) as ei:
        tick(_req(bars=[bar], cursor=1000))
    assert ei.value.code == "LOOKAHEAD"


# --- guards ----------------------------------------------------------------


def test_held_instrument_not_re_entered() -> None:
    pos = TickPosition(instrument_id="X", side="YES", shares=250.0, entry_price=0.4,
                       cost=100.0, fee=0.0, opened_at=900, last_mark=0.4)
    # alpha high enough to fire entry, but X is already held
    resp = tick(_req(
        bars=[MarketBar(instrument_id="X", observed_at=990, close=0.4, alpha=3.0)],
        positions={"X": pos}, cash=900.0,
    ))
    assert set(resp.new_positions) == {"X"}
    assert resp.new_positions["X"].opened_at == 900  # unchanged (not re-opened)
    assert _kinds(resp) == ["mark_to_market"]


def test_unsupported_mode_rejected() -> None:
    with pytest.raises(TickError) as ei:
        tick(_req(bars=[], mode="live"))
    assert ei.value.code == "UNSUPPORTED_MODE"


def test_invalid_spec_ir_rejected() -> None:
    spec = make_spec()
    req = TickRequest(deployment_id="d", strategy_spec_ir=spec, tick_cursor=1000,
                      market_snapshot=[], current_cash=1000.0)
    # corrupt the compiled-condition shape post-construction
    req.strategy_spec_ir.strategy.entry = {"when": "not-a-dict"}
    with pytest.raises(TickError) as ei:
        tick(req)
    assert ei.value.code == "INVALID_SPEC_IR"


# --- contract surface + determinism ---------------------------------------


def test_verification_grade() -> None:
    resp = tick(_req(bars=[]))
    assert resp.verification_boundary.verification_grade == ENGINE_VERIFICATION_GRADE
    assert resp.verification_boundary.verification_grade == "engine-0.3-canonical"
    assert resp.suggested_next_check is None


def test_no_next_check_at_field() -> None:
    # The amended §2.2 response has NO next_check_at (dispatcher owns scheduling).
    resp = tick(_req(bars=[]))
    assert "next_check_at" not in resp.model_dump()
    # The actual field is `suggested_next_check`; assert it directly so the test
    # has teeth (the line above is vacuous — `next_check_at` never existed).
    assert resp.suggested_next_check is None


def test_determinism() -> None:
    args = dict(bars=[MarketBar(instrument_id="X", observed_at=990, close=0.4, alpha=3.0)])
    a = tick(_req(**args))
    b = tick(_req(**args))
    assert a.model_dump() == b.model_dump()


def test_position_round_trip_then_settle() -> None:
    # tick 1: open
    r1 = tick(_req(bars=[MarketBar(instrument_id="X", observed_at=990, close=0.4, alpha=3.0)]))
    pos_map = r1.new_positions
    # tick 2: same position resolves YES → closes, cash returns shares*1
    r2 = tick(_req(
        bars=[_resolved_bar(0.4, 1)],
        positions=pos_map, cash=r1.new_cash, cursor=1100,
    ))
    assert "X" not in r2.new_positions
    assert math.isclose(r2.new_cash, r1.new_cash + pos_map["X"].shares)
