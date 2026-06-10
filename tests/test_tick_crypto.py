"""tick_crypto() — single-bar paper step for the crypto-OHLCV family.

Tests cover:
- Entry fires at bar.close on a qualifying bar (long + short)
- Guard suspension blocks new entry but does NOT block exit
- Exit closes at bar.close with correct PnL
- Non-crypto spec_family is rejected
- PM tick suite is untouched (importing both tick and tick_crypto)
- Determinism: same request → same response
"""

from __future__ import annotations

import math

import pytest

from pancake_engine.crypto_ohlcv.types import CryptoOhlcvSpec
from pancake_engine.runner.tick import (
    CryptoTickBar,
    CryptoTickPosition,
    CryptoTickRequest,
    CryptoTickResponse,
    TickError,
    tick_crypto,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _spec(
    side: str = "long",
    *,
    entry_op: str = "gt",
    entry_rhs: float = 100.0,
    exit_op: str = "lt",
    exit_rhs: float = 100.0,
    slippage_bps: float = 0.0,
    fee_bps: float = 0.0,
    fraction: float = 0.5,
    paper_guard: dict | None = None,
) -> CryptoOhlcvSpec:
    strategy: dict = {
        "side": side,
        "indicators": [],
        "entry": {
            "op": entry_op,
            "left": {"ref": "price", "field": "close"},
            "right": {"ref": "const", "value": entry_rhs},
        },
        "exit": {
            "op": exit_op,
            "left": {"ref": "price", "field": "close"},
            "right": {"ref": "const", "value": exit_rhs},
        },
        "sizing": {"mode": "fixed_fraction", "value": fraction},
    }
    if paper_guard is not None:
        strategy["paper_guard"] = paper_guard
    return CryptoOhlcvSpec(
        spec_family="crypto-ohlcv-spec",
        spec_version="0.1",
        name="test-spec",
        instrument_id="BTC-USD",
        strategy=strategy,  # type: ignore[arg-type]
        costs={"slippage_bps": slippage_bps, "fee_bps": fee_bps},  # type: ignore[arg-type]
        starting_capital=10_000.0,
    )


def _bar(
    close: float,
    t: int = 1_000_000,
    *,
    open_: float | None = None,
    prev_close: float | None = None,
) -> CryptoTickBar:
    o = open_ if open_ is not None else close
    return CryptoTickBar(
        t=t,
        open=o,
        high=close,
        low=close,
        close=close,
        volume=1.0,
        prev_close=prev_close,
    )


def _req(
    *,
    spec: CryptoOhlcvSpec | None = None,
    bar: CryptoTickBar | None = None,
    cash: float = 1_000.0,
    position: CryptoTickPosition | None = None,
    mode: str = "paper",
    peak_equity: float | None = None,
    consecutive_losses: int = 0,
    cooldown_remaining: int = 0,
) -> CryptoTickRequest:
    s = spec or _spec()
    b = bar or _bar(110.0)
    return CryptoTickRequest(
        deployment_id="dep-test",
        mode=mode,
        strategy_spec=s,
        tick_cursor=b.t,
        bar=b,
        current_cash=cash,
        current_position=position,
        peak_equity=peak_equity,
        consecutive_losses=consecutive_losses,
        cooldown_remaining=cooldown_remaining,
    )


def _kinds(resp: CryptoTickResponse) -> list[str]:
    return [e.event_kind for e in resp.events]


# ---------------------------------------------------------------------------
# paper_fill_convention surfacing
# ---------------------------------------------------------------------------


def test_response_always_carries_bar_close_convention() -> None:
    """paper_fill_convention must always be 'bar_close' in every response."""
    resp = tick_crypto(_req())
    assert resp.paper_fill_convention == "bar_close"


def test_entry_event_payloads_carry_convention() -> None:
    """Every entry-related event payload must include paper_fill_convention."""
    resp = tick_crypto(_req())
    for ev in resp.events:
        if ev.event_kind in ("order_placed", "order_filled", "position_opened"):
            assert ev.payload.get("paper_fill_convention") == "bar_close", (
                f"{ev.event_kind} missing paper_fill_convention"
            )


# ---------------------------------------------------------------------------
# entry: long
# ---------------------------------------------------------------------------


def test_long_entry_fires_at_bar_close() -> None:
    """close > 100 triggers a long entry; fill at bar.close (zero costs)."""
    resp = tick_crypto(_req(bar=_bar(110.0), cash=1_000.0))
    assert _kinds(resp) == ["order_placed", "order_filled", "position_opened"]
    assert resp.new_position is not None
    pos = resp.new_position
    assert pos.side == "long"
    assert pos.entry_fill == 110.0   # bar.close, no slippage
    assert pos.entry_quote == 110.0
    # notional = 1000 * 0.5 = 500; fee = 0; qty = 500/110
    expected_qty = 500.0 / 110.0
    assert math.isclose(pos.qty, expected_qty, rel_tol=1e-9)
    assert math.isclose(resp.new_cash, 500.0, rel_tol=1e-9)
    # equity = 500 + qty * 110 = 1000
    assert math.isclose(resp.new_equity, 1_000.0, rel_tol=1e-9)


def test_long_no_entry_when_condition_false() -> None:
    """close <= 100 → no entry (entry condition: close > 100)."""
    resp = tick_crypto(_req(bar=_bar(100.0)))  # not gt 100
    assert resp.new_position is None
    assert resp.events == []
    assert math.isclose(resp.new_equity, 1_000.0)


# ---------------------------------------------------------------------------
# entry: short
# ---------------------------------------------------------------------------


def test_short_entry_fires_at_bar_close() -> None:
    """Short entry: close > 100 fires; short qty is negative; cash increases."""
    spec = _spec(side="short")
    resp = tick_crypto(_req(spec=spec, bar=_bar(110.0), cash=1_000.0))
    assert resp.new_position is not None
    pos = resp.new_position
    assert pos.side == "short"
    assert pos.qty < 0.0        # short = negative qty
    # cash goes up: cash += notional - fee
    assert resp.new_cash > 1_000.0


# ---------------------------------------------------------------------------
# slippage + fee accounting
# ---------------------------------------------------------------------------


def test_long_entry_with_slippage_and_fee() -> None:
    """Verify exact PnL accounting with non-zero costs."""
    spec = _spec(slippage_bps=50.0, fee_bps=10.0, fraction=1.0)
    bar = _bar(1000.0)
    resp = tick_crypto(_req(spec=spec, bar=bar, cash=1_000.0))
    pos = resp.new_position
    assert pos is not None
    # fill = close * (1 + slip) = 1000 * 1.005 = 1005
    assert math.isclose(pos.entry_fill, 1005.0, rel_tol=1e-9)
    # notional = 1000 * 1.0 = 1000; fee = 1000 * 0.001 = 1
    assert math.isclose(pos.entry_fee, 1.0, rel_tol=1e-9)
    # qty = (1000 - 1) / 1005
    expected_qty = 999.0 / 1005.0
    assert math.isclose(pos.qty, expected_qty, rel_tol=1e-9)
    assert math.isclose(resp.new_cash, 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# exit: close at bar.close
# ---------------------------------------------------------------------------


def test_exit_fires_and_closes_at_bar_close() -> None:
    """Exit condition fires (close < 90); position closed at bar.close."""
    spec = _spec(exit_rhs=90.0)  # exit when close < 90
    # open a position manually
    initial_qty = 10.0
    entry_fill = 100.0
    notional = initial_qty * entry_fill
    pos = CryptoTickPosition(
        instrument_id="BTC-USD",
        side="long",
        qty=initial_qty,
        entry_fill=entry_fill,
        entry_quote=entry_fill,
        notional=notional,
        entry_fee=0.0,
        opened_at=999_000,
    )
    bar = _bar(85.0)  # close < 90 → exit fires
    resp = tick_crypto(_req(spec=spec, bar=bar, cash=0.0, position=pos))

    assert resp.new_position is None
    assert "position_closed" in _kinds(resp)
    ev = next(e for e in resp.events if e.event_kind == "position_closed")
    assert ev.payload["reason"] == "exit"
    # exit_fill = 85.0 * (1 - 0) = 85.0 (no slippage)
    assert math.isclose(ev.payload["exit_fill"], 85.0, rel_tol=1e-9)
    # proceeds = 10 * 85 = 850; pnl = 10*(85-100) - 0 - 0 = -150
    assert math.isclose(ev.payload["pnl"], -150.0, rel_tol=1e-9)
    assert math.isclose(resp.new_cash, 850.0, rel_tol=1e-9)
    assert math.isclose(resp.new_equity, 850.0, rel_tol=1e-9)


def test_exit_event_carries_convention() -> None:
    """position_closed payload must carry paper_fill_convention='bar_close'."""
    spec = _spec(exit_rhs=90.0)
    pos = CryptoTickPosition(
        instrument_id="BTC-USD", side="long", qty=10.0,
        entry_fill=100.0, entry_quote=100.0, notional=1000.0,
        entry_fee=0.0, opened_at=999_000,
    )
    resp = tick_crypto(_req(spec=spec, bar=_bar(85.0), cash=0.0, position=pos))
    ev = next(e for e in resp.events if e.event_kind == "position_closed")
    assert ev.payload.get("paper_fill_convention") == "bar_close"


# ---------------------------------------------------------------------------
# paper_guard: guard suspension blocks entry but not exit
# ---------------------------------------------------------------------------


def test_guard_suspension_blocks_entry() -> None:
    """With cooldown_remaining > 0, entry is skipped."""
    resp = tick_crypto(_req(bar=_bar(110.0), cooldown_remaining=1))
    assert resp.new_position is None
    assert resp.events == []
    assert resp.cooldown_remaining == 0  # decremented


def test_guard_trip_emits_event_and_sets_cooldown() -> None:
    """Drawdown guard trips: guard_suspended event emitted, cooldown set."""
    spec = _spec(paper_guard={"max_drawdown_pct": 0.05, "cooldown_bars": 3})
    # peak=1000, start_equity=1000 * 0.9 = 900 → dd=10% > 5% → trip
    resp = tick_crypto(_req(
        spec=spec, bar=_bar(110.0), cash=900.0,
        peak_equity=1_000.0,
    ))
    assert "guard_suspended" in _kinds(resp)
    assert resp.new_position is None  # entry blocked
    assert resp.cooldown_remaining == 3


def test_guard_suspension_does_not_block_exit() -> None:
    """Guard suspended (cooldown_remaining > 0) must NOT prevent a position from
    being closed when the exit condition fires."""
    spec = _spec(exit_rhs=90.0, paper_guard={"max_drawdown_pct": 0.01})
    pos = CryptoTickPosition(
        instrument_id="BTC-USD", side="long", qty=10.0,
        entry_fill=100.0, entry_quote=100.0, notional=1000.0,
        entry_fee=0.0, opened_at=999_000,
    )
    # Already in cooldown; exit condition still fires.
    resp = tick_crypto(_req(
        spec=spec, bar=_bar(85.0), cash=0.0,
        position=pos, cooldown_remaining=2,
    ))
    assert resp.new_position is None  # exited
    assert "position_closed" in _kinds(resp)
    # Cooldown was still decremented (step 2 runs after step 1)
    assert resp.cooldown_remaining == 1


# ---------------------------------------------------------------------------
# validation / rejection
# ---------------------------------------------------------------------------


def test_non_crypto_spec_family_rejected() -> None:
    """A PM EvidenceSpec passed as strategy_spec must be rejected."""
    from pancake_engine.types import EvidenceSpec

    # We can't construct CryptoTickRequest with an EvidenceSpec directly (type
    # differs), so monkey-patch after construction.
    valid_crypto = _spec()
    req = _req(spec=valid_crypto)
    # Swap spec_family to simulate wrong family
    bad_spec = valid_crypto.model_copy(update={"spec_family": "pancake-evidence-spec"})
    req2 = CryptoTickRequest(
        deployment_id="dep-test",
        mode="paper",
        strategy_spec=bad_spec,
        tick_cursor=req.tick_cursor,
        bar=req.bar,
        current_cash=req.current_cash,
    )
    with pytest.raises(TickError) as ei:
        tick_crypto(req2)
    assert ei.value.code == "INVALID_SPEC_FAMILY"


def test_unsupported_mode_rejected() -> None:
    req = _req(mode="live")
    with pytest.raises(TickError) as ei:
        tick_crypto(req)
    assert ei.value.code == "UNSUPPORTED_MODE"


def test_bar_t_mismatch_rejected() -> None:
    """bar.t != tick_cursor must raise LOOKAHEAD."""
    b = _bar(110.0, t=999_000)
    req = CryptoTickRequest(
        deployment_id="dep-test",
        mode="paper",
        strategy_spec=_spec(),
        tick_cursor=1_000_000,  # different from bar.t
        bar=b,
        current_cash=1_000.0,
    )
    with pytest.raises(TickError) as ei:
        tick_crypto(req)
    assert ei.value.code == "LOOKAHEAD"


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_determinism() -> None:
    """Same request → identical response (dict-level)."""
    req = _req(bar=_bar(110.0), cash=1_000.0)
    a = tick_crypto(req)
    b = tick_crypto(req)
    assert a.model_dump() == b.model_dump()


def test_determinism_with_position_and_exit() -> None:
    spec = _spec(exit_rhs=90.0)
    pos = CryptoTickPosition(
        instrument_id="BTC-USD", side="long", qty=5.0,
        entry_fill=100.0, entry_quote=100.0, notional=500.0,
        entry_fee=0.0, opened_at=999_000,
    )
    req = _req(spec=spec, bar=_bar(85.0), cash=500.0, position=pos)
    a = tick_crypto(req)
    b = tick_crypto(req)
    assert a.model_dump() == b.model_dump()


# ---------------------------------------------------------------------------
# PM tick suite not broken (import guard)
# ---------------------------------------------------------------------------


def test_pm_tick_still_importable_and_usable() -> None:
    """Importing tick alongside tick_crypto must not cause conflicts."""
    from pancake_engine.runner.tick import tick, TickRequest, MarketBar

    req = TickRequest(
        deployment_id="dep-pm",
        mode="paper",
        strategy_spec_ir=_make_pm_spec(),
        tick_cursor=1000,
        market_snapshot=[],
        current_cash=1_000.0,
    )
    resp = tick(req)
    assert resp.new_equity == 1_000.0
    assert resp.events == []


def _make_pm_spec():
    from tests._runner_helpers import make_spec
    return make_spec()
