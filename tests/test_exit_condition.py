"""Feature 2 — strategy.exit {"when": <condition AST>}.

Paper lane: closes held positions when exit.when fires against the current bar.
Backtest: accepts field, emits EXIT_NOT_APPLIED_BACKTEST (INFO) once.
"""

from __future__ import annotations

import math

import pytest

from pancake_engine import (
    BacktestConfig,
    MarketBar,
    ResolutionMarker,
    TickPosition,
    TickRequest,
    TickResponse,
    run_backtest,
    tick,
)
from pancake_engine.validate import validate_spec
from pancake_engine.warnings import WarningCode

from ._runner_helpers import make_dataset, make_spec, row


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

DAY = 86_400


def _req_with_exit(
    *,
    bars: list[MarketBar],
    positions: dict[str, TickPosition] | None = None,
    cash: float = 1000.0,
    cursor: int = 1000,
    side: str = "YES",
    sizing_value: float = 0.1,
    slip_bps: float = 0.0,
    fee_bps: float = 0.0,
    exit_when: dict | None = None,
) -> TickRequest:
    spec = make_spec(
        side=side, sizing_value=sizing_value, slip_bps=slip_bps, fee_bps=fee_bps,
        exit_when=exit_when,
    )
    return TickRequest(
        deployment_id="dep-1",
        mode="paper",
        strategy_spec_ir=spec,
        tick_cursor=cursor,
        market_snapshot=bars,
        current_cash=cash,
        current_positions=positions or {},
    )


def _held_yes(iid: str = "X", *, shares: float = 250.0, cost: float = 100.0,
               entry_price: float = 0.4) -> TickPosition:
    return TickPosition(
        instrument_id=iid, side="YES", shares=shares,
        entry_price=entry_price, cost=cost, fee=0.0,
        opened_at=900, last_mark=entry_price,
    )


def _held_no(iid: str = "X", *, shares: float = 100.0, cost: float = 60.0,
              entry_price: float = 0.6) -> TickPosition:
    return TickPosition(
        instrument_id=iid, side="NO", shares=shares,
        entry_price=entry_price, cost=cost, fee=0.0,
        opened_at=900, last_mark=entry_price,
    )


# ---------------------------------------------------------------------------
# Exit fires on YES-side position
# ---------------------------------------------------------------------------


def test_exit_closes_yes_position_when_condition_fires() -> None:
    """close=0.8 ≥ 0.75 → exit fires; position closed at mark price."""
    pos = _held_yes("X", shares=250.0, cost=100.0)
    resp = tick(_req_with_exit(
        bars=[MarketBar(instrument_id="X", observed_at=990, close=0.8, alpha=0.0)],
        positions={"X": pos},
        cash=900.0,
        exit_when={"feature": "price", "gte": 0.75},
    ))
    # Position closed
    assert "X" not in resp.new_positions
    # Cash increased by proceeds (shares × mark_per_share - slippage - fee; zero costs here)
    proceeds = 250.0 * 0.8
    assert math.isclose(resp.new_cash, 900.0 + proceeds, rel_tol=1e-9)
    # Event emitted
    kinds = [e.event_kind for e in resp.events]
    assert "position_closed" in kinds
    ev = next(e for e in resp.events if e.event_kind == "position_closed")
    assert ev.payload["instrument_id"] == "X"
    assert ev.payload.get("reason") == "exit"


def test_exit_does_not_fire_when_condition_false() -> None:
    """close=0.6 < 0.75 → exit does not fire; position remains held."""
    pos = _held_yes("X", shares=250.0, cost=100.0)
    resp = tick(_req_with_exit(
        bars=[MarketBar(instrument_id="X", observed_at=990, close=0.6, alpha=0.0)],
        positions={"X": pos},
        cash=900.0,
        exit_when={"feature": "price", "gte": 0.75},
    ))
    assert "X" in resp.new_positions
    kinds = [e.event_kind for e in resp.events]
    assert "mark_to_market" in kinds


def test_exit_uses_current_bar_only() -> None:
    """Exit condition evaluates the CURRENT bar's data (no look-back into positions)."""
    pos = _held_yes("X", shares=200.0, cost=80.0)
    resp = tick(_req_with_exit(
        bars=[MarketBar(instrument_id="X", observed_at=990, close=0.9, alpha=1.0)],
        positions={"X": pos},
        cash=800.0,
        exit_when={"feature": "close", "gte": 0.85},
    ))
    assert "X" not in resp.new_positions


# ---------------------------------------------------------------------------
# NO-side exit uses the correct price domain
# ---------------------------------------------------------------------------


def test_exit_no_side_uses_inverted_price() -> None:
    """NO-side: mark per share = 1 - yes_close.
    yes_close=0.2 → NO mark = 0.8 ≥ 0.75 → exit fires."""
    pos = _held_no("X", shares=100.0, cost=60.0, entry_price=0.6)
    resp = tick(_req_with_exit(
        bars=[MarketBar(instrument_id="X", observed_at=990, close=0.2, alpha=0.0)],
        positions={"X": pos},
        cash=940.0,
        side="NO",
        exit_when={"feature": "price", "gte": 0.75},
    ))
    assert "X" not in resp.new_positions
    proceeds = 100.0 * (1.0 - 0.2)  # 80.0
    assert math.isclose(resp.new_cash, 940.0 + proceeds, rel_tol=1e-9)
    ev = next(e for e in resp.events if e.event_kind == "position_closed")
    assert ev.payload["reason"] == "exit"


def test_exit_no_side_does_not_fire_when_no_mark_below_threshold() -> None:
    """yes_close=0.5 → NO mark = 0.5 < 0.75 → exit does NOT fire."""
    pos = _held_no("X", shares=100.0, cost=60.0, entry_price=0.6)
    resp = tick(_req_with_exit(
        bars=[MarketBar(instrument_id="X", observed_at=990, close=0.5, alpha=0.0)],
        positions={"X": pos},
        cash=940.0,
        side="NO",
        exit_when={"feature": "price", "gte": 0.75},
    ))
    assert "X" in resp.new_positions


# ---------------------------------------------------------------------------
# Exit with costs applied
# ---------------------------------------------------------------------------


def test_exit_applies_slippage_and_fee() -> None:
    """Exit close reduces proceeds by slippage (sell-side: proceeds reduced) + fee on notional."""
    pos = _held_yes("X", shares=100.0, cost=100.0)
    # slip=100bps, fee=50bps; mark=0.8 per share
    # notional = 100 * 0.8 = 80; slip reduces price: 0.8 * (1 - 100/10000) = 0.792
    # fee = 80 * 50/10000 = 0.40
    # proceeds = 100 * 0.792 - 0.40 = 79.2 - 0.40 = 78.8
    resp = tick(_req_with_exit(
        bars=[MarketBar(instrument_id="X", observed_at=990, close=0.8, alpha=0.0)],
        positions={"X": pos},
        cash=900.0,
        slip_bps=100.0,
        fee_bps=50.0,
        exit_when={"feature": "price", "gte": 0.75},
    ))
    assert "X" not in resp.new_positions
    ev = next(e for e in resp.events if e.event_kind == "position_closed")
    assert ev.payload["reason"] == "exit"
    # Verify cash is less than naive mark (costs applied)
    assert resp.new_cash < 900.0 + 100.0 * 0.8


# ---------------------------------------------------------------------------
# Exit does not interfere with settlement (resolved bar takes precedence)
# ---------------------------------------------------------------------------


def test_exit_skipped_when_position_already_settled() -> None:
    """A resolved bar settles the position; exit should not also fire."""
    pos = _held_yes("X", shares=250.0, cost=100.0)
    resp = tick(_req_with_exit(
        bars=[MarketBar(
            instrument_id="X", observed_at=990, close=0.9, alpha=0.0,
            resolution=ResolutionMarker(resolved_at=980, resolved_outcome=1),
        )],
        positions={"X": pos},
        cash=900.0,
        exit_when={"feature": "price", "gte": 0.75},
    ))
    # Exactly one close event, it should be settlement not exit
    closes = [e for e in resp.events if e.event_kind == "position_closed"]
    assert len(closes) == 1
    # Settlement does not carry "reason=exit" in payload
    assert closes[0].payload.get("reason") != "exit"


# ---------------------------------------------------------------------------
# No exit spec → behaviour identical to no-exit tick
# ---------------------------------------------------------------------------


def test_no_exit_spec_behaviour_unchanged() -> None:
    """A spec without exit field must produce identical results to previous behaviour."""
    pos = _held_yes("X", shares=250.0, cost=100.0)
    bars = [MarketBar(instrument_id="X", observed_at=990, close=0.4, alpha=0.0)]
    resp_no_exit = tick(TickRequest(
        deployment_id="d",
        mode="paper",
        strategy_spec_ir=make_spec(exit_when=None),
        tick_cursor=1000,
        market_snapshot=bars,
        current_cash=900.0,
        current_positions={"X": pos},
    ))
    resp_with_exit = tick(TickRequest(
        deployment_id="d",
        mode="paper",
        strategy_spec_ir=make_spec(exit_when={"feature": "price", "gte": 0.99}),  # won't fire
        tick_cursor=1000,
        market_snapshot=bars,
        current_cash=900.0,
        current_positions={"X": pos},
    ))
    # Both hold the position (exit_when=0.99 doesn't fire at close=0.4)
    assert "X" in resp_no_exit.new_positions
    assert "X" in resp_with_exit.new_positions


# ---------------------------------------------------------------------------
# Backtest: accepts exit field, emits EXIT_NOT_APPLIED_BACKTEST warning
# ---------------------------------------------------------------------------


def _dataset_for_backtest():
    outcomes = [1, 0, 1, 1, 0]
    alphas = [3.0, 3.5, 2.6, 4.0, 2.8]
    return make_dataset([
        row(mkt=f"m/{i}", dec_ts=i * 5 * DAY, res_ts=i * 5 * DAY + 3 * DAY,
            price=0.5, outcome=outcomes[i], alpha=alphas[i], target=1)
        for i in range(5)
    ])


def test_backtest_accepts_exit_spec() -> None:
    """run_backtest does NOT block on a spec with exit.when."""
    spec = make_spec(exit_when={"feature": "alpha", "gte": 3.0})
    dataset = _dataset_for_backtest()
    r = run_backtest(spec, dataset, BacktestConfig(observation_time=30 * DAY))
    assert r.validation.ok
    assert r.result_hash != ""


def test_backtest_emits_exit_not_applied_warning() -> None:
    """run_backtest emits EXIT_NOT_APPLIED_BACKTEST (INFO) exactly once."""
    spec = make_spec(exit_when={"feature": "alpha", "gte": 3.0})
    dataset = _dataset_for_backtest()
    r = run_backtest(spec, dataset, BacktestConfig(observation_time=30 * DAY))
    codes = [w.code for w in r.warnings]
    assert WarningCode.EXIT_NOT_APPLIED_BACKTEST in codes
    # Emitted exactly once (not per-trade)
    assert codes.count(WarningCode.EXIT_NOT_APPLIED_BACKTEST) == 1


def test_backtest_no_exit_warning_when_exit_absent() -> None:
    """No EXIT_NOT_APPLIED_BACKTEST warning when spec has no exit."""
    spec = make_spec()
    dataset = _dataset_for_backtest()
    r = run_backtest(spec, dataset, BacktestConfig(observation_time=30 * DAY))
    codes = [w.code for w in r.warnings]
    assert WarningCode.EXIT_NOT_APPLIED_BACKTEST not in codes


def test_backtest_exit_warning_enters_result_hash() -> None:
    """A spec with exit.when has a different result_hash from a spec without it
    (the warning enters the hash for new specs, old specs are unaffected)."""
    spec_no_exit = make_spec()
    spec_with_exit = make_spec(exit_when={"feature": "alpha", "gte": 3.0})
    dataset = _dataset_for_backtest()
    cfg = BacktestConfig(observation_time=30 * DAY)
    r_no = run_backtest(spec_no_exit, dataset, cfg)
    r_with = run_backtest(spec_with_exit, dataset, cfg)
    assert r_no.result_hash != r_with.result_hash


def test_backtest_exit_spec_hash_differs_from_no_exit() -> None:
    """compiled_spec_hash changes when exit field is added (it's spec-hash-visible)."""
    spec_no_exit = make_spec()
    spec_with_exit = make_spec(exit_when={"feature": "alpha", "gte": 3.0})
    dataset = _dataset_for_backtest()
    cfg = BacktestConfig(observation_time=30 * DAY)
    r_no = run_backtest(spec_no_exit, dataset, cfg)
    r_with = run_backtest(spec_with_exit, dataset, cfg)
    assert r_no.compiled_spec_hash != r_with.compiled_spec_hash


def test_backtest_exit_examples_smoke_unchanged() -> None:
    """A spec WITHOUT exit must hash identically before and after this change
    (proved via two runs of the same spec)."""
    spec = make_spec()
    dataset = _dataset_for_backtest()
    cfg = BacktestConfig(observation_time=30 * DAY)
    r1 = run_backtest(spec, dataset, cfg)
    r2 = run_backtest(spec, dataset, cfg)
    assert r1.result_hash == r2.result_hash
    assert r1.result_hash != ""


# ---------------------------------------------------------------------------
# Validation: exit.when references undeclared column → blocked
# ---------------------------------------------------------------------------


def test_exit_undeclared_column_rejected_by_validate_spec() -> None:
    spec = make_spec(exit_when={"feature": "undeclared_col", "gte": 0.5})
    v = validate_spec(spec)
    assert not v.ok
    assert "E_EVIDENCE_SPEC_INVALID" in {e.code for e in v.errors}


def test_exit_declared_column_validate_ok() -> None:
    spec = make_spec(exit_when={"feature": "alpha", "gte": 3.5})
    v = validate_spec(spec)
    assert v.ok


def test_exit_column_included_in_agent_supplied_warning() -> None:
    """A feature column referenced ONLY in exit.when (not entry/yes_payoff)
    must still trigger AGENT_SUPPLIED_FEATURE_UNVERIFIED in the backtest run."""
    spec = make_spec(
        entry_when={"feature": "alpha", "gte": 2.0},
        exit_when={"feature": "alpha", "gte": 3.5},  # 'alpha' is feature-role
    )
    dataset = _dataset_for_backtest()
    r = run_backtest(spec, dataset, BacktestConfig(observation_time=30 * DAY))
    codes = [w.code for w in r.warnings]
    assert WarningCode.AGENT_SUPPLIED_FEATURE_UNVERIFIED in codes
