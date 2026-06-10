"""Tests for book_replay@1 fill model and SimFillRouter unification (0.9.x Wave A).

TDD contract:
- Worked example: 3 ask levels, partial consumption across 2 — hand-calc VWAP.
- No-future-book: snapshot after decision_time is not used.
- Depth-insufficient blocks the row.
- Missing slice blocks the row.
- Missing book_dataset when fill_model=book_replay@1 blocks the run.
- Hash: static_bps spec + book_dataset supplied → hash identical to without.
- Hash: book_replay spec hash differs from static_bps spec (fill_model is hashed).
- Determinism: 3-run byte-equal for book_replay runs.
- Router unification: tick suite green (via existing test_tick_fill.py); constructor
  exposes fill_model_name/fill_model_version; book_replay raises ValueError.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pancake_engine import BacktestConfig, run_backtest
from pancake_engine.fills.registry import _BookFillBlocked, _BookReplayV1, resolve
from pancake_engine.runner.fill import FillRejection, SimFillRouter
from pancake_engine.types import EvidenceDataset, EvidenceSpec
from pancake_engine.validate import validate_spec

from ._runner_helpers import make_dataset, make_spec, row

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOOK_SCHEMA_COLUMNS = [
    {"name": "market_link",    "type": "string", "semantic_role": "market_link"},
    {"name": "snapshot_time",  "type": "int",    "semantic_role": "decision_time"},
    {"name": "side",           "type": "string", "semantic_role": "feature"},
    {"name": "level_price",    "type": "number", "semantic_role": "feature"},
    {"name": "level_size",     "type": "number", "semantic_role": "feature"},
]


def make_book_dataset(
    rows: list[dict[str, Any]],
    *,
    dataset_id: str = "book_ds_test",
) -> EvidenceDataset:
    return EvidenceDataset.model_validate({
        "id": dataset_id,
        "schema": {"columns": BOOK_SCHEMA_COLUMNS},
        "schema_sha256": "0" * 64,
        "storage_mode": "inline",
        "rows_inline": rows,
        "rows_sha256": "0" * 64,
        "row_count": len(rows),
    })


def book_row(
    *,
    market_link: str,
    snapshot_time: int,
    side: str,
    level_price: float,
    level_size: float,
) -> dict[str, Any]:
    return {
        "market_link": market_link,
        "snapshot_time": snapshot_time,
        "side": side,
        "level_price": level_price,
        "level_size": level_size,
    }


def make_book_replay_spec(
    *,
    slip_bps: float = 0.0,
    fee_bps: float = 0.0,
    params: dict | None = None,
) -> EvidenceSpec:
    from tests._runner_helpers import SCHEMA_COLUMNS
    return EvidenceSpec.model_validate({
        "spec_family": "pancake-evidence-spec",
        "spec_version": "0.1",
        "name": "test-book-replay",
        "evidence_dataset_id": "ev_book_test",
        "schema_requirements": {"required_columns": SCHEMA_COLUMNS},
        "strategy": {
            "side": "YES",
            "entry": {"when": {"feature": "alpha", "gte": 2.0}},
            "yes_payoff": {"when": {"feature_equal": {"a": "target", "b": "outcome"}}},
            "sizing": {"mode": "fixed_fraction", "value": 0.1},
        },
        "costs": {
            "slippage_bps": slip_bps,
            "fee_bps": fee_bps,
            "fill_model": {
                "name": "book_replay",
                "version": 1,
                "params": params or {},
            },
        },
        "starting_capital": 1000.0,
    })


# ---------------------------------------------------------------------------
# Registry: resolve book_replay@1
# ---------------------------------------------------------------------------


def test_registry_resolves_book_replay_v1() -> None:
    model = resolve("book_replay", 1)
    assert model is not None
    assert isinstance(model, _BookReplayV1)


# ---------------------------------------------------------------------------
# Worked example: 3 ask levels, partial consumption across 2
#
# Notional = 100.0 (= 1000 * 0.1 sizing, no fee for this test)
# Levels (sorted ascending by price):
#   Level 1: price=0.50, size=100 shares  → cost = 0.50 * 100 = 50.0
#   Level 2: price=0.52, size=150 shares  → cost = 0.52 * 150 = 78.0
#   Level 3: price=0.55, size=200 shares  → cost = 0.55 * 200 = 110.0
#
# Fill 100.0 notional:
#   After level 1: spent 50.0, shares = 100, remaining = 50.0
#   Level 2: need 50.0 / 0.52 = 96.153846... shares (partial)
#   → shares_here = 50.0 / 0.52 = 96.15384615384615
#   total_shares = 100 + 96.15384615384615 = 196.15384615384615
#   VWAP = fsum([0.50*100, 0.52*96.15384615384615]) / total_shares
#         = fsum([50.0, 50.0]) / 196.15384615384615
#         = 100.0 / 196.15384615384615
#         ≈ 0.5098…
# ---------------------------------------------------------------------------

WORKED_EXAMPLE_LEVELS = [
    book_row(market_link="mkt/1", snapshot_time=90, side="ASK", level_price=0.50, level_size=100.0),
    book_row(market_link="mkt/1", snapshot_time=90, side="ASK", level_price=0.52, level_size=150.0),
    book_row(market_link="mkt/1", snapshot_time=90, side="ASK", level_price=0.55, level_size=200.0),
]


def _hand_calc_vwap() -> tuple[float, float]:
    """Hand-calculate expected fill_price and total_shares for the worked example."""
    import math as _math
    notional = 100.0
    # Level 1: consume fully (50.0 cash, 100 shares)
    shares_l1 = 100.0
    cost_l1 = 0.50 * shares_l1  # 50.0
    remaining = notional - cost_l1  # 50.0
    # Level 2: partial fill
    shares_l2 = remaining / 0.52  # 50.0 / 0.52
    cost_l2 = 0.52 * shares_l2  # = remaining exactly
    total_shares = shares_l1 + shares_l2
    vwap = _math.fsum([cost_l1, cost_l2]) / total_shares
    return vwap, total_shares


def test_book_replay_worked_example_vwap() -> None:
    """VWAP fill across 3 levels (partial consumption of level 2)."""
    model = _BookReplayV1()
    result = model.apply_entry(
        quote=0.50,
        notional=100.0,
        slippage_bps=0.0,
        fee_bps=0.0,
        market_link="mkt/1",
        decision_time=100,
        book_slices=WORKED_EXAMPLE_LEVELS,
    )
    assert not isinstance(result, _BookFillBlocked), f"Expected fill, got block: {result}"
    expected_vwap, expected_shares = _hand_calc_vwap()
    assert math.isclose(result.fill_price, expected_vwap, rel_tol=1e-12), (
        f"fill_price={result.fill_price!r} != expected={expected_vwap!r}"
    )
    assert math.isclose(result.shares, expected_shares, rel_tol=1e-12), (
        f"shares={result.shares!r} != expected={expected_shares!r}"
    )
    assert result.fee == 0.0  # fee_bps=0


def test_book_replay_worked_example_with_fee() -> None:
    """Fee is applied on notional, not on fill arithmetic."""
    model = _BookReplayV1()
    result = model.apply_entry(
        quote=0.50,
        notional=100.0,
        slippage_bps=0.0,
        fee_bps=50.0,  # 5 bps
        market_link="mkt/1",
        decision_time=100,
        book_slices=WORKED_EXAMPLE_LEVELS,
    )
    assert not isinstance(result, _BookFillBlocked)
    expected_fee = 100.0 * (50.0 / 10_000)  # 0.5
    assert math.isclose(result.fee, expected_fee, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# No-future-book: snapshot after decision_time must be ignored
# ---------------------------------------------------------------------------


def test_no_future_book_snapshot_ignored() -> None:
    """A snapshot with snapshot_time > decision_time is not used."""
    future_level = book_row(
        market_link="mkt/1", snapshot_time=200, side="ASK",
        level_price=0.50, level_size=1_000_000.0  # deep book — would fill if used
    )
    result = _BookReplayV1().apply_entry(
        quote=0.50,
        notional=100.0,
        slippage_bps=0.0,
        fee_bps=0.0,
        market_link="mkt/1",
        decision_time=100,  # snapshot_time=200 > 100 → excluded
        book_slices=[future_level],
    )
    assert isinstance(result, _BookFillBlocked)
    assert result.reason == "BOOK_SLICE_MISSING"


def test_latest_past_snapshot_is_selected() -> None:
    """When multiple snapshots <= decision_time exist, the latest is used."""
    # Snapshot at t=50: thin book (would not fill 100 notional)
    thin_levels = [
        book_row(market_link="mkt/1", snapshot_time=50, side="ASK", level_price=0.50, level_size=1.0),
    ]
    # Snapshot at t=90: deep book (fills 100 notional easily)
    deep_levels = WORKED_EXAMPLE_LEVELS  # snapshot_time=90, total depth > 200

    result = _BookReplayV1().apply_entry(
        quote=0.50,
        notional=100.0,
        slippage_bps=0.0,
        fee_bps=0.0,
        market_link="mkt/1",
        decision_time=100,
        book_slices=thin_levels + deep_levels,
    )
    # Should succeed (latest snapshot at t=90 has enough depth)
    assert not isinstance(result, _BookFillBlocked), f"Expected fill, got block: {result}"


# ---------------------------------------------------------------------------
# Depth-insufficient blocks
# ---------------------------------------------------------------------------


def test_depth_insufficient_blocks() -> None:
    """Insufficient total ask depth → E_EVIDENCE_BOOK_DEPTH_INSUFFICIENT, no partial fill."""
    shallow_book = [
        book_row(market_link="mkt/1", snapshot_time=90, side="ASK", level_price=0.50, level_size=10.0),
        # 10 shares * 0.50 = 5.0 cash — far less than 100.0 notional
    ]
    result = _BookReplayV1().apply_entry(
        quote=0.50,
        notional=100.0,
        slippage_bps=0.0,
        fee_bps=0.0,
        market_link="mkt/1",
        decision_time=100,
        book_slices=shallow_book,
    )
    assert isinstance(result, _BookFillBlocked)
    assert result.reason == "BOOK_DEPTH_INSUFFICIENT"
    assert result.context["notional"] == 100.0


# ---------------------------------------------------------------------------
# Missing slice blocks
# ---------------------------------------------------------------------------


def test_missing_slice_blocks_wrong_market() -> None:
    """Snapshots for a different market_link → BOOK_SLICE_MISSING."""
    result = _BookReplayV1().apply_entry(
        quote=0.50,
        notional=100.0,
        slippage_bps=0.0,
        fee_bps=0.0,
        market_link="mkt/WRONG",
        decision_time=100,
        book_slices=WORKED_EXAMPLE_LEVELS,  # all for mkt/1
    )
    assert isinstance(result, _BookFillBlocked)
    assert result.reason == "BOOK_SLICE_MISSING"


def test_missing_slice_blocks_empty_book_slices() -> None:
    """Empty book_slices → BOOK_SLICE_MISSING."""
    result = _BookReplayV1().apply_entry(
        quote=0.50,
        notional=100.0,
        slippage_bps=0.0,
        fee_bps=0.0,
        market_link="mkt/1",
        decision_time=100,
        book_slices=[],
    )
    assert isinstance(result, _BookFillBlocked)
    assert result.reason == "BOOK_SLICE_MISSING"


def test_missing_slice_blocks_none_book_slices() -> None:
    """book_slices=None → BOOK_SLICE_MISSING."""
    result = _BookReplayV1().apply_entry(
        quote=0.50,
        notional=100.0,
        slippage_bps=0.0,
        fee_bps=0.0,
        market_link="mkt/1",
        decision_time=100,
        book_slices=None,
    )
    assert isinstance(result, _BookFillBlocked)
    assert result.reason == "BOOK_SLICE_MISSING"


def test_bid_side_ignored_counts_as_missing() -> None:
    """BID rows for the right market → no ASK levels → BOOK_SLICE_MISSING."""
    bid_levels = [
        book_row(market_link="mkt/1", snapshot_time=90, side="BID",
                 level_price=0.48, level_size=1000.0),
    ]
    result = _BookReplayV1().apply_entry(
        quote=0.50,
        notional=100.0,
        slippage_bps=0.0,
        fee_bps=0.0,
        market_link="mkt/1",
        decision_time=100,
        book_slices=bid_levels,
    )
    assert isinstance(result, _BookFillBlocked)
    assert result.reason == "BOOK_SLICE_MISSING"


# ---------------------------------------------------------------------------
# Missing book_dataset blocks the run
# ---------------------------------------------------------------------------


def test_missing_book_dataset_blocks_run() -> None:
    """fill_model=book_replay@1 + no book_dataset → blocked result."""
    spec = make_book_replay_spec()
    dataset = make_dataset([
        row(mkt="mkt/1", dec_ts=100, res_ts=200, price=0.6, outcome=1),
    ])
    config = BacktestConfig(observation_time=500)

    result = run_backtest(spec, dataset, config)  # no book_dataset
    assert result.result_hash == ""  # blocked
    assert not result.validation.ok
    codes = {e.code for e in result.validation.errors}
    assert "E_EVIDENCE_BOOK_DATASET_REQUIRED" in codes


# ---------------------------------------------------------------------------
# Hash: static_bps spec + book_dataset supplied → hash identical to without
# ---------------------------------------------------------------------------


def test_static_bps_hash_unchanged_with_book_dataset_supplied() -> None:
    """A static_bps spec with a book_dataset kwarg produces the SAME hash as without.

    The book payload is only injected into the hash when book_replay ran.
    static_bps never triggers that path → hashes are byte-identical.
    """
    spec = make_spec(slip_bps=5.0, fee_bps=2.0)
    dataset = make_dataset([
        row(mkt="mkt/1", dec_ts=100, res_ts=200, price=0.6, outcome=1),
        row(mkt="mkt/2", dec_ts=300, res_ts=400, price=0.4, outcome=0),
    ])
    config = BacktestConfig(observation_time=500)
    book_ds = make_book_dataset(WORKED_EXAMPLE_LEVELS)

    r_without = run_backtest(spec, dataset, config)
    r_with = run_backtest(spec, dataset, config, book_dataset=book_ds)

    # Both succeed
    assert r_without.result_hash != ""
    assert r_with.result_hash != ""
    # Hashes are byte-identical
    assert r_without.result_hash == r_with.result_hash
    # book provenance fields are None when not used
    assert r_with.book_dataset_id is None
    assert r_with.book_rows_sha256 is None


# ---------------------------------------------------------------------------
# Hash: book_replay spec hash differs from static_bps spec
# ---------------------------------------------------------------------------


def test_book_replay_spec_hash_differs_from_static_bps() -> None:
    """book_replay@1 spec → different compiled_spec_hash (fill_model field present)."""
    spec_static = make_spec(slip_bps=0.0, fee_bps=0.0)
    spec_book = make_book_replay_spec(slip_bps=0.0, fee_bps=0.0)

    dataset = make_dataset([
        row(mkt="mkt/1", dec_ts=100, res_ts=200, price=0.55, outcome=1),
    ])
    config = BacktestConfig(observation_time=500)
    # Deep enough book for 100 notional: 1000 * 0.1 sizing = 100 cash
    book_levels = [
        book_row(market_link="mkt/1", snapshot_time=90, side="ASK",
                 level_price=0.55, level_size=200.0),  # 200 * 0.55 = 110 > 100
    ]
    book_ds = make_book_dataset(book_levels)

    r_static = run_backtest(spec_static, dataset, config)
    r_book = run_backtest(spec_book, dataset, config, book_dataset=book_ds)

    # Both succeed
    assert r_static.result_hash != ""
    assert r_book.result_hash != ""
    # compiled_spec_hash and result_hash differ (fill_model is in the spec)
    assert r_static.compiled_spec_hash != r_book.compiled_spec_hash
    assert r_static.result_hash != r_book.result_hash
    # book provenance fields are set on the book_replay result
    assert r_book.book_dataset_id == "book_ds_test"
    assert r_book.book_rows_sha256 is not None
    assert r_book.book_schema_sha256 is not None


# ---------------------------------------------------------------------------
# Determinism: 3 identical runs → byte-equal result_hash
# ---------------------------------------------------------------------------


def test_book_replay_run_deterministic() -> None:
    """Three identical book_replay@1 runs produce the same result_hash."""
    spec = make_book_replay_spec()
    dataset = make_dataset([
        row(mkt="mkt/1", dec_ts=100, res_ts=200, price=0.55, outcome=1),
        row(mkt="mkt/2", dec_ts=300, res_ts=400, price=0.60, outcome=0),
    ])
    config = BacktestConfig(observation_time=500)
    book_levels = [
        book_row(market_link="mkt/1", snapshot_time=90, side="ASK",
                 level_price=0.55, level_size=500.0),
        book_row(market_link="mkt/2", snapshot_time=280, side="ASK",
                 level_price=0.60, level_size=500.0),
    ]
    book_ds = make_book_dataset(book_levels)

    hashes = [
        run_backtest(spec, dataset, config, book_dataset=book_ds).result_hash
        for _ in range(3)
    ]
    assert len(set(hashes)) == 1, f"Non-deterministic: {hashes}"
    assert hashes[0] != ""  # sanity: run succeeded


# ---------------------------------------------------------------------------
# Level ordering is deterministic (explicit sort, not dict order)
# ---------------------------------------------------------------------------


def test_book_replay_deterministic_level_order() -> None:
    """Shuffled level order → same fill (sort by (price, size) is enforced)."""
    model = _BookReplayV1()
    levels_asc = WORKED_EXAMPLE_LEVELS  # already sorted ascending
    levels_rev = list(reversed(WORKED_EXAMPLE_LEVELS))

    r_asc = model.apply_entry(
        quote=0.50, notional=100.0, slippage_bps=0.0, fee_bps=0.0,
        market_link="mkt/1", decision_time=100, book_slices=levels_asc,
    )
    r_rev = model.apply_entry(
        quote=0.50, notional=100.0, slippage_bps=0.0, fee_bps=0.0,
        market_link="mkt/1", decision_time=100, book_slices=levels_rev,
    )

    assert not isinstance(r_asc, _BookFillBlocked)
    assert not isinstance(r_rev, _BookFillBlocked)
    assert r_asc.fill_price == r_rev.fill_price
    assert r_asc.shares == r_rev.shares


# ---------------------------------------------------------------------------
# TTR param validation: reserved → E_EVIDENCE_SPEC_INVALID
# ---------------------------------------------------------------------------


def test_ttr_fill_adjustment_param_rejected() -> None:
    """params={"ttr_fill_adjustment": true} → E_EVIDENCE_SPEC_INVALID."""
    spec = make_book_replay_spec(params={"ttr_fill_adjustment": True})
    verdict = validate_spec(spec)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_SPEC_INVALID" in codes
    # Message must mention 'reserved'
    messages = " ".join(e.message for e in verdict.errors)
    assert "reserved" in messages


def test_book_replay_no_params_valid() -> None:
    """book_replay@1 with empty params passes validate_spec."""
    spec = make_book_replay_spec(params={})
    verdict = validate_spec(spec)
    assert verdict.ok


# ---------------------------------------------------------------------------
# SimFillRouter unification
# ---------------------------------------------------------------------------


def test_simfillrouter_default_model_is_static_bps() -> None:
    """Default construction uses static_bps@1 — math unchanged."""
    import math as _math
    r = SimFillRouter(slippage_bps=50.0, fee_bps=25.0)
    f = r.fill(side="YES", yes_close=0.4, available_cash=1000.0, sizing_value=0.1)
    assert not isinstance(f, FillRejection)
    # static_bps@1: fill_price = 0.4 * (1 + 50/10000) = 0.402
    assert _math.isclose(f.fill_price, 0.4 * (1.0 + 50.0 / 10_000), rel_tol=1e-12)


def test_simfillrouter_explicit_static_bps_v1() -> None:
    """Explicit fill_model_name=static_bps, version=1 → same result as default."""
    r_default = SimFillRouter(slippage_bps=50.0, fee_bps=25.0)
    r_explicit = SimFillRouter(
        slippage_bps=50.0, fee_bps=25.0,
        fill_model_name="static_bps", fill_model_version=1,
    )
    f_default = r_default.fill(side="YES", yes_close=0.4, available_cash=1000.0, sizing_value=0.1)
    f_explicit = r_explicit.fill(side="YES", yes_close=0.4, available_cash=1000.0, sizing_value=0.1)
    assert not isinstance(f_default, FillRejection)
    assert not isinstance(f_explicit, FillRejection)
    assert f_default.fill_price == f_explicit.fill_price
    assert f_default.shares == f_explicit.shares


def test_simfillrouter_book_replay_raises_valueerror() -> None:
    """SimFillRouter does not support book_replay@1 (no L2 data at tick time)."""
    with pytest.raises(ValueError, match="book_replay"):
        SimFillRouter(
            slippage_bps=0.0, fee_bps=0.0,
            fill_model_name="book_replay", fill_model_version=1,
        )


def test_simfillrouter_unknown_model_raises() -> None:
    """Unknown fill model name raises ValueError at construction."""
    with pytest.raises(ValueError, match="E_EVIDENCE_SPEC_INVALID"):
        SimFillRouter(
            slippage_bps=0.0, fee_bps=0.0,
            fill_model_name="nonexistent", fill_model_version=1,
        )


# ---------------------------------------------------------------------------
# Integration: full run_backtest with book_replay@1, warning codes, fill math
# ---------------------------------------------------------------------------


def test_book_replay_full_run_success() -> None:
    """Full run with book_replay@1: trade executed, provenance fields set."""
    spec = make_book_replay_spec(fee_bps=50.0)
    dataset = make_dataset([
        row(mkt="mkt/1", dec_ts=100, res_ts=200, price=0.55, outcome=1),
    ])
    config = BacktestConfig(observation_time=500)
    # Single deep ask level: fills 100 notional easily (1000 * 0.1 = 100)
    book_levels = [
        book_row(market_link="mkt/1", snapshot_time=90, side="ASK",
                 level_price=0.55, level_size=300.0),
    ]
    book_ds = make_book_dataset(book_levels)

    result = run_backtest(spec, dataset, config, book_dataset=book_ds)

    assert result.result_hash != "", f"Run blocked: {result.validation.errors}"
    assert len(result.trades) == 1
    trade = result.trades[0]
    # fill_price = VWAP of 100 notional at 0.55/share
    # 100 notional / 0.55 = 181.818... shares; VWAP = 0.55
    assert math.isclose(trade.entry_price, 0.55, rel_tol=1e-12)
    assert result.book_dataset_id == "book_ds_test"
    assert result.book_rows_sha256 is not None


def test_book_replay_slice_missing_warning_in_result() -> None:
    """When a row has no slice, BOOK_SLICE_MISSING warning appears and row is skipped."""
    spec = make_book_replay_spec()
    dataset = make_dataset([
        row(mkt="mkt/1", dec_ts=100, res_ts=200, price=0.55, outcome=1),
        row(mkt="mkt/2", dec_ts=150, res_ts=250, price=0.60, outcome=1),  # no slice for mkt/2
    ])
    config = BacktestConfig(observation_time=500)
    # Only provide slice for mkt/1
    book_levels = [
        book_row(market_link="mkt/1", snapshot_time=90, side="ASK",
                 level_price=0.55, level_size=300.0),
    ]
    book_ds = make_book_dataset(book_levels)

    result = run_backtest(spec, dataset, config, book_dataset=book_ds)

    assert result.result_hash != ""
    assert len(result.trades) == 1  # only mkt/1 filled
    warning_codes = {w.code.value for w in result.warnings}
    assert "BOOK_SLICE_MISSING" in warning_codes


def test_book_replay_depth_insufficient_warning_in_result() -> None:
    """Insufficient depth for a row → BOOK_DEPTH_INSUFFICIENT warning, row skipped."""
    spec = make_book_replay_spec()
    dataset = make_dataset([
        row(mkt="mkt/1", dec_ts=100, res_ts=200, price=0.55, outcome=1),
    ])
    config = BacktestConfig(observation_time=500)
    # Shallow book: only 1 share available × 0.55 = 0.55 cash; need 100
    book_levels = [
        book_row(market_link="mkt/1", snapshot_time=90, side="ASK",
                 level_price=0.55, level_size=1.0),
    ]
    book_ds = make_book_dataset(book_levels)

    result = run_backtest(spec, dataset, config, book_dataset=book_ds)

    assert result.result_hash != ""
    assert len(result.trades) == 0  # skipped
    warning_codes = {w.code.value for w in result.warnings}
    assert "BOOK_DEPTH_INSUFFICIENT" in warning_codes
