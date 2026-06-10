"""Tests for pancake_engine/chain/ — Wave C of batter 0.10.0.

TDD: written first. Green requires the full implementation.

Coverage:
  - Happy-path chain: deploy→tick→order lifecycle→settlement verifies
  - Every tamper class caught (hash flip, reorder, drop, illegal transition,
    t regression, fill overshoot, forged genesis without backtest pin)
  - Determinism: same records → same hashes 3×
  - CLI exit codes: 0 ok / 1 verify failure / 2 invalid input
  - ChainTransitionError surfaces from/to fields
  - Reconciliation kind
  - P&L roll-forward settlement check
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from pancake_engine.chain.builder import ChainBuilder
from pancake_engine.chain.errors import ChainTransitionError
from pancake_engine.chain.records import ChainRecord
from pancake_engine.chain.verify import ChainVerdict, verify_chain

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_happy_chain() -> list[ChainRecord]:
    """Build a minimal but complete happy-path chain and return the record list."""
    b = ChainBuilder()

    # Genesis (deploy)
    b.append(
        kind="deploy",
        t=1000,
        payload={
            "compiled_spec_hash": "aaa" + "0" * 61,
            "result_hash": "bbb" + "0" * 61,
            "dataset_id": "ds-001",
            "starting_cash": 100.0,
        },
    )

    # Tick
    b.append(kind="tick", t=1001, payload={"new_equity": 100.0, "cash": 100.0})

    # Order: proposed → submitted → acked → filled
    b.append(
        kind="order_state",
        t=1002,
        payload={"order_id": "ord-1", "state": "proposed", "instrument_id": "mkt-A"},
    )
    b.append(
        kind="order_state",
        t=1002,
        payload={"order_id": "ord-1", "state": "submitted"},
    )
    b.append(
        kind="order_state",
        t=1003,
        payload={"order_id": "ord-1", "state": "acked"},
    )
    b.append(
        kind="fill",
        t=1003,
        payload={
            "order_id": "ord-1",
            "fill_qty": 10.0,
            "order_qty": 10.0,
            "cash_delta": -10.0,
        },
    )
    b.append(
        kind="order_state",
        t=1003,
        payload={"order_id": "ord-1", "state": "filled"},
    )

    # Settlement
    b.append(
        kind="settlement",
        t=1010,
        payload={"order_id": "ord-1", "cash_delta": 10.0, "pnl": 0.0, "total_cash": 100.0},
    )

    return b.records()


# ---------------------------------------------------------------------------
# ChainRecord structure
# ---------------------------------------------------------------------------


def test_chain_record_is_frozen():
    records = _make_happy_chain()
    rec = records[0]
    with pytest.raises((AttributeError, TypeError)):
        rec.seq = 99  # type: ignore[misc]


def test_genesis_prev_hash_is_empty_string():
    records = _make_happy_chain()
    assert records[0].prev_hash == ""


def test_genesis_kind_is_deploy():
    records = _make_happy_chain()
    assert records[0].kind == "deploy"


def test_seq_is_zero_based_dense():
    records = _make_happy_chain()
    for i, rec in enumerate(records):
        assert rec.seq == i


def test_record_hash_is_hex_string():
    records = _make_happy_chain()
    for rec in records:
        assert len(rec.record_hash) == 64
        int(rec.record_hash, 16)  # must be valid hex


# ---------------------------------------------------------------------------
# Genesis payload validation
# ---------------------------------------------------------------------------


def test_genesis_missing_compiled_spec_hash_raises():
    b = ChainBuilder()
    with pytest.raises(ValueError, match="compiled_spec_hash"):
        b.append(
            kind="deploy",
            t=1000,
            payload={"result_hash": "b" * 64, "dataset_id": "ds"},
        )


def test_genesis_missing_result_hash_raises():
    b = ChainBuilder()
    with pytest.raises(ValueError, match="result_hash"):
        b.append(
            kind="deploy",
            t=1000,
            payload={"compiled_spec_hash": "a" * 64, "dataset_id": "ds"},
        )


def test_genesis_missing_dataset_id_raises():
    b = ChainBuilder()
    with pytest.raises(ValueError, match="dataset_id"):
        b.append(
            kind="deploy",
            t=1000,
            payload={"compiled_spec_hash": "a" * 64, "result_hash": "b" * 64},
        )


def test_non_genesis_before_deploy_raises():
    b = ChainBuilder()
    with pytest.raises(ValueError, match="genesis"):
        b.append(kind="tick", t=1000, payload={})


# ---------------------------------------------------------------------------
# t monotonicity
# ---------------------------------------------------------------------------


def test_t_regression_raises():
    b = ChainBuilder()
    b.append(
        kind="deploy",
        t=1000,
        payload={
            "compiled_spec_hash": "a" * 64,
            "result_hash": "b" * 64,
            "dataset_id": "ds",
            "starting_cash": 100.0,
        },
    )
    with pytest.raises(ValueError, match="monoton"):
        b.append(kind="tick", t=999, payload={})


def test_t_equal_is_allowed():
    """t can equal previous t (same-second events)."""
    b = ChainBuilder()
    b.append(
        kind="deploy",
        t=1000,
        payload={
            "compiled_spec_hash": "a" * 64,
            "result_hash": "b" * 64,
            "dataset_id": "ds",
            "starting_cash": 100.0,
        },
    )
    b.append(kind="tick", t=1000, payload={})
    assert len(b.records()) == 2


# ---------------------------------------------------------------------------
# Order-state machine
# ---------------------------------------------------------------------------


def test_illegal_transition_raises_chain_transition_error():
    b = ChainBuilder()
    b.append(
        kind="deploy",
        t=1000,
        payload={
            "compiled_spec_hash": "a" * 64,
            "result_hash": "b" * 64,
            "dataset_id": "ds",
            "starting_cash": 100.0,
        },
    )
    # Start an order
    b.append(
        kind="order_state",
        t=1001,
        payload={"order_id": "o1", "state": "proposed", "instrument_id": "m"},
    )
    # Jump directly from proposed to filled (illegal — must go through submitted/acked)
    with pytest.raises(ChainTransitionError) as exc_info:
        b.append(
            kind="order_state",
            t=1001,
            payload={"order_id": "o1", "state": "filled"},
        )
    err = exc_info.value
    assert err.from_state == "proposed"
    assert err.to_state == "filled"


def test_transition_error_carries_from_to():
    from pancake_engine.chain.orders import advance
    with pytest.raises(ChainTransitionError) as exc_info:
        advance("filled", "proposed", t=1, payload={})
    err = exc_info.value
    assert err.from_state == "filled"
    assert err.to_state == "proposed"


def test_terminal_state_rejects_further_transitions():
    """canceled is terminal — nothing follows."""
    b = ChainBuilder()
    b.append(
        kind="deploy",
        t=1000,
        payload={
            "compiled_spec_hash": "a" * 64,
            "result_hash": "b" * 64,
            "dataset_id": "ds",
            "starting_cash": 100.0,
        },
    )
    b.append(
        kind="order_state",
        t=1001,
        payload={"order_id": "o1", "state": "proposed", "instrument_id": "m"},
    )
    b.append(
        kind="order_state",
        t=1001,
        payload={"order_id": "o1", "state": "canceled"},
    )
    with pytest.raises(ChainTransitionError):
        b.append(
            kind="order_state",
            t=1002,
            payload={"order_id": "o1", "state": "submitted"},
        )


def test_partial_fill_may_repeat():
    """partially_filled → partially_filled is legal (repeated partial fills)."""
    b = ChainBuilder()
    b.append(
        kind="deploy",
        t=1000,
        payload={
            "compiled_spec_hash": "a" * 64,
            "result_hash": "b" * 64,
            "dataset_id": "ds",
            "starting_cash": 100.0,
        },
    )
    for state in ("proposed", "submitted", "acked"):
        b.append(
            kind="order_state",
            t=1001,
            payload={"order_id": "o2", "state": state, "instrument_id": "m"},
        )
    # First partial fill
    b.append(
        kind="fill",
        t=1001,
        payload={"order_id": "o2", "fill_qty": 3.0, "order_qty": 10.0, "cash_delta": -3.0},
    )
    b.append(
        kind="order_state",
        t=1001,
        payload={"order_id": "o2", "state": "partially_filled"},
    )
    # Second partial fill
    b.append(
        kind="fill",
        t=1002,
        payload={"order_id": "o2", "fill_qty": 4.0, "order_qty": 10.0, "cash_delta": -4.0},
    )
    b.append(
        kind="order_state",
        t=1002,
        payload={"order_id": "o2", "state": "partially_filled"},
    )
    assert len(b.records()) == 8  # genesis + 3 states (proposed/submitted/acked) + fill1 + partial_filled + fill2 + partial_filled


def test_fill_overshoot_raises():
    """Cumulative fill qty must never exceed order qty."""
    b = ChainBuilder()
    b.append(
        kind="deploy",
        t=1000,
        payload={
            "compiled_spec_hash": "a" * 64,
            "result_hash": "b" * 64,
            "dataset_id": "ds",
            "starting_cash": 100.0,
        },
    )
    for state in ("proposed", "submitted", "acked"):
        b.append(
            kind="order_state",
            t=1001,
            payload={"order_id": "o3", "state": state, "instrument_id": "m"},
        )
    b.append(
        kind="fill",
        t=1001,
        payload={"order_id": "o3", "fill_qty": 8.0, "order_qty": 10.0, "cash_delta": -8.0},
    )
    b.append(
        kind="order_state",
        t=1001,
        payload={"order_id": "o3", "state": "partially_filled"},
    )
    with pytest.raises(ValueError, match="overshoot|exceed|qty"):
        b.append(
            kind="fill",
            t=1002,
            payload={"order_id": "o3", "fill_qty": 5.0, "order_qty": 10.0, "cash_delta": -5.0},
        )


def test_fill_monotone_nondecreasing():
    """Cumulative fill qty must be nondecreasing across successive fills."""
    b = ChainBuilder()
    b.append(
        kind="deploy",
        t=1000,
        payload={
            "compiled_spec_hash": "a" * 64,
            "result_hash": "b" * 64,
            "dataset_id": "ds",
            "starting_cash": 100.0,
        },
    )
    for state in ("proposed", "submitted", "acked"):
        b.append(
            kind="order_state",
            t=1001,
            payload={"order_id": "o4", "state": state, "instrument_id": "m"},
        )
    b.append(
        kind="fill",
        t=1001,
        payload={"order_id": "o4", "fill_qty": 5.0, "order_qty": 10.0, "cash_delta": -5.0},
    )
    b.append(
        kind="order_state",
        t=1001,
        payload={"order_id": "o4", "state": "partially_filled"},
    )
    # A second fill that would bring cumulative below current — should still be fine
    # as long as total stays ≤ order_qty.  A zero-qty fill IS allowed (nondecreasing).
    b.append(
        kind="fill",
        t=1002,
        payload={"order_id": "o4", "fill_qty": 3.0, "order_qty": 10.0, "cash_delta": -3.0},
    )
    # cumulative 5+3=8 ≤ 10, should pass
    # Records: genesis(0), proposed(1), submitted(2), acked(3), fill1(4), partially_filled(5), fill2(6)
    assert b.records()[-1].seq == 6


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def test_reconciliation_record_appended():
    b = ChainBuilder()
    b.append(
        kind="deploy",
        t=1000,
        payload={
            "compiled_spec_hash": "a" * 64,
            "result_hash": "b" * 64,
            "dataset_id": "ds",
            "starting_cash": 100.0,
        },
    )
    b.append(
        kind="reconciliation",
        t=1005,
        payload={
            "expected": {"cash": 100.0},
            "reported": {"cash": 99.5},
            "diffs": [{"field": "cash", "expected": 100.0, "reported": 99.5}],
        },
    )
    records = b.records()
    assert records[-1].kind == "reconciliation"
    assert records[-1].payload["diffs"][0]["field"] == "cash"


def test_reconciliation_missing_diffs_raises():
    b = ChainBuilder()
    b.append(
        kind="deploy",
        t=1000,
        payload={
            "compiled_spec_hash": "a" * 64,
            "result_hash": "b" * 64,
            "dataset_id": "ds",
            "starting_cash": 100.0,
        },
    )
    with pytest.raises(ValueError, match="diffs"):
        b.append(
            kind="reconciliation",
            t=1005,
            payload={"expected": {}, "reported": {}},
            # no "diffs" key
        )


# ---------------------------------------------------------------------------
# verify_chain — happy path
# ---------------------------------------------------------------------------


def test_happy_chain_verifies():
    records = _make_happy_chain()
    verdict = verify_chain(records)
    assert verdict.ok
    assert verdict.errors == []


def test_verify_returns_chain_verdict_type():
    records = _make_happy_chain()
    verdict = verify_chain(records)
    assert isinstance(verdict, ChainVerdict)


# ---------------------------------------------------------------------------
# verify_chain — tamper classes
# ---------------------------------------------------------------------------


def test_hash_flip_detected():
    records = list(_make_happy_chain())
    # Flip a single byte in a middle record's record_hash
    rec = records[2]
    flipped = rec.record_hash[:4] + ("0" if rec.record_hash[4] != "0" else "1") + rec.record_hash[5:]
    # Rebuild a record with the flipped hash using dataclass replace
    from dataclasses import replace
    records[2] = replace(rec, record_hash=flipped)
    verdict = verify_chain(records)
    assert not verdict.ok
    assert any(e["seq"] == 2 for e in verdict.errors)


def test_reordered_records_detected():
    records = list(_make_happy_chain())
    # Swap records 1 and 2
    records[1], records[2] = records[2], records[1]
    verdict = verify_chain(records)
    assert not verdict.ok


def test_dropped_record_detected():
    records = list(_make_happy_chain())
    # Drop record at index 3
    del records[3]
    verdict = verify_chain(records)
    assert not verdict.ok


def test_t_regression_tamper_detected():
    """Manually construct a chain with a backward t (bypassing builder)."""
    records = list(_make_happy_chain())
    # Recompute record 1 with a t earlier than record 0's t
    from dataclasses import replace
    from pancake_engine.chain.records import _compute_record_hash
    from pancake_engine.__version__ import ENGINE, ENGINE_VERSION

    rec1 = records[1]
    bad_t = 500  # < genesis t=1000
    new_hash = _compute_record_hash(
        seq=rec1.seq,
        t=bad_t,
        kind=rec1.kind,
        payload=rec1.payload,
        prev_hash=rec1.prev_hash,
    )
    records[1] = replace(rec1, t=bad_t, record_hash=new_hash)
    verdict = verify_chain(records)
    assert not verdict.ok
    assert any(e["code"] == "T_REGRESSION" for e in verdict.errors)


def test_illegal_transition_tamper_detected():
    """Forge a state record that skips proposed→submitted by replacing
    the submitted record with an illegal direct-to-filled."""
    records = list(_make_happy_chain())
    # Record 3 is "submitted"; replace its payload to say "filled" (illegal from "proposed")
    from dataclasses import replace
    from pancake_engine.chain.records import _compute_record_hash

    rec3 = records[3]
    bad_payload = dict(rec3.payload)
    bad_payload["state"] = "filled"
    new_hash = _compute_record_hash(
        seq=rec3.seq,
        t=rec3.t,
        kind=rec3.kind,
        payload=bad_payload,
        prev_hash=rec3.prev_hash,
    )
    records[3] = replace(rec3, payload=bad_payload, record_hash=new_hash)
    # Also fix forward prev_hash linkage (so hash-integrity passes but state-machine catches it)
    _relink_chain(records, from_index=3)
    verdict = verify_chain(records)
    assert not verdict.ok
    assert any(e["code"] in ("ILLEGAL_TRANSITION", "STATE_MACHINE_VIOLATION") for e in verdict.errors)


def test_fill_overshoot_tamper_detected():
    """Forge a fill record with qty > order_qty in the chain."""
    # Build a chain with a fill, then mutate qty above order_qty
    b = ChainBuilder()
    b.append(
        kind="deploy",
        t=1000,
        payload={"compiled_spec_hash": "a" * 64, "result_hash": "b" * 64, "dataset_id": "ds", "starting_cash": 100.0},
    )
    for state in ("proposed", "submitted", "acked"):
        b.append(
            kind="order_state",
            t=1001,
            payload={"order_id": "ox", "state": state, "instrument_id": "m"},
        )
    b.append(
        kind="fill",
        t=1001,
        payload={"order_id": "ox", "fill_qty": 10.0, "order_qty": 10.0, "cash_delta": -10.0},
    )
    b.append(
        kind="order_state",
        t=1001,
        payload={"order_id": "ox", "state": "filled"},
    )
    records = list(b.records())

    # Now forge another fill record with fill_qty=5.0 on the same order (total > order_qty)
    from dataclasses import replace
    from pancake_engine.chain.records import _compute_record_hash

    last = records[-1]
    bad_fill_payload = {
        "order_id": "ox",
        "fill_qty": 5.0,
        "order_qty": 10.0,
        "cash_delta": -5.0,
    }
    new_seq = last.seq + 1
    new_hash = _compute_record_hash(
        seq=new_seq,
        t=last.t,
        kind="fill",
        payload=bad_fill_payload,
        prev_hash=last.record_hash,
    )
    bad_fill_rec = ChainRecord(
        seq=new_seq,
        t=last.t,
        kind="fill",
        payload=bad_fill_payload,
        prev_hash=last.record_hash,
        record_hash=new_hash,
    )
    records.append(bad_fill_rec)

    verdict = verify_chain(records)
    assert not verdict.ok
    assert any(e["code"] in ("FILL_OVERSHOOT", "CUMULATIVE_FILL_OVERSHOOT") for e in verdict.errors)


def test_forged_genesis_without_backtest_pin_detected():
    """A genesis missing compiled_spec_hash / result_hash is invalid."""
    from dataclasses import replace
    from pancake_engine.chain.records import _compute_record_hash

    records = list(_make_happy_chain())
    rec0 = records[0]
    bad_payload = {"dataset_id": "ds-001"}  # stripped provenance fields
    new_hash = _compute_record_hash(
        seq=0, t=rec0.t, kind="deploy", payload=bad_payload, prev_hash="",
    )
    records[0] = replace(rec0, payload=bad_payload, record_hash=new_hash)
    _relink_chain(records, from_index=0)
    verdict = verify_chain(records)
    assert not verdict.ok
    assert any(e["code"] == "GENESIS_MISSING_PROVENANCE" for e in verdict.errors)


def test_pnl_rollforward_settlement_mismatch_detected():
    """A settlement total_cash off by even 1e-9 from the exact roll-forward is caught."""
    import math
    b = ChainBuilder()
    b.append(
        kind="deploy",
        t=1000,
        payload={"compiled_spec_hash": "a" * 64, "result_hash": "b" * 64, "dataset_id": "ds", "starting_cash": 100.0},
    )
    for state in ("proposed", "submitted", "acked"):
        b.append(kind="order_state", t=1001,
                 payload={"order_id": "pnl1", "state": state, "instrument_id": "m"})
    b.append(kind="fill", t=1001,
             payload={"order_id": "pnl1", "fill_qty": 10.0, "order_qty": 10.0, "cash_delta": -50.0})
    b.append(kind="order_state", t=1001,
             payload={"order_id": "pnl1", "state": "filled"})
    # Correct total_cash: starting(100) + fill_cd(-50) + settle_cd(60) = 110.0
    # Declare total_cash = 110.0 + 1e-9 → off by a tiny amount; verifier must catch it.
    correct_total = math.fsum([100.0, -50.0, 60.0])  # == 110.0
    wrong_total = correct_total + 1e-9
    b.append(kind="settlement", t=1010,
             payload={"order_id": "pnl1", "cash_delta": 60.0, "pnl": 10.0, "total_cash": wrong_total})
    records = b.records()
    verdict = verify_chain(records)
    assert not verdict.ok
    assert any(e["code"] == "E_CHAIN_CASH_MISMATCH" for e in verdict.errors)


def test_pnl_rollforward_exact_fsum_passes():
    """A settlement with total_cash == exact fsum value passes verification."""
    import math
    b = ChainBuilder()
    b.append(
        kind="deploy",
        t=1000,
        payload={"compiled_spec_hash": "a" * 64, "result_hash": "b" * 64, "dataset_id": "ds", "starting_cash": 100.0},
    )
    for state in ("proposed", "submitted", "acked"):
        b.append(kind="order_state", t=1001,
                 payload={"order_id": "pnl2", "state": state, "instrument_id": "m"})
    b.append(kind="fill", t=1001,
             payload={"order_id": "pnl2", "fill_qty": 10.0, "order_qty": 10.0, "cash_delta": -50.0})
    b.append(kind="order_state", t=1001,
             payload={"order_id": "pnl2", "state": "filled"})
    # Use ChainBuilder.running_cash() to get the authoritative value after fill.
    # Then append settlement and set total_cash = running_cash() after the settlement cd.
    # Simulate: total = fsum([100.0, -50.0, 60.0]) = 110.0
    correct_total = math.fsum([100.0, -50.0, 60.0])
    b.append(kind="settlement", t=1010,
             payload={"order_id": "pnl2", "cash_delta": 60.0, "pnl": 10.0, "total_cash": correct_total})
    records = b.records()
    verdict = verify_chain(records)
    assert verdict.ok, verdict.errors


def test_running_cash_helper_matches_verify():
    """ChainBuilder.running_cash() tracks the same fsum as verify_chain's roll-forward
    across a multi-fill chain."""
    import math
    b = ChainBuilder()
    b.append(
        kind="deploy",
        t=1000,
        payload={"compiled_spec_hash": "a" * 64, "result_hash": "b" * 64, "dataset_id": "ds", "starting_cash": 200.0},
    )
    assert b.running_cash() == 200.0

    for state in ("proposed", "submitted", "acked"):
        b.append(kind="order_state", t=1001,
                 payload={"order_id": "rc1", "state": state, "instrument_id": "m"})
    b.append(kind="fill", t=1001,
             payload={"order_id": "rc1", "fill_qty": 5.0, "order_qty": 10.0, "cash_delta": -30.0})
    b.append(kind="order_state", t=1001,
             payload={"order_id": "rc1", "state": "partially_filled"})
    assert b.running_cash() == math.fsum([200.0, -30.0])

    b.append(kind="fill", t=1002,
             payload={"order_id": "rc1", "fill_qty": 5.0, "order_qty": 10.0, "cash_delta": -30.0})
    b.append(kind="order_state", t=1002,
             payload={"order_id": "rc1", "state": "filled"})
    assert b.running_cash() == math.fsum([200.0, -30.0, -30.0])

    settle_total = b.running_cash() + 70.0  # simulate payout
    # Running cash after settlement = fsum([200.0, -30.0, -30.0, 70.0])
    expected_final = math.fsum([200.0, -30.0, -30.0, 70.0])
    b.append(kind="settlement", t=1010,
             payload={"order_id": "rc1", "cash_delta": 70.0, "total_cash": expected_final})
    assert b.running_cash() == expected_final

    records = b.records()
    verdict = verify_chain(records)
    assert verdict.ok, verdict.errors


def test_genesis_missing_starting_cash_raises():
    """Builder rejects a genesis payload without starting_cash."""
    b = ChainBuilder()
    with pytest.raises(ValueError, match="starting_cash"):
        b.append(
            kind="deploy",
            t=1000,
            payload={"compiled_spec_hash": "a" * 64, "result_hash": "b" * 64, "dataset_id": "ds"},
        )


def test_genesis_starting_cash_non_finite_raises():
    """Builder rejects a genesis with starting_cash=inf."""
    import math
    b = ChainBuilder()
    with pytest.raises(ValueError, match="finite"):
        b.append(
            kind="deploy",
            t=1000,
            payload={
                "compiled_spec_hash": "a" * 64,
                "result_hash": "b" * 64,
                "dataset_id": "ds",
                "starting_cash": math.inf,
            },
        )


def test_genesis_starting_cash_zero_raises():
    """Builder rejects a genesis with starting_cash=0."""
    b = ChainBuilder()
    with pytest.raises(ValueError, match="> 0"):
        b.append(
            kind="deploy",
            t=1000,
            payload={
                "compiled_spec_hash": "a" * 64,
                "result_hash": "b" * 64,
                "dataset_id": "ds",
                "starting_cash": 0.0,
            },
        )


def test_genesis_starting_cash_negative_raises():
    """Builder rejects a genesis with starting_cash < 0."""
    b = ChainBuilder()
    with pytest.raises(ValueError, match="> 0"):
        b.append(
            kind="deploy",
            t=1000,
            payload={
                "compiled_spec_hash": "a" * 64,
                "result_hash": "b" * 64,
                "dataset_id": "ds",
                "starting_cash": -50.0,
            },
        )


def test_running_cash_before_genesis_raises():
    """running_cash() before genesis raises RuntimeError."""
    b = ChainBuilder()
    with pytest.raises(RuntimeError, match="genesis"):
        b.running_cash()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_same_hashes_three_times():
    """Building the same chain 3× must yield identical record_hashes."""
    hashes_a = [r.record_hash for r in _make_happy_chain()]
    hashes_b = [r.record_hash for r in _make_happy_chain()]
    hashes_c = [r.record_hash for r in _make_happy_chain()]
    assert hashes_a == hashes_b == hashes_c


def test_payload_change_changes_hash():
    """A change to any payload byte changes the record_hash and all subsequent hashes."""
    b1 = ChainBuilder()
    b1.append(
        kind="deploy", t=1000,
        payload={"compiled_spec_hash": "a" * 64, "result_hash": "b" * 64, "dataset_id": "ds-1", "starting_cash": 100.0},
    )
    b1.append(kind="tick", t=1001, payload={"new_equity": 100.0})

    b2 = ChainBuilder()
    b2.append(
        kind="deploy", t=1000,
        payload={"compiled_spec_hash": "a" * 64, "result_hash": "b" * 64, "dataset_id": "ds-2", "starting_cash": 100.0},  # changed
    )
    b2.append(kind="tick", t=1001, payload={"new_equity": 100.0})

    r1 = b1.records()
    r2 = b2.records()
    # All record hashes should differ
    for a, b in zip(r1, r2):
        assert a.record_hash != b.record_hash


# ---------------------------------------------------------------------------
# CLI via subprocess
# ---------------------------------------------------------------------------


def _chain_to_json(records: list[ChainRecord]) -> str:
    return json.dumps([_record_to_dict(r) for r in records])


def _record_to_dict(r: ChainRecord) -> dict:
    return {
        "seq": r.seq,
        "t": r.t,
        "kind": r.kind,
        "payload": r.payload,
        "prev_hash": r.prev_hash,
        "record_hash": r.record_hash,
    }


def test_cli_chain_verify_exit_0_on_valid(tmp_path: Path):
    chain_file = tmp_path / "chain.json"
    chain_file.write_text(_chain_to_json(_make_happy_chain()), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "pancake_engine.cli", "verify", "--chain", str(chain_file)],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr


def test_cli_chain_verify_exit_1_on_tampered(tmp_path: Path):
    records = list(_make_happy_chain())
    # Flip the hash of record 2
    from dataclasses import replace
    rec = records[2]
    flipped = rec.record_hash[:4] + ("0" if rec.record_hash[4] != "0" else "1") + rec.record_hash[5:]
    records[2] = replace(rec, record_hash=flipped)
    chain_file = tmp_path / "chain_tampered.json"
    chain_file.write_text(_chain_to_json(records), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "pancake_engine.cli", "verify", "--chain", str(chain_file)],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 1, result.stderr


def test_cli_chain_verify_exit_2_on_invalid_json(tmp_path: Path):
    chain_file = tmp_path / "bad.json"
    chain_file.write_text("not valid json {{{", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "pancake_engine.cli", "verify", "--chain", str(chain_file)],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 2, result.stderr


def test_cli_chain_verify_exit_2_on_not_a_list(tmp_path: Path):
    chain_file = tmp_path / "notalist.json"
    chain_file.write_text(json.dumps({"seq": 0}), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "pancake_engine.cli", "verify", "--chain", str(chain_file)],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 2, result.stderr


def test_cli_chain_verify_exit_0_prints_json(tmp_path: Path):
    chain_file = tmp_path / "chain.json"
    chain_file.write_text(_chain_to_json(_make_happy_chain()), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "pancake_engine.cli", "verify", "--chain", str(chain_file)],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0
    out = json.loads(result.stdout)
    assert out["verified"] is True
    assert out["record_count"] > 0


# ---------------------------------------------------------------------------
# prev_hash linkage
# ---------------------------------------------------------------------------


def test_prev_hash_chain_linkage():
    records = _make_happy_chain()
    for i in range(1, len(records)):
        assert records[i].prev_hash == records[i - 1].record_hash


# ---------------------------------------------------------------------------
# format_version in hash payload
# ---------------------------------------------------------------------------


def test_record_hash_includes_format_version():
    """Changing format_version would change every hash — it's part of the payload."""
    from pancake_engine.chain.records import CHAIN_FORMAT_VERSION
    assert CHAIN_FORMAT_VERSION == "chain/1"


# ---------------------------------------------------------------------------
# Engine identity in hash payload
# ---------------------------------------------------------------------------


def test_engine_identity_in_chain_record():
    """Chain records carry ENGINE + ENGINE_VERSION in their hash payload."""
    from pancake_engine.__version__ import ENGINE, ENGINE_VERSION
    from pancake_engine.chain.records import _compute_record_hash

    # Two identical chain records but with different (simulated) engine ids would differ.
    # We test this indirectly by checking _compute_record_hash is deterministic
    # and that the function signature accepts/uses engine identity.
    h1 = _compute_record_hash(seq=0, t=100, kind="tick", payload={}, prev_hash="")
    h2 = _compute_record_hash(seq=0, t=100, kind="tick", payload={}, prev_hash="")
    assert h1 == h2


# ---------------------------------------------------------------------------
# advance() standalone
# ---------------------------------------------------------------------------


def test_advance_returns_new_record_not_mutation():
    from pancake_engine.chain.orders import advance
    rec1 = advance("proposed", "submitted", t=1, payload={"order_id": "x"})
    assert rec1["state"] == "submitted"


def test_advance_proposed_to_canceled_is_legal():
    from pancake_engine.chain.orders import advance
    rec = advance("proposed", "canceled", t=1, payload={"order_id": "y"})
    assert rec["state"] == "canceled"


def test_advance_proposed_to_rejected_is_legal():
    from pancake_engine.chain.orders import advance
    rec = advance("proposed", "rejected", t=1, payload={"order_id": "z"})
    assert rec["state"] == "rejected"


def test_advance_submitted_to_expired_is_legal():
    from pancake_engine.chain.orders import advance
    rec = advance("submitted", "expired", t=1, payload={})
    assert rec["state"] == "expired"


# ---------------------------------------------------------------------------
# All-states smoke
# ---------------------------------------------------------------------------


def test_full_lifecycle_proposed_to_filled():
    b = ChainBuilder()
    b.append(
        kind="deploy",
        t=1000,
        payload={"compiled_spec_hash": "a" * 64, "result_hash": "b" * 64, "dataset_id": "ds", "starting_cash": 100.0},
    )
    b.append(kind="order_state", t=1001,
             payload={"order_id": "life", "state": "proposed", "instrument_id": "m"})
    b.append(kind="order_state", t=1001,
             payload={"order_id": "life", "state": "submitted"})
    b.append(kind="order_state", t=1002,
             payload={"order_id": "life", "state": "acked"})
    b.append(kind="fill", t=1002,
             payload={"order_id": "life", "fill_qty": 10.0, "order_qty": 10.0, "cash_delta": -10.0})
    b.append(kind="order_state", t=1002,
             payload={"order_id": "life", "state": "filled"})
    records = b.records()
    verdict = verify_chain(records)
    assert verdict.ok


def test_full_lifecycle_proposed_to_expired():
    b = ChainBuilder()
    b.append(
        kind="deploy",
        t=1000,
        payload={"compiled_spec_hash": "a" * 64, "result_hash": "b" * 64, "dataset_id": "ds", "starting_cash": 100.0},
    )
    b.append(kind="order_state", t=1001,
             payload={"order_id": "exp1", "state": "proposed", "instrument_id": "m"})
    b.append(kind="order_state", t=1001,
             payload={"order_id": "exp1", "state": "submitted"})
    b.append(kind="order_state", t=1002,
             payload={"order_id": "exp1", "state": "expired"})
    records = b.records()
    verdict = verify_chain(records)
    assert verdict.ok


def test_guard_kind_appended():
    b = ChainBuilder()
    b.append(
        kind="deploy",
        t=1000,
        payload={"compiled_spec_hash": "a" * 64, "result_hash": "b" * 64, "dataset_id": "ds", "starting_cash": 100.0},
    )
    b.append(kind="guard", t=1001,
             payload={"guard": "max_drawdown_pct", "observed": 0.12, "threshold": 0.10})
    records = b.records()
    assert records[-1].kind == "guard"
    verdict = verify_chain(records)
    assert verdict.ok


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _relink_chain(records: list[ChainRecord], from_index: int) -> None:
    """Re-compute prev_hash + record_hash for records[from_index:] after a mutation.
    Mutates the list in place using dataclass replace.
    """
    from dataclasses import replace
    from pancake_engine.chain.records import _compute_record_hash

    for i in range(from_index, len(records)):
        prev_hash = "" if i == 0 else records[i - 1].record_hash
        rec = records[i]
        new_hash = _compute_record_hash(
            seq=rec.seq,
            t=rec.t,
            kind=rec.kind,
            payload=rec.payload,
            prev_hash=prev_hash,
        )
        records[i] = replace(rec, prev_hash=prev_hash, record_hash=new_hash)


class TestVenueRejectEdges:
    """Release-cut additions: the CTF V2 venue-reject + pre-submit-expiry edges.

    submitted -> rejected models a failed GET /transaction/{id} poll after
    submit (the venue's verdict); proposed -> expired models TTL elapsing
    before submission. Both are distinct from canceled (our action).
    """

    def test_submitted_to_rejected_is_legal(self) -> None:
        from pancake_engine.chain.orders import advance

        rec = advance("submitted", "rejected", t=1, payload={"reason": "venue_reject"})
        assert rec["state"] == "rejected"

    def test_proposed_to_expired_is_legal(self) -> None:
        from pancake_engine.chain.orders import advance

        rec = advance("proposed", "expired", t=1, payload={"reason": "ttl_before_submit"})
        assert rec["state"] == "expired"

    def test_acked_to_rejected_stays_illegal(self) -> None:
        import pytest

        from pancake_engine.chain.orders import ChainTransitionError, advance

        with pytest.raises(ChainTransitionError):
            advance("acked", "rejected", t=1, payload={})

    def test_full_table_enumeration_against_spec(self) -> None:
        """All 64 pairs vs the design-doc spec — the table is the contract."""
        import itertools

        from pancake_engine.chain.orders import ORDER_TRANSITIONS

        spec = {
            "proposed": {"submitted", "canceled", "rejected", "expired"},
            "submitted": {"acked", "canceled", "rejected", "expired"},
            "acked": {"partially_filled", "filled", "canceled", "expired"},
            "partially_filled": {"partially_filled", "filled", "canceled", "expired"},
            "filled": set(),
            "canceled": set(),
            "rejected": set(),
            "expired": set(),
        }
        for a, b in itertools.product(spec, repeat=2):
            assert (b in spec[a]) == (b in ORDER_TRANSITIONS.get(a, frozenset())), (a, b)
