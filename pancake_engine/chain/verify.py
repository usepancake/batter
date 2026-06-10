"""Chain verification — verify_chain() re-walks a list of ChainRecord objects
and checks every invariant without trusting any stored value.

Checks performed:
  1. Seq density: records[i].seq == i for all i.
  2. t monotonicity: records[i].t >= records[i-1].t.
  3. record_hash recomputed: sha256_canonical matches stored value.
  4. prev_hash linkage: records[i].prev_hash == records[i-1].record_hash
     (genesis: prev_hash == "").
  5. Genesis present and first: records[0].kind == "deploy".
  6. Genesis payload provenance: compiled_spec_hash + result_hash + dataset_id
     + starting_cash (finite float > 0).
  7. Order-state machine replay: every order_state transition is legal.
  8. Cumulative fill monotonicity: fill qty is nondecreasing per order.
  9. Fill overshoot: cumulative fill qty never exceeds order_qty.
  10. Exact P&L roll-forward: running_cash = genesis.starting_cash +
      math.fsum(all cash_delta values seen so far).  Any record whose payload
      carries total_cash must satisfy total_cash == running_cash at that record
      EXACTLY (bit equality on the fsum result).  Producers must compute
      total_cash via ChainBuilder.running_cash() so both sides use the same
      fsum accumulation.

Error codes:
  SEQ_GAP                      — seq not dense
  T_REGRESSION                 — t went backward
  HASH_MISMATCH                — recomputed record_hash != stored
  PREV_LINK_BROKEN             — prev_hash doesn't match prior record_hash
  GENESIS_MISSING              — no records or first record not kind='deploy'
  GENESIS_MISSING_PROVENANCE   — genesis payload missing required provenance keys
  GENESIS_BAD_STARTING_CASH   — starting_cash missing, non-finite, or <= 0
  ILLEGAL_TRANSITION           — order-state machine violation
  FILL_OVERSHOOT               — cumulative fill_qty > order_qty
  STATE_MACHINE_VIOLATION      — fill on non-fillable state
  E_CHAIN_CASH_MISMATCH        — settlement total_cash != running_cash at that seq
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .errors import ChainTransitionError
from .orders import ORDER_TRANSITIONS, FILLABLE_STATES
from .records import ChainRecord, GENESIS_REQUIRED_KEYS, _compute_record_hash

__all__ = ["ChainVerdict", "verify_chain"]


@dataclass
class ChainVerdict:
    """Result of verify_chain().

    Attributes:
        ok:     True when no errors were found.
        errors: List of error dicts {seq, code, message}.
    """

    ok: bool
    errors: list[dict[str, Any]]


def verify_chain(records: list[ChainRecord]) -> ChainVerdict:
    """Verify a list of ChainRecord objects against all chain invariants.

    Args:
        records: The chain to verify (typically from ChainBuilder.records() or
                 deserialized from JSON).

    Returns:
        ChainVerdict with ok=True and errors=[] when valid.
    """
    errors: list[dict[str, Any]] = []

    def err(seq: int, code: str, message: str) -> None:
        errors.append({"seq": seq, "code": code, "message": message})

    # Empty chain is trivially invalid (no genesis).
    if not records:
        err(-1, "GENESIS_MISSING", "Chain is empty — no genesis record")
        return ChainVerdict(ok=False, errors=errors)

    # Check genesis.
    if records[0].kind != "deploy":
        err(0, "GENESIS_MISSING", f"First record must be kind='deploy'; got {records[0].kind!r}")

    # Extract starting_cash from genesis for the exact cash roll-forward.
    # If genesis is malformed we still continue checking other invariants but
    # disable the cash roll-forward (cash_terms stays empty → no mismatch errors
    # on top of the already-reported genesis error).
    cash_terms: list[float] = []
    genesis_payload = records[0].payload if records else {}
    sc_raw = genesis_payload.get("starting_cash")
    if sc_raw is not None:
        try:
            sc_f = float(sc_raw)
        except (TypeError, ValueError):
            sc_f = float("nan")
        if math.isfinite(sc_f) and sc_f > 0:
            cash_terms = [sc_f]
        else:
            err(0, "GENESIS_BAD_STARTING_CASH",
                f"Genesis starting_cash must be finite and > 0; got {sc_raw!r}")
    # starting_cash missing is caught by GENESIS_MISSING_PROVENANCE below.

    # Per-order tracking for state-machine + fill replay.
    order_states: dict[str, str] = {}
    order_fill_qty: dict[str, float] = {}
    order_qty: dict[str, float] = {}

    for i, rec in enumerate(records):
        seq = rec.seq

        # 1. Seq density.
        if seq != i:
            err(seq, "SEQ_GAP", f"seq {seq} at position {i} — expected {i}")

        # 2. t monotonicity.
        if i > 0 and rec.t < records[i - 1].t:
            err(seq, "T_REGRESSION",
                f"t={rec.t} < previous t={records[i - 1].t}")

        # 3. record_hash recomputed.
        expected_hash = _compute_record_hash(
            seq=seq,
            t=rec.t,
            kind=rec.kind,
            payload=rec.payload,
            prev_hash=rec.prev_hash,
        )
        if rec.record_hash != expected_hash:
            err(seq, "HASH_MISMATCH",
                f"record_hash mismatch: stored={rec.record_hash[:16]}… "
                f"recomputed={expected_hash[:16]}…")

        # 4. prev_hash linkage.
        expected_prev = "" if i == 0 else records[i - 1].record_hash
        if rec.prev_hash != expected_prev:
            err(seq, "PREV_LINK_BROKEN",
                f"prev_hash={rec.prev_hash[:16]}… expected={expected_prev[:16]}…")

        # 5+6. Genesis provenance.
        if i == 0:
            missing = GENESIS_REQUIRED_KEYS - set(rec.payload)
            if missing:
                err(seq, "GENESIS_MISSING_PROVENANCE",
                    f"Genesis payload missing required keys: {sorted(missing)}")

        # 7+8+9. Order-state machine + fill.
        if rec.kind == "order_state":
            order_id = rec.payload.get("order_id", "")
            new_state = rec.payload.get("state", "")

            if new_state not in ORDER_TRANSITIONS:
                err(seq, "ILLEGAL_TRANSITION",
                    f"Unknown state {new_state!r} for order {order_id!r}")
            elif order_id not in order_states:
                if new_state != "proposed":
                    err(seq, "ILLEGAL_TRANSITION",
                        f"First transition for order {order_id!r} must be 'proposed'; "
                        f"got {new_state!r}")
                order_states[order_id] = new_state
            else:
                current = order_states[order_id]
                allowed = ORDER_TRANSITIONS.get(current, frozenset())
                if new_state not in allowed:
                    err(seq, "ILLEGAL_TRANSITION",
                        f"Illegal transition {current!r} → {new_state!r} "
                        f"for order {order_id!r}")
                order_states[order_id] = new_state

        elif rec.kind == "fill":
            order_id = rec.payload.get("order_id", "")
            fill_qty = float(rec.payload.get("fill_qty", 0))
            oqty = float(rec.payload.get("order_qty", 0))
            cash_delta = rec.payload.get("cash_delta")

            # Check fillable state.
            current_state = order_states.get(order_id)
            if current_state is None:
                err(seq, "STATE_MACHINE_VIOLATION",
                    f"fill references unknown order_id {order_id!r}")
            elif current_state not in FILLABLE_STATES:
                err(seq, "STATE_MACHINE_VIOLATION",
                    f"fill on order {order_id!r} in state {current_state!r} "
                    f"(must be one of {sorted(FILLABLE_STATES)!r})")

            # Track order_qty consistency.
            if order_id in order_qty:
                if order_qty[order_id] != oqty:
                    err(seq, "FILL_OVERSHOOT",
                        f"order_qty {oqty} inconsistent with prior {order_qty[order_id]} "
                        f"for order {order_id!r}")
            else:
                order_qty[order_id] = oqty

            # Cumulative fill check.
            prev_cum = order_fill_qty.get(order_id, 0.0)
            new_cum = prev_cum + fill_qty
            if new_cum > oqty:
                err(seq, "FILL_OVERSHOOT",
                    f"Cumulative fill qty {new_cum} exceeds order_qty {oqty} "
                    f"for order {order_id!r}")
            order_fill_qty[order_id] = new_cum

            # Accumulate cash_delta into the roll-forward.
            if cash_delta is not None and cash_terms:
                cash_terms.append(float(cash_delta))

        elif rec.kind == "settlement":
            cash_delta = rec.payload.get("cash_delta")
            total_cash = rec.payload.get("total_cash")

            # Accumulate settlement cash_delta before checking total_cash so
            # that the roll-forward is current when we compare.
            if cash_delta is not None and cash_terms:
                cash_terms.append(float(cash_delta))

            # 10. Exact P&L roll-forward: total_cash must equal running_cash.
            if total_cash is not None and cash_terms:
                running = math.fsum(cash_terms)
                declared = float(total_cash)
                if declared != running:
                    err(seq, "E_CHAIN_CASH_MISMATCH",
                        f"total_cash {declared} != running_cash {running} at seq {seq} "
                        f"(expected={running}, declared={declared})")

    return ChainVerdict(ok=(len(errors) == 0), errors=errors)
