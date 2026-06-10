"""ChainBuilder — append records to a chain, enforcing all invariants.

Invariants enforced by append():
  1. Genesis-first: first record must be kind="deploy".
  2. Seq density: seq = len(records) (0-based, no gaps).
  3. t monotonicity: t >= previous t.
  4. Genesis payload must contain compiled_spec_hash, result_hash, dataset_id,
     and starting_cash (finite float > 0).
  5. Order-state machine for order_state and fill kinds:
     - Tracks per-order current state internally.
     - Raises ChainTransitionError for illegal transitions.
     - Raises ValueError for fill overshoot (cumulative > order_qty).
  6. Reconciliation kind: payload must contain "diffs" key (list).

running_cash() returns the current authoritative cash balance computed via
math.fsum over genesis.starting_cash + all fill/settlement cash_delta values
appended so far.  Producers must use this method when setting total_cash on
settlement records — hand-computing the same sum independently risks drift.
"""

from __future__ import annotations

import math
from typing import Any

from .errors import ChainTransitionError
from .orders import (
    ORDER_TRANSITIONS,
    TERMINAL_STATES,
    FILLABLE_STATES,
    advance,
)
from .records import (
    ChainRecord,
    GENESIS_REQUIRED_KEYS,
    _compute_record_hash,
)

__all__ = ["ChainBuilder"]

# Valid record kinds.
VALID_KINDS = frozenset({
    "deploy", "tick", "order_state", "fill",
    "settlement", "guard", "reconciliation",
})


class ChainBuilder:
    """Stateful chain builder. Append records; retrieve the frozen list via records()."""

    def __init__(self) -> None:
        self._records: list[ChainRecord] = []
        # Per-order state tracking: order_id → current state string.
        self._order_states: dict[str, str] = {}
        # Per-order cumulative fill qty: order_id → float.
        self._order_fill_qty: dict[str, float] = {}
        # Per-order order_qty (set on first fill): order_id → float.
        self._order_qty: dict[str, float] = {}
        # Cash roll-forward: starting_cash from genesis + accumulated cash_deltas.
        # _cash_terms[0] = starting_cash; subsequent entries are each cash_delta appended.
        self._cash_terms: list[float] = []

    def append(self, *, kind: str, t: int, payload: dict[str, Any]) -> ChainRecord:
        """Append a new record to the chain.

        Args:
            kind:    Record kind string.
            t:       Unix seconds timestamp.
            payload: Kind-specific payload dict (shallow-copied; caller may mutate theirs).

        Returns:
            The new ChainRecord (also available via records()).

        Raises:
            ValueError:            On any structural violation.
            ChainTransitionError:  On an illegal order-state transition.
        """
        if kind not in VALID_KINDS:
            raise ValueError(f"Unknown chain record kind: {kind!r}. Valid: {sorted(VALID_KINDS)}")

        # Genesis-first check.
        if not self._records and kind != "deploy":
            raise ValueError(
                f"First record must be kind='deploy' (genesis); got {kind!r}"
            )

        # Seq density.
        seq = len(self._records)

        # t monotonicity.
        if self._records:
            prev_t = self._records[-1].t
            if t < prev_t:
                raise ValueError(
                    f"t must be monotone nondecreasing: got t={t} < previous t={prev_t}"
                )

        # Make a shallow copy so the caller can't mutate our payload.
        payload = dict(payload)

        # Kind-specific validation.
        if kind == "deploy":
            self._validate_genesis(payload)
            # Seed the cash roll-forward with starting_cash.
            self._cash_terms = [float(payload["starting_cash"])]

        elif kind == "order_state":
            self._validate_order_state(payload, t)

        elif kind == "fill":
            self._validate_fill(payload)
            # Accumulate cash_delta into the roll-forward.
            cd = payload.get("cash_delta")
            if cd is not None:
                self._cash_terms.append(float(cd))

        elif kind == "settlement":
            # Accumulate cash_delta into the roll-forward.
            cd = payload.get("cash_delta")
            if cd is not None:
                self._cash_terms.append(float(cd))

        elif kind == "reconciliation":
            self._validate_reconciliation(payload)

        # Compute hashes.
        prev_hash = "" if not self._records else self._records[-1].record_hash
        record_hash = _compute_record_hash(
            seq=seq,
            t=t,
            kind=kind,
            payload=payload,
            prev_hash=prev_hash,
        )

        rec = ChainRecord(
            seq=seq,
            t=t,
            kind=kind,
            payload=payload,
            prev_hash=prev_hash,
            record_hash=record_hash,
        )
        self._records.append(rec)
        return rec

    def records(self) -> list[ChainRecord]:
        """Return a snapshot of the current record list (new list, same frozen records)."""
        return list(self._records)

    def running_cash(self) -> float:
        """Return the current authoritative cash balance.

        Computed as math.fsum(starting_cash, *cash_deltas) over every fill and
        settlement cash_delta appended so far.  Raises RuntimeError if genesis
        has not yet been appended (starting_cash not available).

        Producers MUST use this method when populating total_cash on settlement
        records — hand-computing the same sum independently risks floating-point
        drift that verify_chain will catch as E_CHAIN_CASH_MISMATCH.
        """
        if not self._cash_terms:
            raise RuntimeError(
                "running_cash() called before genesis (deploy) record was appended"
            )
        return math.fsum(self._cash_terms)

    # ------------------------------------------------------------------
    # private validators
    # ------------------------------------------------------------------

    def _validate_genesis(self, payload: dict[str, Any]) -> None:
        missing = GENESIS_REQUIRED_KEYS - set(payload)
        if missing:
            raise ValueError(
                f"Genesis (deploy) payload missing required keys: {sorted(missing)}"
            )
        # starting_cash must be a finite float > 0.
        sc = payload["starting_cash"]
        try:
            sc_f = float(sc)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Genesis starting_cash must be a finite float > 0; got {sc!r}"
            ) from exc
        if not math.isfinite(sc_f):
            raise ValueError(
                f"Genesis starting_cash must be finite; got {sc_f}"
            )
        if sc_f <= 0:
            raise ValueError(
                f"Genesis starting_cash must be > 0; got {sc_f}"
            )

    def _validate_order_state(self, payload: dict[str, Any], t: int) -> None:
        order_id = payload.get("order_id")
        if order_id is None:
            raise ValueError("order_state payload must contain 'order_id'")
        new_state = payload.get("state")
        if new_state is None:
            raise ValueError("order_state payload must contain 'state'")
        if new_state not in ORDER_TRANSITIONS:
            raise ValueError(f"Unknown order state: {new_state!r}")

        if order_id not in self._order_states:
            # First record for this order: must be "proposed".
            if new_state != "proposed":
                raise ValueError(
                    f"First order_state for order {order_id!r} must be 'proposed'; got {new_state!r}"
                )
            if "instrument_id" not in payload:
                raise ValueError(
                    f"First order_state ('proposed') for order {order_id!r} must carry 'instrument_id'"
                )
            self._order_states[order_id] = new_state
        else:
            current = self._order_states[order_id]
            # advance() raises ChainTransitionError on illegal transition.
            advance(current, new_state, t=t, payload=payload)
            self._order_states[order_id] = new_state

    def _validate_fill(self, payload: dict[str, Any]) -> None:
        order_id = payload.get("order_id")
        if order_id is None:
            raise ValueError("fill payload must contain 'order_id'")

        fill_qty = payload.get("fill_qty")
        order_qty = payload.get("order_qty")

        if fill_qty is None:
            raise ValueError("fill payload must contain 'fill_qty'")
        if order_qty is None:
            raise ValueError("fill payload must contain 'order_qty'")

        fill_qty = float(fill_qty)
        order_qty = float(order_qty)

        # Check the order is in a fillable state.
        current_state = self._order_states.get(order_id)
        if current_state is None:
            raise ValueError(
                f"fill references unknown order_id {order_id!r} "
                "(no preceding order_state record)"
            )
        if current_state not in FILLABLE_STATES:
            raise ValueError(
                f"fill on order {order_id!r} is only valid in states "
                f"{sorted(FILLABLE_STATES)!r}; current state is {current_state!r}"
            )

        # Track/update order_qty (must be consistent across fills for the same order).
        if order_id in self._order_qty:
            if self._order_qty[order_id] != order_qty:
                raise ValueError(
                    f"fill order_qty {order_qty} is inconsistent with prior "
                    f"order_qty {self._order_qty[order_id]} for order {order_id!r}"
                )
        else:
            self._order_qty[order_id] = order_qty

        # Accumulate and check overshoot.
        prev_cumulative = self._order_fill_qty.get(order_id, 0.0)
        new_cumulative = prev_cumulative + fill_qty
        if new_cumulative > order_qty:
            raise ValueError(
                f"fill overshoot on order {order_id!r}: "
                f"cumulative fill qty {new_cumulative} would exceed "
                f"order_qty {order_qty}"
            )

        self._order_fill_qty[order_id] = new_cumulative

    def _validate_reconciliation(self, payload: dict[str, Any]) -> None:
        if "diffs" not in payload:
            raise ValueError(
                "reconciliation payload must contain 'diffs' key "
                "(list of explicit diff objects)"
            )
        if not isinstance(payload["diffs"], list):
            raise ValueError("reconciliation payload 'diffs' must be a list")
