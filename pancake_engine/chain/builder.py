"""ChainBuilder — append records to a chain, enforcing all invariants.

Invariants enforced by append():
  1. Genesis-first: first record must be kind="deploy".
  2. Seq density: seq = len(records) (0-based, no gaps).
  3. t monotonicity: t >= previous t.
  4. Genesis payload must contain compiled_spec_hash, result_hash, dataset_id.
  5. Order-state machine for order_state and fill kinds:
     - Tracks per-order current state internally.
     - Raises ChainTransitionError for illegal transitions.
     - Raises ValueError for fill overshoot (cumulative > order_qty).
  6. Reconciliation kind: payload must contain "diffs" key (list).
"""

from __future__ import annotations

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

        elif kind == "order_state":
            self._validate_order_state(payload, t)

        elif kind == "fill":
            self._validate_fill(payload)

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

    # ------------------------------------------------------------------
    # private validators
    # ------------------------------------------------------------------

    def _validate_genesis(self, payload: dict[str, Any]) -> None:
        missing = GENESIS_REQUIRED_KEYS - set(payload)
        if missing:
            raise ValueError(
                f"Genesis (deploy) payload missing required keys: {sorted(missing)}"
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
