"""Order-state machine for the chain module.

States: proposed → submitted → acked → partially_filled → filled
                                                         → canceled
                              → canceled
                 → canceled
        proposed → rejected
        submitted → expired
        acked → expired
        partially_filled → canceled | expired

Transition TABLE: state → frozenset(allowed next states).
Terminal states: filled | canceled | rejected | expired — admit nothing.

fill(qty) may only be applied when state ∈ {acked, partially_filled}.
Cumulative fill qty must be monotone nondecreasing and never exceed order_qty.
"""

from __future__ import annotations

from typing import Any

from .errors import ChainTransitionError

__all__ = [
    "ORDER_TRANSITIONS",
    "TERMINAL_STATES",
    "advance",
]

# Frozen transition table.
ORDER_TRANSITIONS: dict[str, frozenset[str]] = {
    # proposed → expired: TTL elapsed before submission (distinct from canceled,
    # which is OUR action). proposed → rejected: pre-submit validation reject.
    "proposed": frozenset({"submitted", "canceled", "rejected", "expired"}),
    # submitted → rejected: the VENUE rejects at submission processing — on CTF
    # Exchange V2 a failed GET /transaction/{id} poll after submit IS this edge
    # (reject = venue's verdict; canceled = our action; never conflate them).
    "submitted": frozenset({"acked", "canceled", "expired", "rejected"}),
    "acked": frozenset({"partially_filled", "filled", "canceled", "expired"}),
    "partially_filled": frozenset({"partially_filled", "filled", "canceled", "expired"}),
    # Terminal states — nothing follows.
    "filled": frozenset(),
    "canceled": frozenset(),
    "rejected": frozenset(),
    "expired": frozenset(),
}

TERMINAL_STATES: frozenset[str] = frozenset({"filled", "canceled", "rejected", "expired"})

# States from which a fill event is valid.
FILLABLE_STATES: frozenset[str] = frozenset({"acked", "partially_filled"})


def advance(from_state: str, to_state: str, *, t: int, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a state transition and return a new record dict.

    Raises:
        ChainTransitionError: if the transition is not in the TABLE.
    """
    allowed = ORDER_TRANSITIONS.get(from_state, frozenset())
    if to_state not in allowed:
        raise ChainTransitionError(from_state, to_state)
    return {**payload, "state": to_state, "t": t}
