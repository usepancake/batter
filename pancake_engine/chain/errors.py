"""Typed errors for the chain module."""

from __future__ import annotations

__all__ = ["ChainTransitionError"]


class ChainTransitionError(Exception):
    """Raised when an illegal order-state transition is attempted.

    Attributes:
        from_state: The state the order is currently in.
        to_state: The state that was illegally requested.
    """

    def __init__(self, from_state: str, to_state: str) -> None:
        super().__init__(
            f"Illegal order-state transition: {from_state!r} → {to_state!r}"
        )
        self.from_state = from_state
        self.to_state = to_state
