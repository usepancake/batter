"""PR-1 sizing: ``fixed_fraction`` × ``available_cash`` only.

Other modes and bases are spec'd in architecture.md but not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = ["SizingResult", "compute_sizing"]


@dataclass(frozen=True)
class SizingResult:
    notional: float
    """Notional after clip. May be 0 if available_cash is 0."""

    requested: float
    """Notional before clip."""

    clipped: bool
    """True if ``notional < requested``."""

    basis_value: float
    """The basis at sizing time (= available_cash in PR-1)."""


def compute_sizing(available_cash: float, sizing_value: float) -> SizingResult:
    """``fixed_fraction`` × ``available_cash``.

    Clips at ``available_cash`` so the runner never goes negative. Emits a
    ``SIZING_CLIPPED`` warning at the caller when ``clipped`` is True.
    """
    if available_cash < 0:
        # Cannot happen under correct ledger invariants; defensive.
        available_cash = 0.0
    requested = available_cash * sizing_value
    notional = min(requested, available_cash)
    clipped = notional < requested
    return SizingResult(
        notional=notional,
        requested=requested,
        clipped=clipped,
        basis_value=available_cash,
    )
