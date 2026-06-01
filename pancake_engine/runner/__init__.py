"""Event-time ledger runner for Pancake Engine.

``run_backtest`` is the full DECISION→RESOLUTION walk (PR-1). ``tick`` is the
single-bar paper step (ADR-0035), reusing the same fill/ledger primitives.
"""

from .engine import run_backtest
from .fill import Fill, FillRejection, SimFillRouter
from .tick import (
    MarketBar,
    PaperEvent,
    ResolutionMarker,
    TickError,
    TickPosition,
    TickRequest,
    TickResponse,
    VerificationBoundary,
    tick,
)

__all__ = [
    "run_backtest",
    "tick",
    "SimFillRouter",
    "Fill",
    "FillRejection",
    "MarketBar",
    "ResolutionMarker",
    "TickPosition",
    "PaperEvent",
    "VerificationBoundary",
    "TickRequest",
    "TickResponse",
    "TickError",
]
