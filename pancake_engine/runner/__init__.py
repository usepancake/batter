"""Event-time ledger runner for Pancake Engine.

``run_backtest`` is the full DECISION→RESOLUTION walk (PR-1). ``tick`` is the
single-bar paper step (ADR-0035), reusing the same fill/ledger primitives.
"""

from .engine import run_backtest
from .batch import run_many, TrialLedger, BatchResult
from .fill import Fill, FillRejection, SimFillRouter
from .tick import (
    CryptoTickBar,
    CryptoTickPosition,
    CryptoTickRequest,
    CryptoTickResponse,
    MarketBar,
    PaperEvent,
    ResolutionMarker,
    TickError,
    TickPosition,
    TickRequest,
    TickResponse,
    VerificationBoundary,
    tick,
    tick_crypto,
)

__all__ = [
    "run_backtest",
    "run_many",
    "TrialLedger",
    "BatchResult",
    "tick",
    "tick_crypto",
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
    "CryptoTickBar",
    "CryptoTickPosition",
    "CryptoTickRequest",
    "CryptoTickResponse",
]
