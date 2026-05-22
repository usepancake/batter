"""Completed-trade record for the result."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

__all__ = ["Trade"]


@dataclass(frozen=True)
class Trade:
    """A resolved trade. Mirrors the TS ``Trade`` shape for parity with the TS runner."""

    market_slug: str
    outcome: str  # "YES" or "NO" — side traded
    entry_t: int
    entry_price: float           # post-slip fill
    entry_price_quote: float     # pre-slip quote
    exit_t: int
    exit_price: float            # settle_value 0 or 1
    exit_price_quote: float      # same as exit_price under frictionless settle
    exit_reason: str             # "hold_to_resolution" in PR-1
    shares: float
    cost: float
    proceeds: float
    pnl: float
    return_pct: float
    days_held: int
    resolved_outcome: Optional[int]  # null in PR-1 (TS parity)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
