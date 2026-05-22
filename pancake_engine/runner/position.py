"""Open-position record in the ledger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Position"]


@dataclass(frozen=True)
class Position:
    """An open position. Immutable; closed positions become ``Trade`` records."""

    id: int
    """Position id == source_row_index of the row that opened it."""

    market_link: str
    side: str  # "YES" or "NO"

    decision_time: int
    resolution_time: int

    entry_price: float
    """Post-slip fill price (the price paid for one share of the side traded)."""

    entry_price_quote: float
    """Pre-slip quoted price (the literal ``entry_price`` column value)."""

    shares: float
    cost: float
    """Cash deducted at decision time = notional (includes fee)."""

    fee: float
    """Fee component of cost."""

    row: dict[str, Any]
