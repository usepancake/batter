"""Event records for the ledger walk.

Each row in the dataset produces (DECISION, RESOLUTION) event pair, both
sharing the ``source_row_index`` from the original ``rows_inline`` order.
Events sort by ``(time, kind_order, market_link, source_row_index)`` with
``DECISION < RESOLUTION`` at equal times (architecture §Event ordering).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

__all__ = ["EventKind", "Event"]


class EventKind(IntEnum):
    DECISION = 0
    RESOLUTION = 1


@dataclass(frozen=True)
class Event:
    time: int
    kind: EventKind
    market_link: str
    source_row_index: int
    row: dict[str, Any]

    def sort_key(self) -> tuple[int, int, str, int]:
        return (self.time, int(self.kind), self.market_link, self.source_row_index)
