"""Calendar / seconds anchor helpers for fold scheduling.

PR-2 supports two anchor units: ``MS`` (month start, UTC) and ``QS`` (quarter
start, UTC). Quarters anchor at months 1, 4, 7, 10 of each year. Integer values
are interpreted as plain unix seconds.

Multipliers parsed from prefixes: ``3MS`` = 3 months, ``2QS`` = 2 quarters = 6
months. ``MS`` alone = 1 month; ``QS`` alone = 1 quarter.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Union

__all__ = [
    "parse_anchor",
    "advance_anchor",
    "next_anchor_boundary",
    "AnchorSpec",
]

AnchorSpec = Union[int, str]

_ANCHOR_RE = re.compile(r"^(\d*)(MS|QS)$")


def parse_anchor(value: AnchorSpec) -> tuple[int, str]:
    """Parse an anchor spec into ``(n, unit)``.

    For integers: ``(seconds, 'sec')``.
    For strings: ``('3MS' -> (3, 'MS'))``, ``('QS' -> (1, 'QS'))``.

    Raises ``ValueError`` for unsupported strings.
    """
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"E_WALKFORWARD_CONFIG: anchor seconds must be > 0, got {value}")
        return value, "sec"
    if not isinstance(value, str):
        raise ValueError(f"E_WALKFORWARD_CONFIG: anchor must be int or str, got {type(value).__name__}")
    m = _ANCHOR_RE.match(value)
    if not m:
        raise ValueError(
            f"E_WALKFORWARD_CONFIG: unknown anchor {value!r} — supported: MS, QS, NMS, NQS"
        )
    n_str, unit = m.groups()
    n = int(n_str) if n_str else 1
    if n <= 0:
        raise ValueError(f"E_WALKFORWARD_CONFIG: anchor multiplier must be > 0, got {n}")
    return n, unit


def _to_utc_dt(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _to_ts(dt: datetime) -> int:
    return int(dt.timestamp())


def _add_months(dt: datetime, months: int) -> datetime:
    """Add ``months`` months to ``dt`` (preserving day=1; UTC)."""
    new_year = dt.year + (dt.month - 1 + months) // 12
    new_month = (dt.month - 1 + months) % 12 + 1
    return datetime(new_year, new_month, 1, tzinfo=timezone.utc)


def next_anchor_boundary(ts: int, unit: str) -> int:
    """First UTC anchor boundary at or after ``ts``.

    For ``MS``: next month start (1st of month, UTC midnight). If ``ts`` is
    already on a month start, returns ``ts``.

    For ``QS``: next quarter start (months 1, 4, 7, 10, day 1, UTC midnight).
    If ``ts`` is already on a quarter start, returns ``ts``.

    For ``sec``: returns ``ts`` (no-op).
    """
    if unit == "sec":
        return ts
    dt = _to_utc_dt(ts)
    if unit == "MS":
        floor = datetime(dt.year, dt.month, 1, tzinfo=timezone.utc)
        if floor == dt:
            return _to_ts(dt)
        return _to_ts(_add_months(floor, 1))
    if unit == "QS":
        # Quarter months: 1, 4, 7, 10
        quarter_month = ((dt.month - 1) // 3) * 3 + 1
        floor = datetime(dt.year, quarter_month, 1, tzinfo=timezone.utc)
        if floor == dt:
            return _to_ts(dt)
        return _to_ts(_add_months(floor, 3))
    raise ValueError(f"E_WALKFORWARD_CONFIG: unsupported anchor unit {unit!r}")


def advance_anchor(ts: int, n: int, unit: str) -> int:
    """Advance ``ts`` by ``n`` units of ``unit``.

    For ``sec``: ``ts + n``.
    For ``MS``: ``ts`` is assumed to be on a month boundary; add ``n`` months.
    For ``QS``: ``ts`` is assumed to be on a quarter boundary; add ``n × 3`` months.
    """
    if unit == "sec":
        return ts + n
    dt = _to_utc_dt(ts)
    if unit == "MS":
        return _to_ts(_add_months(dt, n))
    if unit == "QS":
        return _to_ts(_add_months(dt, n * 3))
    raise ValueError(f"E_WALKFORWARD_CONFIG: unsupported anchor unit {unit!r}")
