"""Time-series helpers: daily resampling (carry-forward), drawdown, monthly returns.

Engine 0.3 is correctness-first, not TS parity. **Daily resampling diverges from
TS**: TS picks the last point per day-that-has-a-point and skips days without
events. Engine 0.3 carries forward the last observed equity to every UTC day in
the ``[start_date, end_date]`` window (architecture decision; documented as
D-14 in ts-divergences.md).

This matters under sparse trading: TS Sharpe over (decision_day, resolution_day)
treats them as adjacent and yields one (potentially huge) daily return; Engine
0.3 produces zero returns on the in-between days, surfacing the strategy's
duty cycle honestly.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from ..result import DrawdownPoint, EquityPoint, MonthlyReturn

__all__ = [
    "daily_returns_carry_forward",
    "build_drawdown_curve",
    "build_monthly_returns",
]

SECONDS_PER_DAY = 86_400


def _utc_day_floor_secs(ts: int) -> int:
    """Floor a unix-seconds timestamp to UTC midnight."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    floor = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(floor.timestamp())


def daily_returns_carry_forward(equity_curve: list[EquityPoint]) -> list[float]:
    """Resample equity to one observation per UTC day via last-value carry-forward.

    Returns the list of simple daily returns ``(today / yesterday) - 1`` for
    consecutive days in the carry-forward window.

    - ``[]`` if equity_curve has fewer than 2 points
    - skips a day's return if the previous day's equity ≤ 0 (matches TS guard)
    """
    if len(equity_curve) < 2:
        return []

    sorted_curve = sorted(equity_curve, key=lambda p: p.t)
    start_day = _utc_day_floor_secs(sorted_curve[0].t)
    end_day = _utc_day_floor_secs(sorted_curve[-1].t)

    # Build day-by-day equity via last-value carry-forward.
    daily: list[float] = []
    curve_idx = 0
    current_equity = sorted_curve[0].equity
    day = start_day
    while day <= end_day:
        # Advance curve_idx while the next equity_curve point is at or before this day.
        while curve_idx + 1 < len(sorted_curve) and sorted_curve[curve_idx + 1].t <= day + SECONDS_PER_DAY - 1:
            curve_idx += 1
            current_equity = sorted_curve[curve_idx].equity
        # Special-case first day: use the first equity point if it's within this day
        if day == start_day:
            current_equity = sorted_curve[0].equity
            # advance past any same-day duplicates picking the last one
            while curve_idx + 1 < len(sorted_curve) and sorted_curve[curve_idx + 1].t <= day + SECONDS_PER_DAY - 1:
                curve_idx += 1
                current_equity = sorted_curve[curve_idx].equity
        daily.append(current_equity)
        day += SECONDS_PER_DAY

    out: list[float] = []
    for i in range(1, len(daily)):
        prev = daily[i - 1]
        if prev <= 0:
            continue
        r = daily[i] / prev - 1.0
        if math.isfinite(r):  # AF-3: filters both NaN and ±Inf (e.g. when equity overflows float64)
            out.append(r)
    return out


def build_drawdown_curve(equity_curve: list[EquityPoint]) -> tuple[list[DrawdownPoint], float]:
    """Walk the event-time equity curve; return ``(curve, max_drawdown)``.

    Drawdown reported positive: ``(peak − equity) / peak``.
    """
    if not equity_curve:
        return [], 0.0

    curve: list[DrawdownPoint] = []
    peak = equity_curve[0].equity
    max_dd = 0.0
    for p in equity_curve:
        if p.equity > peak:
            peak = p.equity
        dd = (peak - p.equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
        curve.append(DrawdownPoint(t=p.t, drawdown=dd))
    return curve, max_dd


def build_monthly_returns(equity_curve: list[EquityPoint]) -> list[MonthlyReturn]:
    """Group equity_curve points by UTC (year, month); return per-month return.

    Per-month return = ``end_equity / start_equity - 1`` where start/end are
    the first/last equity points in that month. Uses raw event-time samples;
    does NOT carry-forward across months (a month with no events does not appear).
    """
    if not equity_curve:
        return []
    sorted_curve = sorted(equity_curve, key=lambda p: p.t)
    by_month: dict[tuple[int, int], list[EquityPoint]] = {}
    for p in sorted_curve:
        dt = datetime.fromtimestamp(p.t, tz=timezone.utc)
        key = (dt.year, dt.month)
        by_month.setdefault(key, []).append(p)

    out: list[MonthlyReturn] = []
    for (year, month) in sorted(by_month.keys()):
        pts = by_month[(year, month)]
        start = pts[0].equity
        end = pts[-1].equity
        r = (end / start - 1.0) if start > 0 else 0.0
        out.append(MonthlyReturn(year=year, month=month, return_pct=r))
    return out
