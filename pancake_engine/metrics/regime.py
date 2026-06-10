"""Regime-stability block — Wave E (0.10.0).

Additive non-hashed block attached to BacktestResult.regime under with_inference.
Splits the run into 4 equal-duration quartiles and reports per-quartile
trade activity + return + drawdown, plus a stability summary.

Design: docs/design-0.9.0-contracts-and-fills.md §5 Wave E.

Hash policy: NOT in result_hash — same pattern as calibration_bins / cost_sensitivity.
"""

from __future__ import annotations

import math
from typing import Any

from ..result import EquityPoint

__all__ = ["regime_stability"]

_MIN_TRADES = 8
_MIN_EQUITY_TIMESTAMPS = 4
_N_QUARTILES = 4


def regime_stability(
    equity_curve: list[EquityPoint],
    trades: list[Any],
) -> dict[str, Any] | None:
    """Compute regime-stability block.

    Split the run's time span into 4 equal-duration quartiles.  Per quartile:
      - ``num_trades``      — count of trades whose decision_time falls in the quartile
      - ``total_return``    — equity at end of quartile / equity at start of quartile - 1
                              (uses carry-forward from equity_curve)
      - ``max_drawdown``    — max drawdown within the quartile (from equity points
                              that fall inside the quartile boundary, inclusive)

    Stability summary:
      - ``return_sign_consistency`` — fraction of quartiles whose return sign matches
                                      the overall run return sign (range [0.0, 1.0])
      - ``worst_quartile_return``   — the lowest (most negative) per-quartile total_return

    Returns None when:
      - fewer than _MIN_TRADES (8) trades in the run, OR
      - fewer than _MIN_EQUITY_TIMESTAMPS (4) distinct equity curve timestamps.

    Trades must expose an ``entry_t`` attribute (int, epoch sec) — the engine's
    Trade.entry_t field, which corresponds to the decision timestamp.
    """
    distinct_ts = len({p.t for p in equity_curve})
    if distinct_ts < _MIN_EQUITY_TIMESTAMPS:
        return None
    if len(trades) < _MIN_TRADES:
        return None

    sorted_curve = sorted(equity_curve, key=lambda p: p.t)
    t_start = sorted_curve[0].t
    t_end = sorted_curve[-1].t
    span = t_end - t_start

    # Guard: zero-span (all equity points at same timestamp) → degenerate.
    if span == 0:
        return None

    # Quartile boundaries: 4 equal-duration windows.
    # Quartile i covers [t_start + i*(span/4), t_start + (i+1)*(span/4))
    # Last quartile is inclusive of t_end.
    q_duration = span / _N_QUARTILES
    boundaries = [t_start + i * q_duration for i in range(_N_QUARTILES + 1)]
    # boundaries[0]=t_start, boundaries[4]=t_end

    # Build a carry-forward equity lookup: for a given timestamp, what was equity?
    def _equity_at(ts: float) -> float:
        """Carry-forward equity at timestamp ts (linear scan; small curves)."""
        last = sorted_curve[0].equity
        for p in sorted_curve:
            if p.t <= ts:
                last = p.equity
            else:
                break
        return last

    # Assign each trade to a quartile by decision_time.
    # A trade whose decision_time == t is in quartile i where
    # boundaries[i] <= t < boundaries[i+1]; last quartile uses <=.
    def _quartile_idx(t: int) -> int:
        for i in range(_N_QUARTILES - 1):
            if boundaries[i] <= t < boundaries[i + 1]:
                return i
        return _N_QUARTILES - 1

    trade_counts: list[int] = [0] * _N_QUARTILES
    for trade in trades:
        # Engine Trade uses entry_t (= decision timestamp; the runner populates
        # this from the DECISION event time). Test stubs may use decision_time.
        dt = int(getattr(trade, "entry_t", getattr(trade, "decision_time", 0)))
        qi = _quartile_idx(dt)
        trade_counts[qi] += 1

    # Per-quartile: total_return + max_drawdown.
    quartile_blocks: list[dict[str, Any]] = []
    overall_end_equity = sorted_curve[-1].equity
    overall_start_equity = sorted_curve[0].equity
    overall_total_return = (
        (overall_end_equity / overall_start_equity - 1.0)
        if overall_start_equity > 0
        else 0.0
    )

    for i in range(_N_QUARTILES):
        q_t_start = boundaries[i]
        q_t_end = boundaries[i + 1]

        eq_start = _equity_at(q_t_start)
        eq_end = _equity_at(q_t_end)

        q_return = (eq_end / eq_start - 1.0) if eq_start > 0 else 0.0

        # Max drawdown: use equity points strictly inside the quartile window
        # plus the carry-forward values at the quartile boundaries.
        q_points = [ep for ep in sorted_curve if q_t_start <= ep.t <= q_t_end]
        # Ensure boundary values are represented.
        if not q_points or q_points[0].t > q_t_start:
            q_points = [EquityPoint(t=int(q_t_start), equity=eq_start)] + q_points
        if not q_points or q_points[-1].t < q_t_end:
            q_points = q_points + [EquityPoint(t=int(q_t_end), equity=eq_end)]

        q_max_dd = _max_drawdown_local(q_points)

        quartile_blocks.append({
            "quartile": i + 1,
            "t_start": int(q_t_start),
            "t_end": int(q_t_end),
            "num_trades": trade_counts[i],
            "total_return": q_return,
            "max_drawdown": q_max_dd,
        })

    # Stability summary.
    overall_positive = overall_total_return >= 0
    sign_matches = sum(
        1 for q in quartile_blocks
        if (q["total_return"] >= 0) == overall_positive
    )
    return_sign_consistency = sign_matches / _N_QUARTILES

    worst_quartile_return = min(q["total_return"] for q in quartile_blocks)

    return {
        "quartiles": quartile_blocks,
        "stability": {
            "return_sign_consistency": return_sign_consistency,
            "worst_quartile_return": worst_quartile_return,
        },
    }


def _max_drawdown_local(points: list[EquityPoint]) -> float:
    """Max drawdown over a list of equity points (same formula as standard.py)."""
    if not points:
        return 0.0
    peak = points[0].equity
    max_dd = 0.0
    for p in points:
        if p.equity > peak:
            peak = p.equity
        if peak > 0:
            dd = (peak - p.equity) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd
