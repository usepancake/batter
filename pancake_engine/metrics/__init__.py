"""Metrics layer for Pancake Engine 0.3.

Two metric blocks:
    ``standard`` — total_return / cagr / sharpe / sortino / max_drawdown / win_rate
    ``pm``      — prediction-market-native: Wilson CI on win_rate, brier_crowd,
                  brier_strategy (null in PR-1 — rule-based spec)

Per-trade PM fields (implied_prob_at_entry, payoff_per_unit, etc.) are emitted
on the ``Trade`` records by the runner, not in this layer.
"""

from .credibility import emit_credibility_warnings
from .pm import compute_pm
from .series import build_drawdown_curve, build_monthly_returns, daily_returns_carry_forward
from .standard import compute_standard

__all__ = [
    "compute_standard",
    "compute_pm",
    "emit_credibility_warnings",
    "build_drawdown_curve",
    "build_monthly_returns",
    "daily_returns_carry_forward",
]
