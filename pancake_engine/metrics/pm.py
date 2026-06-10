"""Prediction-market-native metrics.

Per-trade PM quantities are computed by the runner and stored on ``Trade``.
This module computes the strategy-level aggregates: Wilson 95% CI on win_rate,
Brier scores, calibration ECE, and related signals.

Engine 0.3 is correctness-first, not TS parity. ``brier_strategy`` is ``None``
for rule-based EvidenceSpecs (no independent probability emitted) and a
``BRIER_NOT_APPLICABLE`` info-warning is emitted at the caller. ``brier_crowd``
is always computable when at least one trade exists.
"""

from __future__ import annotations

import math
from typing import Optional

from ..result import MetricsPM
from ..runner.trade import Trade

__all__ = [
    "compute_pm",
    "wilson_ci95",
    "brier_crowd_score",
    "calibration_ece",
    "calibration_bins",
    "implied_prob_at_entry",
    "realized_outcome_for_trade",
]

# ECE bin edges: 10 fixed bins [0,0.1)…[0.8,0.9),[0.9,1.0].
# Last bin is CLOSED at 1.0 to capture implied_prob == 1.0 edge.
_ECE_EDGES = [i / 10 for i in range(11)]  # [0.0, 0.1, 0.2, …, 1.0]
_ECE_MIN_TRADES = 10  # deterministic threshold; None when num_trades < this

Z95 = 1.959963984540054  # standard normal 97.5 percentile


def implied_prob_at_entry(trade: Trade) -> float:
    """Side-aware implied probability at decision time.

    For the side traded, implied prob == ``entry_price_quote`` (pre-slip).
    No inversion (locked by v1.3 dogfood).
    """
    return float(trade.entry_price_quote)


def realized_outcome_for_trade(trade: Trade) -> int:
    """1 if the strategy won this trade, else 0.

    Derived from ``exit_price``: settle_value = 1 means the strategy won
    (proceeds = shares × 1). settle_value = 0 means it lost.
    """
    return 1 if trade.exit_price >= 1.0 else 0


def wilson_ci95(wins: int, n: int) -> tuple[Optional[float], Optional[float]]:
    """Wilson 95% confidence interval for binomial proportion ``wins/n``.

    Returns ``(None, None)`` if ``n == 0`` (no information — not ``(0, 1)``).
    Architecture decision M-1.
    """
    if n == 0:
        return None, None
    p_hat = wins / n
    z = Z95
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))) / denom
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return low, high


def calibration_ece(trades: list[Trade]) -> Optional[float]:
    """Expected Calibration Error (ECE) over 10 fixed probability bins.

    Both ``implied_prob_at_entry`` and ``realized_outcome_for_trade`` are on the
    traded-side axis — no inversion (locked by the NO-price convention).

    Returns ``None`` when ``len(trades) < 10`` (deterministic threshold).
    Bins: [0,0.1),[0.1,0.2),…,[0.8,0.9),[0.9,1.0] (last bin closed at 1.0).
    ECE = Σ (n_b/N) · |acc_b − conf_b|  via ``math.fsum`` — pure arithmetic.
    """
    if len(trades) < _ECE_MIN_TRADES:
        return None

    n_total = len(trades)
    # Accumulate per-bin sums using lists for fsum.
    bin_probs: list[list[float]] = [[] for _ in range(10)]
    bin_outcomes: list[list[float]] = [[] for _ in range(10)]

    for t in trades:
        p = implied_prob_at_entry(t)
        o = float(realized_outcome_for_trade(t))
        # Determine bin index: floor(p * 10), clamped so that p==1.0 lands in bin 9.
        b = min(int(p * 10), 9)
        bin_probs[b].append(p)
        bin_outcomes[b].append(o)

    weighted_abs_errors: list[float] = []
    for b in range(10):
        n_b = len(bin_probs[b])
        if n_b == 0:
            continue
        conf_b = math.fsum(bin_probs[b]) / n_b
        acc_b = math.fsum(bin_outcomes[b]) / n_b
        weighted_abs_errors.append((n_b / n_total) * abs(acc_b - conf_b))

    return math.fsum(weighted_abs_errors)


def calibration_bins(trades: list[Trade]) -> Optional[list[dict]]:
    """Per-bin calibration data (same threshold as ``calibration_ece``).

    Returns ``None`` when ``len(trades) < 10``.
    Each non-empty bin dict: {bin_low, bin_high, n, confidence, accuracy}.
    """
    if len(trades) < _ECE_MIN_TRADES:
        return None

    bin_probs: list[list[float]] = [[] for _ in range(10)]
    bin_outcomes: list[list[float]] = [[] for _ in range(10)]

    for t in trades:
        p = implied_prob_at_entry(t)
        o = float(realized_outcome_for_trade(t))
        b = min(int(p * 10), 9)
        bin_probs[b].append(p)
        bin_outcomes[b].append(o)

    result = []
    for b in range(10):
        n_b = len(bin_probs[b])
        if n_b == 0:
            continue
        conf_b = math.fsum(bin_probs[b]) / n_b
        acc_b = math.fsum(bin_outcomes[b]) / n_b
        result.append({
            "bin_low": _ECE_EDGES[b],
            "bin_high": _ECE_EDGES[b + 1],
            "n": n_b,
            "confidence": conf_b,
            "accuracy": acc_b,
        })
    return result


def brier_crowd_score(trades: list[Trade]) -> Optional[float]:
    """``mean((implied_prob_at_entry - realized_outcome)²)`` over all trades.

    ``None`` when no trades.
    """
    if not trades:
        return None
    total = 0.0
    for t in trades:
        p = implied_prob_at_entry(t)
        o = realized_outcome_for_trade(t)
        diff = p - o
        total += diff * diff
    return total / len(trades)


def compute_pm(
    *,
    trades: list[Trade],
    sharpe_equity_curve: Optional[float],
) -> MetricsPM:
    """Strategy-level PM metric aggregates."""
    if not trades:
        return MetricsPM(
            win_rate_ci95_low=None,
            win_rate_ci95_high=None,
            mean_return_pct=None,
            std_return_pct=None,
            sharpe_trade_level=None,
            sharpe_equity_curve=sharpe_equity_curve,
            brier_strategy=None,
            brier_crowd=None,
            brier_skill_score=None,
            mean_edge=None,
        )

    wins = sum(1 for t in trades if t.pnl > 0)
    ci_low, ci_high = wilson_ci95(wins, len(trades))

    returns = [t.return_pct for t in trades]
    mean_r = math.fsum(returns) / len(returns)
    if len(returns) >= 2:
        var = math.fsum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r: Optional[float] = math.sqrt(var)
    else:
        std_r = None

    sharpe_trade: Optional[float] = None
    if std_r is not None and std_r > 0:
        sharpe_trade = mean_r / std_r  # trade-level, NOT annualized (cadence unknown)

    brier_crowd = brier_crowd_score(trades)
    # brier_strategy is null in PR-1 — rule-based spec emits no independent probability.
    brier_strategy = None
    brier_skill = None  # 1 − strategy/crowd; null when strategy is null

    ece = calibration_ece(trades)

    return MetricsPM(
        win_rate_ci95_low=ci_low,
        win_rate_ci95_high=ci_high,
        mean_return_pct=mean_r,
        std_return_pct=std_r,
        sharpe_trade_level=sharpe_trade,
        sharpe_equity_curve=sharpe_equity_curve,
        brier_strategy=brier_strategy,
        brier_crowd=brier_crowd,
        brier_skill_score=brier_skill,
        mean_edge=None,  # null in PR-1 (no fair_probability column)
        calibration_ece=ece,
    )
