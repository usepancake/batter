"""Portfolio receipt — Wave E (0.10.0).

Combine N strategy BacktestResults with allocation weights into a joint equity
curve + portfolio metrics.  This is a NEW artifact class with its own
``portfolio_hash``; individual ``result_hash`` values on the legs are pinned as
provenance — they never change.

Design: docs/design-0.9.0-contracts-and-fills.md §5 Wave E.

Hash policy:
  - Each leg's ``result_hash`` is included verbatim — tamper with a leg and the
    portfolio_hash breaks.
  - portfolio_hash = sha256_canonical(all fields of to_dict() minus
    portfolio_hash itself) so it is self-contained.
  - regime / additive non-hashed leg fields are NOT included.
"""

from __future__ import annotations

import math
from math import fsum
from typing import Any

from .hash import sha256_canonical
from .metrics.series import daily_returns_carry_forward
from .metrics.standard import (
    _max_drawdown,           # reuse internal; same module boundary
    sharpe_ratio,
    sortino_ratio,
    total_return,
)
from .result import BacktestResult, EquityPoint

__all__ = [
    "PortfolioResult",
    "PortfolioError",
    "compute_portfolio",
]


class PortfolioError(ValueError):
    """Typed error for portfolio validation failures."""


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class PortfolioResult:
    """Immutable portfolio receipt — the aggregate of N weighted strategy legs.

    Attributes
    ----------
    format_version:   "portfolio/1"
    engine:           copied from legs (all legs must match)
    engine_version:   copied from legs
    leg_result_hashes: result_hash of each leg in insertion order (provenance)
    weights:          allocation fractions (positive, sum 1.0 by construction)
    metrics:          portfolio-level metrics dict (total_return, max_drawdown,
                      sharpe, sortino, num_legs, per_leg_metrics list)
    correlation_matrix: pairwise Pearson of leg daily returns (None on degenerate legs)
    joint_equity_curve: list of {"t": int, "equity": float} on the union of leg timestamps
    portfolio_hash:   sha256_canonical over all of to_dict() minus portfolio_hash itself
    """

    def __init__(
        self,
        *,
        format_version: str,
        engine: str,
        engine_version: str,
        leg_result_hashes: list[str],
        weights: list[float],
        metrics: dict[str, Any],
        correlation_matrix: list[list[float | None]],
        joint_equity_curve: list[dict[str, Any]],
        portfolio_hash: str,
    ) -> None:
        self.format_version = format_version
        self.engine = engine
        self.engine_version = engine_version
        self.leg_result_hashes = leg_result_hashes
        self.weights = weights
        self.metrics = metrics
        self.correlation_matrix = correlation_matrix
        self.joint_equity_curve = joint_equity_curve
        self.portfolio_hash = portfolio_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "engine": self.engine,
            "engine_version": self.engine_version,
            "leg_result_hashes": self.leg_result_hashes,
            "weights": self.weights,
            "metrics": self.metrics,
            "correlation_matrix": self.correlation_matrix,
            "joint_equity_curve": self.joint_equity_curve,
            "portfolio_hash": self.portfolio_hash,
        }


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_portfolio(
    legs: list[BacktestResult],
    weights: list[float],
) -> PortfolioResult:
    """Combine N strategy BacktestResults with allocation weights.

    Parameters
    ----------
    legs:    ≥2 BacktestResult values, all must be ok (non-blocked) and share
             the same (engine, engine_version) identity.
    weights: positive floats that sum exactly to 1.0 (tolerance 1e-12 via fsum).

    Returns a PortfolioResult with:
    - joint equity curve in return space (start=1.0 for the portfolio),
    - portfolio total_return / max_drawdown / sharpe / sortino,
    - per-leg weight echoes,
    - pairwise Pearson correlation matrix.

    Raises PortfolioError on any validation failure.
    """
    _validate_legs(legs)
    _validate_weights(weights, len(legs))

    n = len(legs)
    engine = legs[0].engine
    engine_version = legs[0].engine_version

    # --- 1. Normalise each leg's equity curve to start at 1.0 ---
    # "Start" = first equity point in the leg.
    # The portfolio is in pure-return space; we don't pick a joint capital.
    normalised: list[list[EquityPoint]] = []
    for leg in legs:
        curve = sorted(leg.equity_curve, key=lambda p: p.t)
        start = curve[0].equity if curve else 1.0
        if start == 0.0:
            start = 1.0  # degenerate: treat as flat
        normalised.append([EquityPoint(t=p.t, equity=p.equity / start) for p in curve])

    # --- 2. Union of timestamps ---
    all_ts: list[int] = sorted({p.t for curve in normalised for p in curve})

    # --- 3. Carry-forward each leg to every union timestamp ---
    def _carry_forward(curve: list[EquityPoint], timestamps: list[int]) -> list[float]:
        """Return leg equity at each timestamp using last-value carry-forward."""
        idx = 0
        last = curve[0].equity
        result: list[float] = []
        for ts in timestamps:
            while idx + 1 < len(curve) and curve[idx + 1].t <= ts:
                idx += 1
                last = curve[idx].equity
            # Also pick up the exact point if the first point is at/before ts.
            if curve[0].t <= ts:
                result.append(last)
            else:
                # ts is before this leg's start — carry forward the opening value
                result.append(curve[0].equity)
        return result

    leg_values: list[list[float]] = [
        _carry_forward(normalised[i], all_ts) for i in range(n)
    ]

    # --- 4. Joint equity = Σ w_i * leg_i(t) ---
    joint_equity: list[float] = []
    for j in range(len(all_ts)):
        val = fsum(weights[i] * leg_values[i][j] for i in range(n))
        joint_equity.append(val)

    joint_curve = [EquityPoint(t=all_ts[j], equity=joint_equity[j]) for j in range(len(all_ts))]

    # --- 5. Portfolio metrics ---
    joint_daily = daily_returns_carry_forward(joint_curve)
    port_total_return = total_return(joint_curve[0].equity, joint_curve[-1].equity)
    port_max_dd = _max_drawdown(joint_curve)
    port_sharpe = sharpe_ratio(joint_daily)
    port_sortino = sortino_ratio(joint_daily)

    # Per-leg metrics summary
    per_leg_metrics = []
    for i, leg in enumerate(legs):
        std = leg.metrics.standard
        per_leg_metrics.append({
            "leg_index": i,
            "result_hash": leg.result_hash,
            "weight": weights[i],
            "total_return": std.total_return,
            "max_drawdown": std.max_drawdown,
            "sharpe": std.sharpe,
            "num_trades": std.num_trades,
        })

    metrics: dict[str, Any] = {
        "total_return": port_total_return,
        "max_drawdown": port_max_dd,
        "sharpe": port_sharpe,
        "sortino": port_sortino,
        "num_legs": n,
        "per_leg": per_leg_metrics,
    }

    # --- 6. Pairwise Pearson correlation matrix ---
    leg_daily_returns: list[list[float]] = []
    for i in range(n):
        leg_curve = sorted(legs[i].equity_curve, key=lambda p: p.t)
        leg_daily_returns.append(daily_returns_carry_forward(leg_curve))

    correlation_matrix: list[list[float | None]] = []
    for i in range(n):
        row: list[float | None] = []
        for j in range(n):
            if i == j:
                row.append(1.0)
            else:
                row.append(_pearson(leg_daily_returns[i], leg_daily_returns[j]))
        correlation_matrix.append(row)

    # --- 7. Hash ---
    joint_curve_dicts = [{"t": p.t, "equity": p.equity} for p in joint_curve]
    leg_hashes = [leg.result_hash for leg in legs]

    pre_hash_payload: dict[str, Any] = {
        "format_version": "portfolio/1",
        "engine": engine,
        "engine_version": engine_version,
        "leg_result_hashes": leg_hashes,
        "weights": weights,
        "metrics": metrics,
        "correlation_matrix": correlation_matrix,
        "joint_equity_curve": joint_curve_dicts,
    }
    portfolio_hash = sha256_canonical(pre_hash_payload)

    return PortfolioResult(
        format_version="portfolio/1",
        engine=engine,
        engine_version=engine_version,
        leg_result_hashes=leg_hashes,
        weights=weights,
        metrics=metrics,
        correlation_matrix=correlation_matrix,
        joint_equity_curve=joint_curve_dicts,
        portfolio_hash=portfolio_hash,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_legs(legs: list[BacktestResult]) -> None:
    if len(legs) < 2:
        raise PortfolioError(
            f"compute_portfolio requires ≥2 legs; got {len(legs)}"
        )

    # All legs must be non-blocked (ok).
    for i, leg in enumerate(legs):
        if not leg.validation.ok:
            raise PortfolioError(
                f"leg {i} is blocked (validation failed); only ok results can be combined"
            )
        if not leg.equity_curve:
            raise PortfolioError(
                f"leg {i} has an empty equity_curve; cannot compute portfolio"
            )

    # All legs must share the same engine identity.
    ref_engine = legs[0].engine
    ref_version = legs[0].engine_version
    for i, leg in enumerate(legs[1:], start=1):
        if leg.engine != ref_engine or leg.engine_version != ref_version:
            raise PortfolioError(
                f"leg {i} has engine identity ({leg.engine!r}, {leg.engine_version!r}) "
                f"which differs from leg 0 ({ref_engine!r}, {ref_version!r}); "
                "all legs must share the same engine identity"
            )


def _validate_weights(weights: list[float], n_legs: int) -> None:
    if len(weights) != n_legs:
        raise PortfolioError(
            f"weights length {len(weights)} does not match legs length {n_legs}"
        )
    for i, w in enumerate(weights):
        if not isinstance(w, (int, float)) or isinstance(w, bool):
            raise PortfolioError(f"weight[{i}]={w!r} is not a number")
        if not math.isfinite(w) or w <= 0:
            raise PortfolioError(
                f"weight[{i}]={w} is not positive; all weights must be > 0"
            )
    weight_sum = fsum(weights)
    if abs(weight_sum - 1.0) > 1e-12:
        raise PortfolioError(
            f"weights must sum to exactly 1.0 (tolerance 1e-12); "
            f"got sum={weight_sum!r} (deviation={abs(weight_sum - 1.0)!r})"
        )


# ---------------------------------------------------------------------------
# Pearson correlation
# ---------------------------------------------------------------------------


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson r of xs and ys.

    Returns None when either sequence has <2 observations or zero variance.
    Uses fsum arithmetic for cross-runtime stability.
    """
    if len(xs) < 2 or len(ys) < 2:
        return None

    # Pair only the common-length prefix (legs may differ in length due to
    # different trading horizons; we pair by position in the carry-forward
    # daily series — same convention used for joint equity above).
    n = min(len(xs), len(ys))
    if n < 2:
        return None
    xs = xs[:n]
    ys = ys[:n]

    mean_x = fsum(xs) / n
    mean_y = fsum(ys) / n

    cov = fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = fsum((x - mean_x) ** 2 for x in xs)
    var_y = fsum((y - mean_y) ** 2 for y in ys)

    if var_x == 0.0 or var_y == 0.0:
        return None

    denom = math.sqrt(var_x * var_y)
    if denom == 0.0:
        return None

    r = cov / denom
    # Clamp to [-1, 1] to absorb float rounding at exact ±1.
    return max(-1.0, min(1.0, r))
