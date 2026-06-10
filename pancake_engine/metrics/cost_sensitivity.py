"""Transaction-cost sensitivity + break-even cost multiplier (0.8.0).

Re-prices the realised trade log under scaled slippage + fees — NO re-simulation —
and reports net performance at a few cost multiples plus the break-even multiplier
(the cost level at which the strategy's mean trade return crosses zero). The
break-even multiplier is the single most honest number for a trader whose real
costs differ from the spec's.

Method (trade-level; the realised sizing path is held fixed, so this is exact at
the per-trade level and does not model the equity-compounding feedback of changed
costs — that would require a re-simulation, which this deliberately avoids):

For each trade, recover the original slippage and fee from the stored fields and
re-price at multiplier ``k``:

    fill_k    = quote + k·(fill₁ − quote)        # slippage scales linearly off the quote
    fee_k     = k·fee₁                            # fee_bps scales linearly
    shares_k  = (cost − fee_k) / fill_k
    pnl_k     = shares_k·settle − cost            # settle ∈ {0, 1}; losers pay −cost at every k

``k = 1`` reproduces the stored trade exactly. Mean trade return is monotone
non-increasing in ``k`` (winners shrink, losers are flat), so the break-even
multiplier is the unique root, found by bisection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..runner.trade import Trade

__all__ = ["cost_sensitivity", "CostSensitivityResult", "CostSensitivityPoint"]

_DEFAULT_MULTIPLIERS = (0.0, 0.5, 1.0, 2.0, 5.0)
_BREAK_EVEN_CAP = 50.0  # if still profitable at 50× cost, report None (robust to any cost)


@dataclass(frozen=True)
class CostSensitivityPoint:
    multiplier: float
    mean_return: float       # mean per-trade return (fraction) at this cost multiple
    total_pnl: float
    n_trades: int


@dataclass(frozen=True)
class CostSensitivityResult:
    points: list[CostSensitivityPoint]
    break_even_multiplier: float | None  # cost× where mean trade return crosses 0


def _rescaled_pnl(t: Trade, k: float) -> float | None:
    """P&L of trade ``t`` if slippage and fees were scaled by ``k``. None if degenerate."""
    quote = t.entry_price_quote
    fill_orig = t.entry_price
    cost = t.cost
    if cost <= 0.0 or quote <= 0.0:
        return None
    fee_orig = cost - t.shares * fill_orig  # investable = shares · fill
    fill_k = quote + k * (fill_orig - quote)
    investable_k = cost - k * fee_orig
    if fill_k <= 0.0 or investable_k <= 0.0:
        return None
    shares_k = investable_k / fill_k
    settle = t.exit_price  # 0.0 or 1.0
    return shares_k * settle - cost


def _mean_return(trades: list[Trade], k: float) -> float | None:
    rets = []
    for t in trades:
        pnl = _rescaled_pnl(t, k)
        if pnl is not None and t.cost > 0.0:
            rets.append(pnl / t.cost)
    return (math.fsum(rets) / len(rets)) if rets else None


def _break_even_multiplier(trades: list[Trade]) -> float | None:
    m0 = _mean_return(trades, 0.0)
    if m0 is None:
        return None
    if m0 <= 0.0:
        return 0.0  # unprofitable even with zero costs
    lo, hi = 0.0, 1.0  # mean(lo) > 0
    mhi = _mean_return(trades, hi)
    while mhi is not None and mhi > 0.0:
        hi *= 2.0
        if hi > _BREAK_EVEN_CAP:
            return None  # profitable beyond the tested cost ceiling
        mhi = _mean_return(trades, hi)
    if mhi is None:
        return None
    for _ in range(60):  # bisection: mean(lo) > 0 >= mean(hi)
        mid = (lo + hi) / 2.0
        mm = _mean_return(trades, mid)
        if mm is None:
            return None
        if mm > 0.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def cost_sensitivity(
    trades: list[Trade], multipliers: tuple[float, ...] = _DEFAULT_MULTIPLIERS
) -> CostSensitivityResult:
    """Net performance across cost multiples + the break-even multiplier.

    ``multipliers`` are applied to the spec's declared slippage + fees (1.0 = as
    declared, 0.0 = frictionless). Empty trade log → empty curve, no break-even.
    """
    points: list[CostSensitivityPoint] = []
    for k in multipliers:
        pnls = [p for t in trades if (p := _rescaled_pnl(t, k)) is not None]
        rets = [
            _rescaled_pnl(t, k) / t.cost
            for t in trades
            if t.cost > 0.0 and _rescaled_pnl(t, k) is not None
        ]
        points.append(
            CostSensitivityPoint(
                multiplier=k,
                mean_return=(math.fsum(rets) / len(rets)) if rets else 0.0,
                total_pnl=math.fsum(pnls),
                n_trades=len(pnls),
            )
        )
    return CostSensitivityResult(
        points=points,
        break_even_multiplier=_break_even_multiplier(trades) if trades else None,
    )
