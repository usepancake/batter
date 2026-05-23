"""Standard metric block: total_return, cagr, sharpe, sortino, max_drawdown, win_rate.

Engine 0.3 is correctness-first, not TS parity. Locked constants pulled from
``pancake-production/lib/backtest/metrics.ts``:

- ``SECONDS_PER_YEAR = 365.25 * 86400`` (Julian year)
- ``ANNUALIZATION_DAYS = 252`` (trading-day convention even for prediction markets)
- ``stdev`` uses Bessel correction (``N-1``)
- ``win_rate`` uses strict ``pnl > 0``

Divergences from TS:

- **Sortino denominator**: TS divides by ``len(negs)``; Engine 0.3 divides by
  the full sample size ``N`` (true Sortino with target = 0). See D-13.
- **Sharpe / Sortino with ``n < 2``**: TS returns ``0``; Engine 0.3 returns
  ``None`` (no information).
- **win_rate with empty trades**: TS returns ``0``; Engine 0.3 returns ``None``.
- **CAGR ruined case**: TS computes ``(0/start)^(1/y) - 1 = -1``; Engine 0.3
  explicit piecewise returns ``-1.0`` and emits ``RUINED``.
"""

from __future__ import annotations

import math
from typing import Optional

from ..result import EquityPoint, MetricsStandard
from ..runner.trade import Trade

__all__ = ["compute_standard", "SECONDS_PER_YEAR", "ANNUALIZATION_DAYS"]

SECONDS_PER_DAY = 86_400
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY     # matches TS metrics.ts
ANNUALIZATION_DAYS = 252                         # matches TS metrics.ts
MIN_YEAR_FRACTION = 1 / 365                      # matches TS cagr floor


def total_return(start: float, end: float) -> float:
    if start == 0:
        return 0.0
    return end / start - 1.0


def cagr_piecewise(
    *,
    num_trades: int,
    starting_capital: float,
    ending_equity: float,
    period_seconds: int,
) -> tuple[Optional[float], bool, bool]:
    """Piecewise CAGR. Returns ``(cagr, ruined, overflowed)``.

    ``ruined=True`` indicates ``RUINED`` warning should be emitted.
    ``overflowed=True`` indicates ``CAGR_EXTRAPOLATION_OVERFLOW`` warning should
    be emitted and ``cagr`` is ``None``. Overflow happens when
    ``(ending/starting)^(1/year_fraction)`` exceeds float64 max — e.g., a 20×
    return in 1 day under the ``year_fraction = 1/365`` floor pushes the
    extrapolation past ``1.8e308``. The original input is still recoverable
    via ``total_return``, ``starting_capital``, ``ending_capital``.
    """
    if num_trades == 0:
        return 0.0, False, False
    if starting_capital <= 0:
        # Validation should catch first; defensive.
        return 0.0, False, False
    if ending_equity <= 0:
        return -1.0, True, False
    years = max(period_seconds / SECONDS_PER_YEAR, MIN_YEAR_FRACTION)
    try:
        return (ending_equity / starting_capital) ** (1.0 / years) - 1.0, False, False
    except OverflowError:
        return None, False, True


def sharpe_ratio(daily_returns: list[float]) -> Optional[float]:
    """Annualized Sharpe with rf=0. ``None`` if n<2 or std=0."""
    if len(daily_returns) < 2:
        return None
    mean = _mean(daily_returns)
    std = _stdev_sample(daily_returns, mean)
    if std == 0:
        return None
    return (mean / std) * math.sqrt(ANNUALIZATION_DAYS)


def sortino_ratio(daily_returns: list[float]) -> Optional[float]:
    """Annualized Sortino with rf=0 and target=0.

    Denominator: ``sqrt(Σ_{r<0} r² / N)`` — divides by the full sample size
    (true Sortino), **not** by the count of negatives. This diverges from TS,
    which divides by ``len(negs)``. Documented as D-13.

    ``None`` if n<2, no negative observations, or denominator is 0.
    """
    if len(daily_returns) < 2:
        return None
    negs = [r for r in daily_returns if r < 0]
    if not negs:
        return None
    n = len(daily_returns)
    downside_var = sum(r * r for r in negs) / n
    ds = math.sqrt(downside_var)
    if ds == 0:
        return None
    mean = _mean(daily_returns)
    return (mean / ds) * math.sqrt(ANNUALIZATION_DAYS)


def win_rate_strict(trades: list[Trade]) -> Optional[float]:
    """Strict ``pnl > 0`` win-rate.

    Returns ``None`` if no trades (TS returns ``0`` — divergence documented).
    """
    if not trades:
        return None
    return sum(1 for t in trades if t.pnl > 0) / len(trades)


def compute_standard(
    *,
    trades: list[Trade],
    equity_curve: list[EquityPoint],
    daily_rets: list[float],
    starting_capital: float,
    period_seconds: int,
) -> tuple[MetricsStandard, bool, bool]:
    """Return ``(MetricsStandard, ruined_flag, cagr_overflowed_flag)``."""
    ending_equity = equity_curve[-1].equity if equity_curve else starting_capital
    tr = total_return(starting_capital, ending_equity)
    cg, ruined, overflowed = cagr_piecewise(
        num_trades=len(trades),
        starting_capital=starting_capital,
        ending_equity=ending_equity,
        period_seconds=period_seconds,
    )
    sh = sharpe_ratio(daily_rets)
    so = sortino_ratio(daily_rets)
    wr = win_rate_strict(trades)
    # max_drawdown is computed by build_drawdown_curve; passed via series module.
    # We compute it again here from equity_curve for the standard metric.
    max_dd = _max_drawdown(equity_curve)
    standard = MetricsStandard(
        total_return=tr,
        cagr=cg,
        sharpe=sh,
        sortino=so,
        max_drawdown=max_dd,
        win_rate=wr,
        num_trades=len(trades),
        starting_capital=float(starting_capital),
        ending_capital=float(ending_equity),
    )
    return standard, ruined, overflowed


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _stdev_sample(xs: list[float], mean: float) -> float:
    """Sample stdev with Bessel correction (n-1). Matches TS."""
    if len(xs) < 2:
        return 0.0
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def _max_drawdown(equity_curve: list[EquityPoint]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0].equity
    max_dd = 0.0
    for p in equity_curve:
        if p.equity > peak:
            peak = p.equity
        if peak > 0:
            dd = (peak - p.equity) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd
