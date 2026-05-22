"""Aggregate metrics + dispersion warnings across folds."""

from __future__ import annotations

import math
from typing import Optional

from ..metrics.pm import brier_crowd_score, wilson_ci95
from ..warnings import Severity, Warning, WarningCode
from .result import (
    AggregateMetrics,
    Fold,
    FoldMeanMetrics,
    FoldStdMetrics,
    PooledMetrics,
)

__all__ = ["compute_aggregate", "emit_aggregate_warnings"]


def compute_aggregate(folds: list[Fold]) -> AggregateMetrics:
    """Compute pooled / fold-mean / fold-std metrics across folds."""
    all_trades = [t for f in folds for t in f.result.trades]
    non_empty = [f for f in folds if f.result.metrics.standard.num_trades > 0]

    # Pooled
    if all_trades:
        wins = sum(1 for t in all_trades if t.pnl > 0)
        win_rate = wins / len(all_trades)
        returns = [t.return_pct for t in all_trades]
        mean_r = sum(returns) / len(returns)
        if len(returns) >= 2:
            var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
            std_r: Optional[float] = math.sqrt(var)
        else:
            std_r = None
        sharpe_trade = mean_r / std_r if std_r and std_r > 0 else None
        brier = brier_crowd_score(all_trades)
        pooled = PooledMetrics(
            num_trades=len(all_trades),
            win_rate=win_rate,
            mean_return_pct=mean_r,
            std_return_pct=std_r,
            sharpe_trade_level=sharpe_trade,
            brier_crowd=brier,
        )
    else:
        pooled = PooledMetrics(
            num_trades=0,
            win_rate=None,
            mean_return_pct=None,
            std_return_pct=None,
            sharpe_trade_level=None,
            brier_crowd=None,
        )

    # Fold-mean / fold-std
    def _fold_scalar(getter, only_non_empty: bool = False) -> tuple[Optional[float], Optional[float]]:
        source = non_empty if only_non_empty else folds
        vals = [getter(f) for f in source if getter(f) is not None]
        if not vals:
            return None, None
        mean = sum(vals) / len(vals)
        if len(vals) < 2:
            return mean, None
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        return mean, math.sqrt(var)

    tr_mean, tr_std = _fold_scalar(lambda f: f.result.metrics.standard.total_return)
    sh_mean, sh_std = _fold_scalar(lambda f: f.result.metrics.standard.sharpe, only_non_empty=True)
    so_mean, so_std = _fold_scalar(lambda f: f.result.metrics.standard.sortino, only_non_empty=True)
    md_mean, md_std = _fold_scalar(lambda f: f.result.metrics.standard.max_drawdown)
    wr_mean, wr_std = _fold_scalar(lambda f: f.result.metrics.standard.win_rate, only_non_empty=True)
    nt_mean, nt_std = _fold_scalar(lambda f: float(f.result.metrics.standard.num_trades))

    fold_mean = FoldMeanMetrics(
        total_return=tr_mean,
        sharpe=sh_mean,
        sortino=so_mean,
        max_drawdown=md_mean,
        win_rate=wr_mean,
        num_trades=nt_mean if nt_mean is not None else 0.0,
    )
    fold_std = FoldStdMetrics(
        total_return=tr_std,
        sharpe=sh_std,
        sortino=so_std,
        max_drawdown=md_std,
        win_rate=wr_std,
        num_trades=nt_std,
    )

    # Dispersion
    sharpe_vals = [f.result.metrics.standard.sharpe for f in non_empty
                   if f.result.metrics.standard.sharpe is not None]
    fold_sharpe_dispersion: Optional[float] = None
    if len(sharpe_vals) >= 2:
        s_mean = sum(sharpe_vals) / len(sharpe_vals)
        if abs(s_mean) > 0:
            s_var = sum((v - s_mean) ** 2 for v in sharpe_vals) / (len(sharpe_vals) - 1)
            fold_sharpe_dispersion = math.sqrt(s_var) / abs(s_mean)

    wr_vals = [f.result.metrics.standard.win_rate for f in non_empty
               if f.result.metrics.standard.win_rate is not None]
    fold_win_rate_dispersion: Optional[float] = None
    if len(wr_vals) >= 2:
        fold_win_rate_dispersion = max(wr_vals) - min(wr_vals)

    return AggregateMetrics(
        fold_count=len(folds),
        non_empty_fold_count=len(non_empty),
        pooled=pooled,
        fold_mean=fold_mean,
        fold_std=fold_std,
        fold_sharpe_dispersion=fold_sharpe_dispersion,
        fold_win_rate_dispersion=fold_win_rate_dispersion,
    )


def emit_aggregate_warnings(folds: list[Fold], agg: AggregateMetrics) -> list[Warning]:
    """Aggregate-level warnings: dispersion, sign-flip, single-fold-carries,
    unequal-fold-size."""
    out: list[Warning] = []
    non_empty = [f for f in folds if f.result.metrics.standard.num_trades > 0]

    # WALKFORWARD_DISPERSION_HIGH — overfit signal
    if agg.fold_sharpe_dispersion is not None and agg.fold_sharpe_dispersion > 2.0:
        out.append(Warning(
            code=WarningCode.WALKFORWARD_DISPERSION_HIGH,
            severity=Severity.WARN,
            message=(
                f"Per-fold Sharpe dispersion = {agg.fold_sharpe_dispersion:.2f} (> 2): "
                "high fold-to-fold variability suggests overfitting."
            ),
            context={"fold_sharpe_dispersion": agg.fold_sharpe_dispersion},
        ))

    # UNEQUAL_FOLD_SIZE
    nz_counts = [f.result.metrics.standard.num_trades for f in non_empty]
    if nz_counts and max(nz_counts) > 3 * min(nz_counts):
        out.append(Warning(
            code=WarningCode.UNEQUAL_FOLD_SIZE,
            severity=Severity.WARN,
            message=(
                f"Largest non-empty fold ({max(nz_counts)} trades) is more than 3× "
                f"the smallest ({min(nz_counts)})."
            ),
            context={"max": max(nz_counts), "min": min(nz_counts)},
        ))

    # WALKFORWARD_SIGN_FLIP — any non-empty fold opposite-sign vs pooled total pnl
    pooled_pnl = sum(t.pnl for f in folds for t in f.result.trades)
    if pooled_pnl != 0:
        pooled_sign = 1 if pooled_pnl > 0 else -1
        for f in non_empty:
            fold_pnl = sum(t.pnl for t in f.result.trades)
            if fold_pnl * pooled_sign < 0:
                out.append(Warning(
                    code=WarningCode.WALKFORWARD_SIGN_FLIP,
                    severity=Severity.WARN,
                    message=(
                        f"Fold {f.definition.index} pnl sign-flips vs pooled aggregate "
                        "— results may be regime-dependent."
                    ),
                    context={"fold_index": f.definition.index, "fold_pnl": fold_pnl,
                             "pooled_pnl": pooled_pnl},
                ))
                break  # one warning per WF run, not per-fold

    # WALKFORWARD_SINGLE_FOLD_CARRIES — one fold > 70% of |pooled pnl|
    if pooled_pnl != 0 and non_empty:
        fold_pnls = [(f.definition.index, sum(t.pnl for t in f.result.trades)) for f in folds]
        max_abs_idx, max_abs_pnl = max(fold_pnls, key=lambda p: abs(p[1]))
        if abs(max_abs_pnl) > 0.7 * abs(pooled_pnl):
            out.append(Warning(
                code=WarningCode.WALKFORWARD_SINGLE_FOLD_CARRIES,
                severity=Severity.WARN,
                message=(
                    f"Fold {max_abs_idx} contributes {abs(max_abs_pnl) / abs(pooled_pnl) * 100:.0f}% "
                    "of total |pnl| — concentration risk."
                ),
                context={"fold_index": max_abs_idx, "fold_pnl": max_abs_pnl,
                         "pooled_pnl": pooled_pnl},
            ))

    return out
