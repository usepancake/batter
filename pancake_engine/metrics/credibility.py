"""Credibility warnings catalog (architecture §Credibility warnings).

These fire automatically and must be rendered alongside metrics on any UI
consuming the result. Without them, a tiny-sample-size Sharpe = 8 looks
respectable; with them it carries ``LOW_SAMPLE_SIZE`` + ``IMPLAUSIBLY_HIGH_SHARPE``
next to the number.

Engine 0.3 is correctness-first, not TS parity. TS emits none of these.
"""

from __future__ import annotations

from typing import Optional

from ..result import EquityPoint, MetricsPM, MetricsStandard
from ..runner.trade import Trade
from ..warnings import Severity, Warning, WarningCode

__all__ = ["emit_credibility_warnings"]


def emit_credibility_warnings(
    *,
    standard: MetricsStandard,
    pm: MetricsPM,
    trades: list[Trade],
    equity_curve: list[EquityPoint],
    mark_policy: str,
    ruined: bool,
    span_seconds: int,
    cagr_overflowed: bool = False,
) -> list[Warning]:
    """Return the list of credibility warnings for the given result."""
    out: list[Warning] = []
    n = standard.num_trades

    if n == 0:
        out.append(Warning(
            code=WarningCode.NO_TRADES_GENERATED,
            severity=Severity.WARN,
            message="Strategy never fired — zero trades over the dataset.",
            context={"row_count": len(equity_curve)},
        ))
        out.append(Warning(
            code=WarningCode.NO_TRADES_NO_CI,
            severity=Severity.INFO,
            message="Wilson CI bounds are null because num_trades = 0.",
            context={},
        ))
    else:
        if n < 10:
            out.append(Warning(
                code=WarningCode.MICRO_SAMPLE_SIZE,
                severity=Severity.ERROR,
                message=f"Only {n} trades — results are not statistically meaningful.",
                context={"num_trades": n, "threshold": 10},
            ))
        elif n < 30:
            out.append(Warning(
                code=WarningCode.LOW_SAMPLE_SIZE,
                severity=Severity.WARN,
                message=f"Only {n} trades — results have wide confidence bands.",
                context={"num_trades": n, "threshold": 30},
            ))

    sharpe = standard.sharpe
    if sharpe is not None and sharpe > 3 and n < 100:
        out.append(Warning(
            code=WarningCode.IMPLAUSIBLY_HIGH_SHARPE,
            severity=Severity.WARN,
            message=f"Sharpe ratio {sharpe:.2f} with only {n} trades is implausibly high.",
            context={"sharpe": sharpe, "num_trades": n},
        ))

    SECONDS_PER_YEAR_APPROX = 365 * 86_400
    if standard.total_return > 5 and 0 < span_seconds < SECONDS_PER_YEAR_APPROX:
        out.append(Warning(
            code=WarningCode.IMPLAUSIBLY_HIGH_RETURN,
            severity=Severity.WARN,
            message=f"Total return {standard.total_return * 100:.0f}% over <1y is implausibly high.",
            context={"total_return": standard.total_return, "span_seconds": span_seconds},
        ))

    if n > 0 and standard.win_rate in (0.0, 1.0) and n < 100:
        out.append(Warning(
            code=WarningCode.DEGENERATE_HIT_RATE,
            severity=Severity.WARN,
            message=f"Win rate is {standard.win_rate} with only {n} trades — too few to interpret.",
            context={"win_rate": standard.win_rate, "num_trades": n},
        ))

    markets = {t.market_slug for t in trades}
    if n > 0 and len(markets) == 1:
        out.append(Warning(
            code=WarningCode.SINGLE_MARKET_RESULT,
            severity=Severity.WARN,
            message=f"All {n} trades on a single market_link — no diversification.",
            context={"market_link": next(iter(markets)), "num_trades": n},
        ))

    if mark_policy == "mark_at_cost" and standard.max_drawdown < 0.01 and n > 10:
        out.append(Warning(
            code=WarningCode.MARK_AT_COST_DRAWDOWN_MUTED,
            severity=Severity.INFO,
            message=f"max_drawdown={standard.max_drawdown:.4f} under mark_at_cost is structurally "
                    f"muted — DD reflects realized losses + entry fees only.",
            context={"max_drawdown": standard.max_drawdown, "mark_policy": mark_policy},
        ))

    if ruined:
        out.append(Warning(
            code=WarningCode.RUINED,
            severity=Severity.WARN,
            message="Ending equity ≤ 0; cagr capped at -1.0.",
            context={"ending_capital": standard.ending_capital},
        ))

    if cagr_overflowed:
        out.append(Warning(
            code=WarningCode.CAGR_EXTRAPOLATION_OVERFLOW,
            severity=Severity.WARN,
            message=(
                "CAGR extrapolation overflowed float64; cagr set to null. "
                "Use total_return for the realized return; CAGR annualization is "
                "unreliable over very short windows with extreme multipliers."
            ),
            context={"starting_capital": standard.starting_capital,
                     "ending_capital": standard.ending_capital,
                     "span_seconds": span_seconds},
        ))

    if n > 0 and span_seconds > 0:
        active_seconds = sum(t.days_held * 86_400 for t in trades)
        duty_cycle = active_seconds / span_seconds if span_seconds > 0 else 0.0
        if duty_cycle < 0.1:
            out.append(Warning(
                code=WarningCode.CAGR_LOW_DUTY_CYCLE,
                severity=Severity.INFO,
                message=f"Strategy active only {duty_cycle * 100:.1f}% of the dataset span — "
                        f"cagr extrapolation is fragile.",
                context={"duty_cycle": duty_cycle, "span_seconds": span_seconds},
            ))

    if n > 0 and span_seconds > 0:
        first_t = trades[0].entry_t
        last_t = trades[-1].entry_t
        trade_span = last_t - first_t
        if trade_span > 0 and trade_span < 0.2 * span_seconds:
            out.append(Warning(
                code=WarningCode.TIME_CLUSTERED_TRADES,
                severity=Severity.WARN,
                message="Trade timing clustered into <20% of the dataset span.",
                context={"trade_span_seconds": trade_span, "dataset_span_seconds": span_seconds},
            ))

    # BRIER_NOT_APPLICABLE — fires whenever strategy emits no independent probability
    # (true for every rule-based EvidenceSpec in PR-1). Emit only when there are trades
    # to score against; otherwise it's redundant with NO_TRADES_GENERATED.
    if n > 0:
        out.append(Warning(
            code=WarningCode.BRIER_NOT_APPLICABLE,
            severity=Severity.INFO,
            message="brier_strategy is null because rule-based specs emit no independent probability. "
                    "Use brier_crowd as a market-baseline comparison.",
            context={},
        ))

    return out
