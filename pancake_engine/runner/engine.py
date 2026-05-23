"""Event-time ledger runner — the core of Engine 0.3.

Pure function ``(spec, dataset, config) -> BacktestResult``. No clock reads,
no I/O, no network. Same inputs → same ``result_hash``.

Engine 0.3 is correctness-first, not TS parity. Known TS divergences are
documented in pancake-production/docs/research/pancake-engine-0.3-ts-divergences.md.
"""

from __future__ import annotations

import math
from itertools import groupby
from typing import Any, Optional

from ..__version__ import ENGINE, ENGINE_MODE, ENGINE_VERSION
from ..compile import CompiledSpec, compile_spec
from ..config import BacktestConfig
from ..hash import sha256_canonical
from ..metrics import (
    build_drawdown_curve,
    build_monthly_returns,
    compute_pm,
    compute_standard,
    daily_returns_carry_forward,
    emit_credibility_warnings,
)
from ..result import (
    BacktestResult,
    DrawdownPoint,
    EquityPoint,
    Metrics,
    MonthlyReturn,
    compute_result_hash,
)
from ..types import EvidenceDataset, EvidenceSpec
from ..validate import ValidationVerdict, validate_dataset, validate_spec
from ..warnings import Severity, Warning, WarningCode
from .events import Event, EventKind
from .ledger import Ledger
from .observation import ObservationTimeError, resolve_observation_time
from .position import Position
from .sizing import compute_sizing

__all__ = ["run_backtest"]

SECONDS_PER_DAY = 86_400
BPS_DIVISOR = 10_000


def run_backtest(
    spec: EvidenceSpec,
    dataset: EvidenceDataset,
    config: Optional[BacktestConfig] = None,
) -> BacktestResult:
    """Run an EvidenceSpec against an EvidenceDataset.

    Pure function. Determinism is a function of inputs alone.
    """
    config = config or BacktestConfig()

    # 1. Validation (errors block)
    verdict = validate_spec(spec)
    dataset_verdict, role_lookup = validate_dataset(dataset, spec)
    verdict.merge(dataset_verdict)

    if not verdict.ok:
        return _empty_result(spec, dataset, config, verdict, blocked=True)

    # 2. Compile spec (condition AST → callables)
    compiled = compile_spec(spec)

    # 3. Resolve observation_time
    warnings: list[Warning] = []
    try:
        obs_time, derived = resolve_observation_time(
            dataset,
            config,
            resolved_outcome_col=role_lookup["resolved_outcome_numeric"],
            resolution_time_col=role_lookup["resolution_time"],
        )
    except ObservationTimeError as e:
        verdict.add_error("E_OBSERVATION_TIME_REQUIRED", str(e))
        return _empty_result(spec, dataset, config, verdict, blocked=True)

    if derived:
        warnings.append(Warning(
            code=WarningCode.OBSERVATION_TIME_DERIVED,
            severity=Severity.INFO,
            message=f"observation_time auto-derived from dataset max(resolution_time) = {obs_time}.",
            context={"observation_time": obs_time},
        ))

    # 4. Build event list with row-skip warnings
    events: list[Event] = []
    rows = dataset.rows_inline or []
    market_col = role_lookup["market_link"]
    dec_col = role_lookup["decision_time"]
    res_col = role_lookup["resolution_time"]
    outcome_col = role_lookup["resolved_outcome_numeric"]
    future_rows_count = 0
    unresolved_rows_count = 0

    for source_row_index, row in enumerate(rows):
        res_t = int(row[res_col])
        outcome = row.get(outcome_col)
        if res_t > obs_time:
            future_rows_count += 1
            warnings.append(Warning(
                code=WarningCode.FUTURE_ROW_SKIPPED,
                severity=Severity.INFO,
                message=f"row {source_row_index}: resolution_time {res_t} > observation_time {obs_time}; "
                        f"row skipped (no DECISION, no RESOLUTION).",
                context={"row_index": source_row_index, "resolution_time": res_t,
                         "observation_time": obs_time},
            ))
            continue
        if outcome is None:
            unresolved_rows_count += 1
            warnings.append(Warning(
                code=WarningCode.UNRESOLVED_ROW_SKIPPED,
                severity=Severity.WARN,
                message=f"row {source_row_index}: resolved_outcome_numeric is null; row skipped.",
                context={"row_index": source_row_index},
            ))
            continue
        events.append(Event(
            time=int(row[dec_col]),
            kind=EventKind.DECISION,
            market_link=str(row[market_col]),
            source_row_index=source_row_index,
            row=row,
        ))
        events.append(Event(
            time=res_t,
            kind=EventKind.RESOLUTION,
            market_link=str(row[market_col]),
            source_row_index=source_row_index,
            row=row,
        ))

    events.sort(key=Event.sort_key)

    # 5. Ledger walk grouped by unique timestamp
    ledger = Ledger(starting_capital=compiled.starting_capital)
    equity_curve: list[EquityPoint] = []

    for t, group_iter in groupby(events, key=lambda e: e.time):
        for event in group_iter:
            if event.kind == EventKind.DECISION:
                _process_decision(event, ledger, compiled, warnings)
            else:  # RESOLUTION
                _process_resolution(event, ledger, compiled, warnings)
        equity_curve.append(EquityPoint(t=t, equity=ledger.equity()))

    # Empty equity_curve → anchor one point at observation_time with starting_capital
    if not equity_curve:
        equity_curve = [EquityPoint(t=obs_time, equity=float(compiled.starting_capital))]

    # 6. Series + metrics
    daily_rets = daily_returns_carry_forward(equity_curve)
    drawdown_curve_list, _ = build_drawdown_curve(equity_curve)
    if len(equity_curve) == 1:
        drawdown_curve_list = [DrawdownPoint(t=equity_curve[0].t, drawdown=0.0)]
    monthly_returns_list = build_monthly_returns(equity_curve) if len(equity_curve) >= 2 else []

    period_seconds = max(equity_curve[-1].t - equity_curve[0].t, 1)
    metrics_standard, ruined, cagr_overflowed = compute_standard(
        trades=ledger.trades,
        equity_curve=equity_curve,
        daily_rets=daily_rets,
        starting_capital=float(compiled.starting_capital),
        period_seconds=period_seconds,
    )
    metrics_pm = compute_pm(
        trades=ledger.trades,
        sharpe_equity_curve=metrics_standard.sharpe,
    )
    metrics = Metrics(standard=metrics_standard, pm=metrics_pm)

    # 7. Credibility warnings
    warnings.extend(emit_credibility_warnings(
        standard=metrics_standard,
        pm=metrics_pm,
        trades=ledger.trades,
        equity_curve=equity_curve,
        mark_policy=config.mark_policy,
        ruined=ruined,
        span_seconds=period_seconds,
        cagr_overflowed=cagr_overflowed,
    ))

    # 8. Hashes
    schema_sha256 = sha256_canonical(_dataset_schema_dict(dataset))
    rows_sha256 = sha256_canonical(rows)
    config_hash = sha256_canonical({**config.canonical_dict(), "observation_time": obs_time})

    result_hash = compute_result_hash(
        engine=ENGINE,
        engine_version=ENGINE_VERSION,
        engine_mode=ENGINE_MODE,
        compiled_spec_hash=compiled.compiled_spec_hash,
        schema_sha256=schema_sha256,
        rows_sha256=rows_sha256,
        config_hash=config_hash,
        metrics=metrics,
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve_list,
        monthly_returns=monthly_returns_list,
        trades=ledger.trades,
        warnings=warnings,
        future_rows_count=future_rows_count,
    )

    return BacktestResult(
        engine=ENGINE,
        engine_version=ENGINE_VERSION,
        engine_mode=ENGINE_MODE,
        compiled_spec_hash=compiled.compiled_spec_hash,
        schema_sha256=schema_sha256,
        rows_sha256=rows_sha256,
        config_hash=config_hash,
        result_hash=result_hash,
        metrics=metrics,
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve_list,
        monthly_returns=monthly_returns_list,
        trades=list(ledger.trades),
        warnings=warnings,
        validation=verdict,
        meta={
            "observation_time": obs_time,
            "observation_time_derived": derived,
            "row_count": len(rows),
            "future_rows_count": future_rows_count,
            "unresolved_rows_count": unresolved_rows_count,
            "duration_ms": 0,  # not measured; placeholder for ABI stability
        },
    )


# -----------------------------------------------------------------------------
# DECISION / RESOLUTION processors
# -----------------------------------------------------------------------------


def _process_decision(
    event: Event,
    ledger: Ledger,
    compiled: CompiledSpec,
    warnings: list[Warning],
) -> None:
    row = event.row
    if not compiled.entry_condition(row):
        return

    # Entry price range guard
    entry_price = row[_entry_price_col(compiled, event)]
    if not isinstance(entry_price, (int, float)) or isinstance(entry_price, bool):
        warnings.append(Warning(
            code=WarningCode.ENTRY_PRICE_OUT_OF_RANGE,
            severity=Severity.WARN,
            message=f"row {event.source_row_index}: entry_price is not a finite number; row skipped.",
            context={"row_index": event.source_row_index, "value": entry_price},
        ))
        return
    if not (0 < entry_price < 1):
        warnings.append(Warning(
            code=WarningCode.ENTRY_PRICE_OUT_OF_RANGE,
            severity=Severity.WARN,
            message=f"row {event.source_row_index}: entry_price={entry_price} outside (0, 1); row skipped.",
            context={"row_index": event.source_row_index, "value": entry_price},
        ))
        return

    # Sizing: fixed_fraction × available_cash
    sizing = compute_sizing(ledger.cash, compiled.sizing_value)
    if sizing.clipped:
        warnings.append(Warning(
            code=WarningCode.SIZING_CLIPPED,
            severity=Severity.WARN,
            message=f"row {event.source_row_index}: sizing clipped from {sizing.requested:.6f} "
                    f"to {sizing.notional:.6f} by available_cash.",
            context={
                "row_index": event.source_row_index,
                "requested": sizing.requested,
                "actual": sizing.notional,
                "basis_value": sizing.basis_value,
            },
        ))
    if sizing.notional <= 0:
        warnings.append(Warning(
            code=WarningCode.SIZING_ZERO,
            severity=Severity.INFO,
            message=f"row {event.source_row_index}: notional ≤ 0; row skipped.",
            context={"row_index": event.source_row_index, "basis_value": sizing.basis_value},
        ))
        return

    # Slippage: multiplicative bps
    fill_price = entry_price * (1 + compiled.slippage_bps / BPS_DIVISOR)
    if not (0 < fill_price < 1):
        warnings.append(Warning(
            code=WarningCode.FILL_PRICE_OUT_OF_RANGE,
            severity=Severity.WARN,
            message=f"row {event.source_row_index}: post-slip fill_price={fill_price:.6f} outside (0, 1); "
                    f"row skipped.",
            context={
                "row_index": event.source_row_index,
                "entry_price": entry_price,
                "fill_price": fill_price,
                "slippage_bps": compiled.slippage_bps,
            },
        ))
        return

    # Fees + shares
    fee = sizing.notional * (compiled.fee_bps / BPS_DIVISOR)
    investable = sizing.notional - fee
    shares = investable / fill_price

    position = Position(
        id=event.source_row_index,
        market_link=event.market_link,
        side=compiled.side,
        decision_time=event.time,
        resolution_time=int(row[_resolution_time_col(compiled, event)]),
        entry_price=fill_price,
        entry_price_quote=float(entry_price),
        shares=shares,
        cost=sizing.notional,
        fee=fee,
        row=row,
    )
    ledger.open(position)


def _process_resolution(
    event: Event,
    ledger: Ledger,
    compiled: CompiledSpec,
    warnings: list[Warning],
) -> None:
    position_id = event.source_row_index
    if position_id not in ledger.open_positions:
        # No position opened (entry condition was false or skipped). Nothing to do.
        return

    position = ledger.open_positions[position_id]
    yes_payoff = compiled.yes_payoff_condition(position.row)
    if compiled.side == "YES":
        strategy_wins = yes_payoff
    else:  # "NO"
        strategy_wins = not yes_payoff
    settle_value = 1.0 if strategy_wins else 0.0

    days_held = max(0, round((position.resolution_time - position.decision_time) / SECONDS_PER_DAY))
    ledger.close(position_id, settle_value=settle_value, days_held=days_held)


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def _entry_price_col(compiled: CompiledSpec, event: Event) -> str:
    return _role_col(compiled, "entry_price")


def _resolution_time_col(compiled: CompiledSpec, event: Event) -> str:
    return _role_col(compiled, "resolution_time")


def _role_col(compiled: CompiledSpec, role: str) -> str:
    for req in compiled.raw.schema_requirements.required_columns:
        if req.semantic_role == role:
            return req.name
    raise ValueError(f"spec missing required role {role!r} (validation should have caught)")


def _dataset_schema_dict(dataset: EvidenceDataset) -> dict[str, Any]:
    """Round-trip the dataset schema through pydantic to get a canonical-shape dict."""
    return dataset.dataset_schema.model_dump(exclude_none=True, mode="python")


def _empty_result(
    spec: EvidenceSpec,
    dataset: EvidenceDataset,
    config: BacktestConfig,
    verdict: ValidationVerdict,
    *,
    blocked: bool,
) -> BacktestResult:
    """Build a result for the validation-failed path. No metrics, no equity curve."""
    from ..result import MetricsPM, MetricsStandard

    # Attempt schema / rows hashes even when blocked (best-effort, may fail on
    # truly malformed input — wrap defensively).
    try:
        schema_sha256 = sha256_canonical(_dataset_schema_dict(dataset))
    except Exception:
        schema_sha256 = ""
    try:
        rows_sha256 = sha256_canonical(dataset.rows_inline or [])
    except Exception:
        rows_sha256 = ""
    config_hash = sha256_canonical({**config.canonical_dict(), "observation_time": config.observation_time})

    metrics = Metrics(
        standard=MetricsStandard(
            total_return=0.0, cagr=0.0, sharpe=None, sortino=None,
            max_drawdown=0.0, win_rate=None, num_trades=0,
            starting_capital=float(spec.starting_capital) if not blocked else 0.0,
            ending_capital=float(spec.starting_capital) if not blocked else 0.0,
        ),
        pm=MetricsPM(
            win_rate_ci95_low=None, win_rate_ci95_high=None,
            mean_return_pct=None, std_return_pct=None,
            sharpe_trade_level=None, sharpe_equity_curve=None,
            brier_strategy=None, brier_crowd=None, brier_skill_score=None,
            mean_edge=None,
        ),
    )

    return BacktestResult(
        engine=ENGINE,
        engine_version=ENGINE_VERSION,
        engine_mode=ENGINE_MODE,
        compiled_spec_hash="",
        schema_sha256=schema_sha256,
        rows_sha256=rows_sha256,
        config_hash=config_hash,
        result_hash="",
        metrics=metrics,
        equity_curve=[],
        drawdown_curve=[],
        monthly_returns=[],
        trades=[],
        warnings=[],
        validation=verdict,
        meta={
            "observation_time": config.observation_time,
            "observation_time_derived": False,
            "row_count": len(dataset.rows_inline or []),
            "future_rows_count": 0,
            "unresolved_rows_count": 0,
            "duration_ms": 0,
            "blocked": blocked,
        },
    )
