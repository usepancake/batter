"""Event-time ledger runner — the core of Engine 0.3.

Pure function ``(spec, dataset, config) -> BacktestResult``. No clock reads,
no I/O, no network. Same inputs → same ``result_hash``.

Engine 0.3 is correctness-first, not TS parity. Known TS divergences are
documented in docs/math-audit-0.4.md.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from itertools import groupby
from typing import Any, Optional

from ..__version__ import ENGINE, ENGINE_MODE, ENGINE_VERSION
from ..compile import CompiledSpec, compile_spec
from ..fills.registry import FillBlocked
from ..compile.condition import Condition, extract_referenced_columns
from ..metrics.cost_sensitivity import cost_sensitivity
from ..metrics.psr import deflated_sharpe_ratio
from ..metrics.regime import regime_stability
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
from ..metrics.pm import calibration_bins as _calibration_bins
from ..result import (
    BacktestResult,
    DrawdownPoint,
    EquityPoint,
    Metrics,
    MonthlyReturn,
    compute_result_hash,
)
from ..trials import TrialHistory
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
    *,
    with_inference: bool = True,
    _entry_override: Optional[Condition] = None,
    trial_history: Optional[TrialHistory] = None,
    book_dataset: Optional[EvidenceDataset] = None,
) -> BacktestResult:
    """Run an EvidenceSpec against an EvidenceDataset.

    Pure function. Determinism is a function of inputs alone.

    ``with_inference=False`` skips the bootstrap CIs + permutation test (the
    expensive metrics inference), leaving those fields at their empty sentinels.
    Used by the parameter sweep (ADR-0046), where each cell only needs Sharpe;
    it is an EXECUTION argument, not part of ``config``/``config_hash``, so the
    default (True) leaves ``result_hash`` byte-identical for every receipt.

    ``_entry_override`` is ENGINE-INTERNAL (the baseline pass): it replaces the
    compiled entry condition after compilation. Never exposed through any public
    surface — an always-true condition is intentionally not expressible as a spec
    (see the 0.8 empty-``all_of`` guard), and the override does not touch
    ``compiled_spec_hash``.

    ``trial_history`` is an EXECUTION argument (like ``with_inference``): it is
    NOT part of ``config``/``config_hash`` and does NOT affect ``result_hash``.
    When supplied AND the run has defined daily returns, a Deflated Sharpe Ratio
    block is added as ``BacktestResult.deflated`` (additive, non-hashed). The
    platform (pancake 0.10.0+) supplies the caller's TRUE search history so DSR
    is computed against the real trial count rather than a synthetic sweep grid.
    The current run's own Sharpe is always appended to the trial set (the run is
    itself a trial); ``deflated.own_sharpe_included`` is always ``True``.
    """
    config = config or BacktestConfig()

    # 1. Validation (errors block)
    verdict = validate_spec(spec)
    dataset_verdict, role_lookup = validate_dataset(dataset, spec)
    verdict.merge(dataset_verdict)

    # book_dataset required when fill_model is book_replay@1.
    _using_book_replay = (
        spec.costs.fill_model is not None
        and spec.costs.fill_model.name == "book_replay"
        and spec.costs.fill_model.version == 1
    )
    if _using_book_replay and book_dataset is None:
        verdict.add_error(
            "E_EVIDENCE_BOOK_DATASET_REQUIRED",
            "fill_model book_replay@1 requires book_dataset to be supplied to run_backtest; "
            "pass book_dataset=<EvidenceDataset> with L2 snapshot rows.",
        )

    if not verdict.ok:
        return _empty_result(spec, dataset, config, verdict, blocked=True)

    # 2. Compile spec (condition AST → callables)
    compiled = compile_spec(spec)
    if _entry_override is not None:
        compiled = dataclasses.replace(compiled, entry_condition=_entry_override)

    # 2b. Extract book_slices from book_dataset (book_replay@1 only).
    _book_slices: Optional[list[dict[str, Any]]] = None
    _book_dataset_id: Optional[str] = None
    _book_rows_sha256: Optional[str] = None
    _book_schema_sha256: Optional[str] = None
    if _using_book_replay and book_dataset is not None:
        _book_slices = book_dataset.rows_inline or []
        _book_dataset_id = book_dataset.id
        _book_rows_sha256 = sha256_canonical(_book_slices)
        _book_schema_sha256 = sha256_canonical(_dataset_schema_dict(book_dataset))

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

    # AF-3: track equity overflow so we can halt the walk and emit EQUITY_OVERFLOW_BOUND.
    # float64 overflow → equity becomes inf; next step: inf - inf = nan propagates into
    # trade fields (pnl, proceeds) which then reach sha256_canonical → E_NONFINITE.
    # Fix: stop processing events the moment equity goes non-finite; clamp the last equity
    # point to sys.float_info.max so all downstream floats stay canonical.
    _FLOAT_MAX = sys.float_info.max
    _equity_overflowed = False
    _overflow_t: int = 0

    for t, group_iter in groupby(events, key=lambda e: e.time):
        if _equity_overflowed:
            # Consume the group iterator without processing (keep groupby state clean).
            list(group_iter)
            continue
        for event in group_iter:
            if event.kind == EventKind.DECISION:
                _process_decision(event, ledger, compiled, warnings, book_slices=_book_slices)
            else:  # RESOLUTION
                _process_resolution(event, ledger, compiled, warnings)
        eq = ledger.equity()
        if not math.isfinite(eq):
            _equity_overflowed = True
            _overflow_t = t
            equity_curve.append(EquityPoint(t=t, equity=_FLOAT_MAX))
        else:
            equity_curve.append(EquityPoint(t=t, equity=eq))

    if _equity_overflowed:
        warnings.append(Warning(
            code=WarningCode.EQUITY_OVERFLOW_BOUND,
            severity=Severity.WARN,
            message=(
                f"Equity exceeded float64 range at t={_overflow_t}; "
                f"halted further trade processing and clamped to sys.float_info.max ({_FLOAT_MAX:.3e}). "
                "Total return and CAGR are unreliable for this run."
            ),
            context={"overflow_t": _overflow_t},
        ))

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
    metrics_standard, ruined, cagr_overflowed, bootstrap_warnings = compute_standard(
        trades=ledger.trades,
        equity_curve=equity_curve,
        daily_rets=daily_rets,
        starting_capital=float(compiled.starting_capital),
        period_seconds=period_seconds,
        with_inference=with_inference,
    )
    warnings.extend(bootstrap_warnings)
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

    # 7b. Verification-boundary: AGENT_SUPPLIED_FEATURE_UNVERIFIED (E3b parity with TS runner)
    # Collect columns referenced in entry / yes_payoff / exit predicates that carry semantic_role=feature.
    # These are agent-supplied (derived by the user before handing data to Pancake); Pancake
    # cannot verify their provenance, look-ahead cleanliness, or derivation correctness.
    entry_when = spec.strategy.entry.get("when", {})
    yes_payoff_when = spec.strategy.yes_payoff.get("when", {})
    exit_when_for_feature = {}
    if isinstance(spec.strategy.exit, dict):
        ew = spec.strategy.exit.get("when")
        if isinstance(ew, dict):
            exit_when_for_feature = ew
    all_referenced = (
        extract_referenced_columns(entry_when if isinstance(entry_when, dict) else {})
        | extract_referenced_columns(yes_payoff_when if isinstance(yes_payoff_when, dict) else {})
        | extract_referenced_columns(exit_when_for_feature)
    )

    # 0.9: EXIT_NOT_APPLIED_BACKTEST — evidence rows are one-shot hold-to-resolution;
    # exit applies only to the paper/live lanes until the bar-series domain lands.
    # NOTE: this warning enters result_hash for specs that include exit (new specs only);
    # specs without exit.when are byte-identical to pre-0.9 receipts.
    if spec.strategy.exit is not None:
        warnings.append(Warning(
            code=WarningCode.EXIT_NOT_APPLIED_BACKTEST,
            severity=Severity.INFO,
            message=(
                "strategy.exit is set but not applied in the backtest lane: "
                "evidence rows are one-shot hold-to-resolution; "
                "exit applies to the paper/live lanes until the bar-series domain lands."
            ),
            context={},
        ))
    # Filter to columns that are declared as semantic_role=feature in schema_requirements.
    feature_role_cols = {
        req.name
        for req in spec.schema_requirements.required_columns
        if req.semantic_role == "feature"
    }
    referenced_feature_columns = sorted(all_referenced & feature_role_cols)
    if referenced_feature_columns:
        warnings.append(Warning(
            code=WarningCode.AGENT_SUPPLIED_FEATURE_UNVERIFIED,
            severity=Severity.INFO,
            message=(
                f"{len(referenced_feature_columns)} agent-supplied feature column(s) referenced in "
                "entry/yes_payoff predicates; Pancake did not verify their derivation"
            ),
            context={"feature_columns": referenced_feature_columns},
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
        book_dataset_id=_book_dataset_id,
        book_rows_sha256=_book_rows_sha256,
        book_schema_sha256=_book_schema_sha256,
    )

    # 0.8: transaction-cost sensitivity (additive, NOT hashed). Gated with the rest
    # of the inference block so the sweep's fast path stays free of it. Best-effort:
    # on a degenerate clamped-overflow run (AF-3) the rescale can overflow fsum — cost
    # analysis is meaningless there, so fall back to None rather than raise.
    cost_sens = None
    if with_inference and ledger.trades:
        try:
            cost_sens = cost_sensitivity(ledger.trades).to_dict()
        except (OverflowError, ValueError, ZeroDivisionError):
            cost_sens = None

    # 0.9: deflated Sharpe block (additive; NOT hashed — trial_history is an
    # execution argument like with_inference; same gate as cost_sensitivity).
    # Scale convention: metrics.standard.sharpe is ANNUALIZED (already ×sqrt(252)).
    # When trial_history.annualized=True the own sharpe is used as-is; when False
    # it is divided by sqrt(252) to match the per-period scale PSR uses internally.
    # deflated_sharpe_ratio always receives per-period or always annualized sharpes
    # (the sharpes_annualized flag tells it which); we pass the combined list on the
    # SAME scale as the caller supplied, appending own sharpe on that scale.
    deflated_block = None
    if with_inference and trial_history is not None and daily_rets:
        own_sharpe_ann: Optional[float] = metrics_standard.sharpe  # may be None
        if own_sharpe_ann is not None:
            # Scale the run's own (annualized) Sharpe to match the caller's scale.
            # metrics_standard.sharpe is annualized (×sqrt(252)); when the caller
            # supplied per-period Sharpes (annualized=False), divide back down.
            if trial_history.annualized:
                own_sharpe_on_scale = own_sharpe_ann
            else:
                own_sharpe_on_scale = own_sharpe_ann / math.sqrt(252)
            all_trial_sharpes = list(trial_history.trial_sharpes) + [own_sharpe_on_scale]
            dsr = deflated_sharpe_ratio(
                list(daily_rets),
                all_trial_sharpes,
                sharpes_annualized=trial_history.annualized,
            )
            deflated_block = {
                "dsr": dsr,
                "n_trials": len(all_trial_sharpes),
                "source": trial_history.source,
                "own_sharpe_included": True,
            }

    # 0.9: calibration reliability curve (additive; NOT hashed — scalar headline ECE
    # IS hashed via MetricsPM.calibration_ece; same with_inference gate as cost_sensitivity).
    calibration_bins_block = None
    if with_inference and ledger.trades:
        raw_bins = _calibration_bins(ledger.trades)
        if raw_bins is not None:
            calibration_bins_block = {"bins": raw_bins, "n_trades": len(ledger.trades)}

    # 0.10.0 Wave E: regime-stability block (additive; NOT hashed — calibration_bins
    # pattern: computed under with_inference, omitted from compute_result_hash).
    # None when <8 trades or <4 distinct equity timestamps.
    regime_block = None
    if with_inference:
        regime_block = regime_stability(equity_curve, ledger.trades)

    # 0.8 baseline (spec v0.2 subset; additive, NOT hashed — the REQUEST is hashed
    # via the spec, the output block folds into the hash at the 0.9.0 break).
    # NO-FILTER convention: same side/sizing/costs on every candidate row — the
    # baseline differs from the strategy only by the entry condition, so it
    # isolates the entry condition's selection value. Implemented as an internal
    # second pass with the entry condition replaced by always-true; the inner
    # run's warnings/hashes are internal and dropped.
    baseline_block = None
    if with_inference and _entry_override is None and spec.strategy.baseline:
        base_res = run_backtest(
            spec, dataset, config,
            with_inference=False,
            _entry_override=lambda _row: True,
            book_dataset=book_dataset,
        )
        bs = base_res.metrics.standard
        baseline_block = {
            "kind": spec.strategy.baseline.get("kind", "buy_and_hold"),
            "convention": "no_filter",
            "total_return": bs.total_return,
            "cagr": bs.cagr,
            "sharpe": bs.sharpe,
            "sortino": bs.sortino,
            "max_drawdown": bs.max_drawdown,
            "win_rate": bs.win_rate,
            "num_trades": bs.num_trades,
            "ending_capital": bs.ending_capital,
            "equity_curve": [
                {"t": p.t, "equity": p.equity} for p in base_res.equity_curve
            ],
        }

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
        cost_sensitivity=cost_sens,
        baseline=baseline_block,
        deflated=deflated_block,
        calibration_bins=calibration_bins_block,
        regime=regime_block,
        book_dataset_id=_book_dataset_id,
        book_rows_sha256=_book_rows_sha256,
        book_schema_sha256=_book_schema_sha256,
    )


# -----------------------------------------------------------------------------
# DECISION / RESOLUTION processors
# -----------------------------------------------------------------------------


def _process_decision(
    event: Event,
    ledger: Ledger,
    compiled: CompiledSpec,
    warnings: list[Warning],
    *,
    book_slices: Optional[list[dict[str, Any]]] = None,
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

    # Fill model: delegates to the compiled fill model (default: static_bps@1).
    # book_replay@1 receives market_link + decision_time + book_slices for L2 walk.
    # static_bps@1 / next_bar_open@1 ignore those extra kwargs.
    market_link_col = _role_col(compiled, "market_link")
    dec_col = _role_col(compiled, "decision_time")
    entry_fill_or_blocked = compiled.fill_model.apply_entry(
        quote=entry_price,
        notional=sizing.notional,
        slippage_bps=compiled.slippage_bps,
        fee_bps=compiled.fee_bps,
        market_link=str(row.get(market_link_col, "")),
        decision_time=int(row.get(dec_col, event.time)),
        book_slices=book_slices,
    )

    # book_replay@1 returns a FillBlocked sentinel on SLICE_MISSING or DEPTH_INSUFFICIENT.
    if isinstance(entry_fill_or_blocked, FillBlocked):
        blocked = entry_fill_or_blocked
        code = WarningCode.BOOK_SLICE_MISSING if blocked.reason == "BOOK_SLICE_MISSING" else WarningCode.BOOK_DEPTH_INSUFFICIENT
        warnings.append(Warning(
            code=code,
            severity=Severity.WARN,
            message=(
                f"row {event.source_row_index}: book_replay@1 blocked — {blocked.reason}; "
                "row skipped (no silent fallback)."
            ),
            context={"row_index": event.source_row_index, **blocked.context},
        ))
        return

    entry_fill = entry_fill_or_blocked
    fill_price = entry_fill.fill_price
    fee = entry_fill.fee
    shares = entry_fill.shares

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
