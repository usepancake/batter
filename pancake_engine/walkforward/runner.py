"""Frozen-spec walk-forward runner.

Pure function ``(spec, dataset, wf_config, backtest_config) -> WalkforwardResult``.
"""

from __future__ import annotations

from typing import Any, Optional

from ..__version__ import ENGINE, ENGINE_MODE, ENGINE_VERSION
from ..compile import compile_spec
from ..config import BacktestConfig, WalkforwardConfig
from ..hash import sha256_canonical
from ..result import (
    BacktestResult,
    DrawdownPoint,
    EquityPoint,
    Metrics,
    MetricsPM,
    MetricsStandard,
    MonthlyReturn,
    compute_result_hash,
)
from ..runner import run_backtest
from ..runner.observation import resolve_observation_time
from ..types import EvidenceDataset, EvidenceSpec
from ..validate import ValidationVerdict, validate_dataset, validate_spec
from ..warnings import Severity, Warning, WarningCode
from .aggregate import compute_aggregate, emit_aggregate_warnings
from .result import (
    AggregateMetrics,
    Fold,
    FoldDefinition,
    FoldMeanMetrics,
    FoldStdMetrics,
    PooledMetrics,
    WALKFORWARD_VERSION,
    WalkforwardResult,
    compute_aggregate_hash,
)
from .schedule import build_fold_schedule

__all__ = ["run_walkforward"]


def run_walkforward(
    spec: EvidenceSpec,
    dataset: EvidenceDataset,
    wf_config: WalkforwardConfig,
    backtest_config: Optional[BacktestConfig] = None,
) -> WalkforwardResult:
    """Run frozen-spec walk-forward evaluation."""
    backtest_config = backtest_config or BacktestConfig()

    # 1. Validate spec + dataset
    verdict = validate_spec(spec)
    dataset_verdict, role_lookup = validate_dataset(dataset, spec)
    verdict.merge(dataset_verdict)

    # 2. Validate WF config + handle truncate raise
    _validate_wf_config(wf_config, verdict)

    if not verdict.ok:
        return _empty_wf_result(spec, dataset, wf_config, backtest_config, verdict)

    if wf_config.resolution_policy == "truncate_at_window_end":
        raise NotImplementedError(
            "E_OVERHANG_TRUNCATION_UNSUPPORTED: truncate_at_window_end requires "
            "mark_at_last_observed_price (PR-3+). Use 'allow_overhang' or 'skip_overhang'."
        )

    # 3. Build schedule
    schedule = build_fold_schedule(dataset, wf_config, role_lookup)
    if len(schedule) < 2:
        verdict.add_error(
            "E_WALKFORWARD_INSUFFICIENT_FOLDS",
            f"schedule produced {len(schedule)} folds (need ≥ 2); check dataset span vs step/test_horizon",
            fold_count=len(schedule),
        )
        return _empty_wf_result(spec, dataset, wf_config, backtest_config, verdict)

    aggregate_warnings: list[Warning] = []
    # OVERRIDE_MIN_FOLD_COUNT — fires whenever user explicitly drops min_fold_count
    # below the default of 3 (i.e., to 2). Below 2 was already rejected above.
    if wf_config.min_fold_count == 2:
        aggregate_warnings.append(Warning(
            code=WarningCode.OVERRIDE_MIN_FOLD_COUNT,
            severity=Severity.INFO,
            message=(
                f"min_fold_count overridden to 2 (default 3). Walk-forward "
                f"dispersion checks are weaker with fewer folds."
            ),
            context={"min_fold_count": wf_config.min_fold_count},
        ))

    if len(schedule) < wf_config.min_fold_count:
        verdict.add_error(
            "E_WALKFORWARD_INSUFFICIENT_FOLDS",
            f"schedule produced {len(schedule)} folds (need ≥ {wf_config.min_fold_count})",
            fold_count=len(schedule),
            min_fold_count=wf_config.min_fold_count,
        )
        return _empty_wf_result(spec, dataset, wf_config, backtest_config, verdict)

    # 4. Provenance lookahead-check
    provenance = _get_provenance(dataset)
    if not provenance.get("feature_construction_verified_no_lookahead", False):
        aggregate_warnings.append(Warning(
            code=WarningCode.FEATURE_LOOKAHEAD_UNCHECKED,
            severity=Severity.INFO,
            message=(
                "dataset.provenance.feature_construction_verified_no_lookahead is not true. "
                "Engine cannot verify that feature columns were constructed without lookahead; "
                "this is a dataset-producer responsibility."
            ),
            context={},
        ))

    # 5. Per-fold backtests
    decision_time_col = role_lookup["decision_time"]
    resolution_time_col = role_lookup["resolution_time"]
    rows = dataset.rows_inline or []
    folds: list[Fold] = []
    for fold_def in schedule:
        test_start, test_end = fold_def.test_window
        fold_rows = [r for r in rows
                     if test_start <= int(r[decision_time_col]) < test_end]

        # Overhang policy
        if wf_config.resolution_policy == "skip_overhang":
            kept = []
            for r in fold_rows:
                if int(r[resolution_time_col]) > test_end:
                    aggregate_warnings.append(Warning(
                        code=WarningCode.OVERHANG_SKIPPED,
                        severity=Severity.INFO,
                        message=(
                            f"fold {fold_def.index}: row with resolution_time "
                            f"{r[resolution_time_col]} > test_window end {test_end}; skipped."
                        ),
                        context={"fold_index": fold_def.index},
                    ))
                else:
                    kept.append(r)
            fold_rows = kept

        if not fold_rows:
            fold_result = _empty_fold_backtest(spec, dataset, backtest_config,
                                                fold_def, role_lookup)
        else:
            fold_dataset = _slice_dataset(dataset, fold_rows)
            # Use fold-specific observation_time: either explicit (from
            # backtest_config) or derived from the fold's rows.
            fold_bt_config = backtest_config
            fold_result = run_backtest(spec, fold_dataset, fold_bt_config)
            # Per-fold low/empty-trade warnings
            if fold_result.metrics.standard.num_trades == 0:
                fold_result.warnings.append(Warning(
                    code=WarningCode.EMPTY_FOLD,
                    severity=Severity.WARN,
                    message=f"fold {fold_def.index}: zero trades over test window.",
                    context={"fold_index": fold_def.index},
                ))
            elif fold_result.metrics.standard.num_trades < 10:
                fold_result.warnings.append(Warning(
                    code=WarningCode.LOW_TRADES_IN_FOLD,
                    severity=Severity.WARN,
                    message=(
                        f"fold {fold_def.index}: {fold_result.metrics.standard.num_trades} "
                        f"trades (< 10)."
                    ),
                    context={"fold_index": fold_def.index,
                             "num_trades": fold_result.metrics.standard.num_trades},
                ))

        folds.append(Fold(definition=fold_def, result=fold_result))

    # 6. Aggregate metrics + warnings
    aggregate = compute_aggregate(folds)
    aggregate_warnings.extend(emit_aggregate_warnings(folds, aggregate))

    # 7. Hashes
    compiled = compile_spec(spec)
    schema_sha256 = sha256_canonical(dataset.dataset_schema.model_dump(exclude_none=True, mode="python"))
    rows_sha256 = sha256_canonical(rows)
    wf_canonical = wf_config.canonical_dict()
    bt_canonical = backtest_config.canonical_dict()
    config_hash = sha256_canonical({"backtest": bt_canonical, "walkforward": wf_canonical})

    aggregate_result_hash = compute_aggregate_hash(
        engine=ENGINE,
        engine_version=ENGINE_VERSION,
        engine_mode=ENGINE_MODE,
        compiled_spec_hash=compiled.compiled_spec_hash,
        schema_sha256=schema_sha256,
        rows_sha256=rows_sha256,
        config_hash=config_hash,
        folds=folds,
        aggregate=aggregate,
        warnings=aggregate_warnings,
    )

    return WalkforwardResult(
        engine=ENGINE,
        engine_version=ENGINE_VERSION,
        engine_mode=ENGINE_MODE,
        walkforward_version=WALKFORWARD_VERSION,
        result_kind="walkforward",
        compiled_spec_hash=compiled.compiled_spec_hash,
        schema_sha256=schema_sha256,
        rows_sha256=rows_sha256,
        config_hash=config_hash,
        aggregate_result_hash=aggregate_result_hash,
        folds=folds,
        aggregate=aggregate,
        warnings=aggregate_warnings,
        validation=verdict,
        meta={
            "fold_count": len(folds),
            "schedule_span": [folds[0].definition.test_window[0],
                              folds[-1].definition.test_window[1]] if folds else None,
        },
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _validate_wf_config(wf_config: WalkforwardConfig, verdict: ValidationVerdict) -> None:
    if wf_config.min_fold_count < 2:
        verdict.add_error(
            "E_WALKFORWARD_CONFIG_INVALID",
            f"min_fold_count must be >= 2 (got {wf_config.min_fold_count})",
            field="min_fold_count",
        )
    if wf_config.min_test_rows < 0:
        verdict.add_error(
            "E_WALKFORWARD_CONFIG_INVALID",
            f"min_test_rows must be >= 0 (got {wf_config.min_test_rows})",
            field="min_test_rows",
        )
    if wf_config.window_type not in ("expanding", "rolling", "anchored"):
        verdict.add_error(
            "E_WALKFORWARD_CONFIG_INVALID",
            f"window_type must be expanding/rolling/anchored (got {wf_config.window_type!r})",
            field="window_type",
        )
    if wf_config.resolution_policy not in (
        "allow_overhang", "skip_overhang", "truncate_at_window_end"
    ):
        verdict.add_error(
            "E_WALKFORWARD_CONFIG_INVALID",
            f"resolution_policy invalid (got {wf_config.resolution_policy!r})",
            field="resolution_policy",
        )


def _get_provenance(dataset: EvidenceDataset) -> dict[str, Any]:
    """Get the provenance dict from the dataset, defaulting to ``{}``."""
    extras = dataset.model_extra or {}
    prov = extras.get("provenance")
    if isinstance(prov, dict):
        return prov
    return {}


def _slice_dataset(dataset: EvidenceDataset, rows: list[dict[str, Any]]) -> EvidenceDataset:
    """Build a copy of dataset with only the given rows (rows_sha256 recomputed)."""
    return dataset.model_copy(update={
        "rows_inline": rows,
        "row_count": len(rows),
        "rows_sha256": sha256_canonical(rows),
    })


def _empty_fold_backtest(
    spec: EvidenceSpec,
    dataset: EvidenceDataset,
    backtest_config: BacktestConfig,
    fold_def: FoldDefinition,
    role_lookup: dict[str, str],
) -> BacktestResult:
    """Construct an empty-but-valid BacktestResult for a fold with zero rows.

    Avoids calling run_backtest (which would raise E_EVIDENCE_ROWS_MISSING).
    """
    from ..compile import compile_spec

    compiled = compile_spec(spec)
    schema_sha256 = sha256_canonical(dataset.dataset_schema.model_dump(exclude_none=True, mode="python"))
    rows_sha256 = sha256_canonical([])
    obs_time = (backtest_config.observation_time
                if backtest_config.observation_time is not None
                else fold_def.test_window[1])
    config_hash = sha256_canonical({**backtest_config.canonical_dict(), "observation_time": obs_time})

    metrics = Metrics(
        standard=MetricsStandard(
            total_return=0.0, cagr=0.0, sharpe=None, sortino=None,
            max_drawdown=0.0, win_rate=None, num_trades=0,
            starting_capital=float(compiled.starting_capital),
            ending_capital=float(compiled.starting_capital),
        ),
        pm=MetricsPM(
            win_rate_ci95_low=None, win_rate_ci95_high=None,
            mean_return_pct=None, std_return_pct=None,
            sharpe_trade_level=None, sharpe_equity_curve=None,
            brier_strategy=None, brier_crowd=None, brier_skill_score=None,
            mean_edge=None,
        ),
    )
    equity_curve = [EquityPoint(t=obs_time, equity=float(compiled.starting_capital))]
    drawdown_curve = [DrawdownPoint(t=obs_time, drawdown=0.0)]
    warnings: list[Warning] = [
        Warning(code=WarningCode.NO_TRADES_GENERATED, severity=Severity.WARN,
                message=f"fold {fold_def.index}: no rows in test window.",
                context={"fold_index": fold_def.index}),
        Warning(code=WarningCode.NO_TRADES_NO_CI, severity=Severity.INFO,
                message="Wilson CI bounds are null because num_trades = 0.",
                context={}),
        Warning(code=WarningCode.EMPTY_FOLD, severity=Severity.WARN,
                message=f"fold {fold_def.index}: empty fold.",
                context={"fold_index": fold_def.index}),
    ]

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
        drawdown_curve=drawdown_curve,
        monthly_returns=[],
        trades=[],
        warnings=warnings,
        future_rows_count=0,
    )

    return BacktestResult(
        engine=ENGINE, engine_version=ENGINE_VERSION, engine_mode=ENGINE_MODE,
        compiled_spec_hash=compiled.compiled_spec_hash,
        schema_sha256=schema_sha256, rows_sha256=rows_sha256,
        config_hash=config_hash, result_hash=result_hash,
        metrics=metrics, equity_curve=equity_curve,
        drawdown_curve=drawdown_curve, monthly_returns=[],
        trades=[], warnings=warnings, validation=ValidationVerdict(),
        meta={
            "observation_time": obs_time,
            "observation_time_derived": False,
            "row_count": 0,
            "future_rows_count": 0,
            "unresolved_rows_count": 0,
            "duration_ms": 0,
            "fold_empty": True,
        },
    )


def _empty_wf_result(
    spec: EvidenceSpec,
    dataset: EvidenceDataset,
    wf_config: WalkforwardConfig,
    backtest_config: BacktestConfig,
    verdict: ValidationVerdict,
) -> WalkforwardResult:
    """Build a WF result for the validation-failed path."""
    empty_agg = AggregateMetrics(
        fold_count=0, non_empty_fold_count=0,
        pooled=PooledMetrics(num_trades=0, win_rate=None, mean_return_pct=None,
                             std_return_pct=None, sharpe_trade_level=None, brier_crowd=None),
        fold_mean=FoldMeanMetrics(total_return=None, sharpe=None, sortino=None,
                                  max_drawdown=None, win_rate=None, num_trades=0.0),
        fold_std=FoldStdMetrics(total_return=None, sharpe=None, sortino=None,
                                max_drawdown=None, win_rate=None, num_trades=None),
        fold_sharpe_dispersion=None,
        fold_win_rate_dispersion=None,
    )
    return WalkforwardResult(
        engine=ENGINE, engine_version=ENGINE_VERSION, engine_mode=ENGINE_MODE,
        walkforward_version=WALKFORWARD_VERSION, result_kind="walkforward",
        compiled_spec_hash="", schema_sha256="", rows_sha256="",
        config_hash="", aggregate_result_hash="",
        folds=[], aggregate=empty_agg, warnings=[], validation=verdict,
        meta={"blocked": True},
    )
