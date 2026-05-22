"""Walk-forward result dataclasses + ``aggregate_result_hash`` contract.

Engine 0.3 is correctness-first, not TS parity. TS has no walk-forward — the
aggregate hash is an Engine-0.3-native sentinel.

Aggregate hash payload includes ``walkforward_version`` and ``result_kind``
so a single-fold WF run cannot collide with a plain BacktestResult hash.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ..hash import sha256_canonical
from ..result import BacktestResult
from ..validate.verdict import ValidationVerdict
from ..warnings import Warning

__all__ = [
    "FoldDefinition",
    "Fold",
    "PooledMetrics",
    "FoldMeanMetrics",
    "FoldStdMetrics",
    "AggregateMetrics",
    "WalkforwardResult",
    "compute_aggregate_hash",
    "WALKFORWARD_VERSION",
]

WALKFORWARD_VERSION = "0.1"


@dataclass(frozen=True)
class FoldDefinition:
    """The scheduled window pair for a fold. ``train_window`` is metadata only in PR-2."""

    index: int
    train_window: tuple[int, int]
    test_window: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "train_window": list(self.train_window),
            "test_window": list(self.test_window),
        }


@dataclass(frozen=True)
class Fold:
    """A scheduled fold + its per-fold BacktestResult."""

    definition: FoldDefinition
    result: BacktestResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition": self.definition.to_dict(),
            "result": self.result.to_dict(),
        }


@dataclass(frozen=True)
class PooledMetrics:
    """Metrics over the union of trades across all folds.

    Pool-equity metrics (cagr, max_drawdown, sharpe_equity_curve) are
    deliberately absent — folds reset starting_capital each, so pooling
    those would be misleading.
    """

    num_trades: int
    win_rate: Optional[float]
    mean_return_pct: Optional[float]
    std_return_pct: Optional[float]
    sharpe_trade_level: Optional[float]
    brier_crowd: Optional[float]


@dataclass(frozen=True)
class FoldMeanMetrics:
    """Mean across per-fold scalar metrics. ``None`` when no non-null values."""

    total_return: Optional[float]
    sharpe: Optional[float]
    sortino: Optional[float]
    max_drawdown: Optional[float]
    win_rate: Optional[float]
    num_trades: float


@dataclass(frozen=True)
class FoldStdMetrics:
    """Standard deviation across per-fold scalar metrics. ``None`` if n<2."""

    total_return: Optional[float]
    sharpe: Optional[float]
    sortino: Optional[float]
    max_drawdown: Optional[float]
    win_rate: Optional[float]
    num_trades: Optional[float]


@dataclass(frozen=True)
class AggregateMetrics:
    fold_count: int
    non_empty_fold_count: int
    pooled: PooledMetrics
    fold_mean: FoldMeanMetrics
    fold_std: FoldStdMetrics
    fold_sharpe_dispersion: Optional[float]
    fold_win_rate_dispersion: Optional[float]


@dataclass
class WalkforwardResult:
    engine: str
    engine_version: str
    engine_mode: str
    walkforward_version: str
    result_kind: str  # "walkforward" — domain separator from BacktestResult

    compiled_spec_hash: str
    schema_sha256: str
    rows_sha256: str
    config_hash: str           # canonical_dict of WalkforwardConfig + BacktestConfig
    aggregate_result_hash: str  # filled after construction

    folds: list[Fold]
    aggregate: AggregateMetrics
    warnings: list[Warning]
    validation: ValidationVerdict
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "engine_version": self.engine_version,
            "engine_mode": self.engine_mode,
            "walkforward_version": self.walkforward_version,
            "result_kind": self.result_kind,
            "hashes": {
                "compiled_spec_hash": self.compiled_spec_hash,
                "schema_sha256": self.schema_sha256,
                "rows_sha256": self.rows_sha256,
                "config_hash": self.config_hash,
                "aggregate_result_hash": self.aggregate_result_hash,
            },
            "folds": [f.to_dict() for f in self.folds],
            "aggregate": {
                "fold_count": self.aggregate.fold_count,
                "non_empty_fold_count": self.aggregate.non_empty_fold_count,
                "pooled": asdict(self.aggregate.pooled),
                "fold_mean": asdict(self.aggregate.fold_mean),
                "fold_std": asdict(self.aggregate.fold_std),
                "fold_sharpe_dispersion": self.aggregate.fold_sharpe_dispersion,
                "fold_win_rate_dispersion": self.aggregate.fold_win_rate_dispersion,
            },
            "warnings": [w.to_dict() for w in self.warnings],
            "validation": self.validation.to_dict(),
            "meta": self.meta,
        }


def compute_aggregate_hash(
    *,
    engine: str,
    engine_version: str,
    engine_mode: str,
    compiled_spec_hash: str,
    schema_sha256: str,
    rows_sha256: str,
    config_hash: str,
    folds: list[Fold],
    aggregate: AggregateMetrics,
    warnings: list[Warning],
) -> str:
    """Compute the aggregate WF result hash.

    Payload includes ``walkforward_version`` and ``result_kind`` so a
    single-fold WF run cannot collide with a vanilla ``BacktestResult.result_hash``.
    """
    payload = {
        "engine": engine,
        "engine_version": engine_version,
        "engine_mode": engine_mode,
        "walkforward_version": WALKFORWARD_VERSION,
        "result_kind": "walkforward",
        "compiled_spec_hash": compiled_spec_hash,
        "schema_sha256": schema_sha256,
        "rows_sha256": rows_sha256,
        "config_hash": config_hash,
        "schedule": [f.definition.to_dict() for f in folds],
        "fold_result_hashes": [f.result.result_hash for f in folds],
        "aggregate": {
            "fold_count": aggregate.fold_count,
            "non_empty_fold_count": aggregate.non_empty_fold_count,
            "pooled": asdict(aggregate.pooled),
            "fold_mean": asdict(aggregate.fold_mean),
            "fold_std": asdict(aggregate.fold_std),
            "fold_sharpe_dispersion": aggregate.fold_sharpe_dispersion,
            "fold_win_rate_dispersion": aggregate.fold_win_rate_dispersion,
        },
        "warnings": [w.hashable_pair() for w in warnings],
    }
    return sha256_canonical(payload)
