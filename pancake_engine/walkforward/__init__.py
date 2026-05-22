"""Frozen-spec walk-forward evaluation for Pancake Engine 0.3.

Engine 0.3 is correctness-first, not TS parity. The TS evidence-runner has
no walk-forward — this is an Engine 0.3 native feature.

PR-2 ships frozen-spec mode only: the engine never trains. The ``train_window``
on each ``Fold`` is metadata for documentation.
"""

from .result import (
    AggregateMetrics,
    Fold,
    FoldDefinition,
    FoldMeanMetrics,
    FoldStdMetrics,
    PooledMetrics,
    WalkforwardResult,
)
from .runner import run_walkforward
from .schedule import build_fold_schedule

__all__ = [
    "AggregateMetrics",
    "Fold",
    "FoldDefinition",
    "FoldMeanMetrics",
    "FoldStdMetrics",
    "PooledMetrics",
    "WalkforwardResult",
    "build_fold_schedule",
    "run_walkforward",
]
