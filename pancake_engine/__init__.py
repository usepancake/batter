"""Pancake Engine 0.3 — deterministic Python research engine over EvidenceDataset.

Engine 0.3 is correctness-first, not TS parity. Known TS divergences are documented
in pancake-production/docs/research/pancake-engine-0.3-ts-divergences.md.

PR-0 shipped the canonicalization substrate.
PR-1 ships the event-time ledger runner, validation, metrics, warnings, and CLI.
"""

from .__version__ import (
    ENGINE,
    ENGINE_MODE,
    ENGINE_VERIFICATION_GRADE,
    ENGINE_VERSION,
    __version__,
)
from .canonical import canonical_string, canonicalize
from .config import BacktestConfig, WalkforwardConfig
from .hash import sha256_canonical
from .io.dump import dump_result, result_to_canonical_json
from .io.load import load_dataset, load_json, load_spec
from .result import (
    BacktestResult,
    DrawdownPoint,
    EquityPoint,
    Metrics,
    MetricsPM,
    MetricsStandard,
    MonthlyReturn,
)
from .runner import (
    Fill,
    FillRejection,
    MarketBar,
    PaperEvent,
    ResolutionMarker,
    SimFillRouter,
    TickError,
    TickPosition,
    TickRequest,
    TickResponse,
    VerificationBoundary,
    run_backtest,
    tick,
)
from .types import EvidenceDataset, EvidenceSpec
from .validate import ValidationVerdict
from .walkforward import (
    AggregateMetrics,
    Fold,
    FoldDefinition,
    WalkforwardResult,
    run_walkforward,
)
from .warnings import Severity, Warning, WarningCode

__all__ = [
    "ENGINE",
    "ENGINE_MODE",
    "ENGINE_VERSION",
    "ENGINE_VERIFICATION_GRADE",
    "__version__",
    "canonical_string",
    "canonicalize",
    "sha256_canonical",
    "load_dataset",
    "load_json",
    "load_spec",
    "dump_result",
    "result_to_canonical_json",
    "EvidenceDataset",
    "EvidenceSpec",
    "BacktestConfig",
    "BacktestResult",
    "Metrics",
    "MetricsStandard",
    "MetricsPM",
    "EquityPoint",
    "DrawdownPoint",
    "MonthlyReturn",
    "Severity",
    "Warning",
    "WarningCode",
    "ValidationVerdict",
    "run_backtest",
    "WalkforwardConfig",
    "WalkforwardResult",
    "Fold",
    "FoldDefinition",
    "AggregateMetrics",
    "run_walkforward",
    # ADR-0035 paper /tick surface
    "tick",
    "SimFillRouter",
    "Fill",
    "FillRejection",
    "MarketBar",
    "ResolutionMarker",
    "TickPosition",
    "PaperEvent",
    "VerificationBoundary",
    "TickRequest",
    "TickResponse",
    "TickError",
]
