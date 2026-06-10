"""batter — a deterministic backtest/verification engine for trading strategies.

Correctness-first and reproducible: identical inputs + the same engine version
produce a byte-identical ``result_hash`` on any machine. Divergences from the
reference TypeScript implementation are documented in ``docs/math-audit-0.4.md``.
"""

import sys as _sys

if _sys.version_info < (3, 12):  # pragma: no cover
    raise RuntimeError(
        "batter requires Python >= 3.12 "
        f"(got {_sys.version_info.major}.{_sys.version_info.minor}). On Python 3.11 "
        "float accumulation in sum() differs, which silently changes result_hash. "
        "See docs/py311-investigation-2026-05-27.md."
    )

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
from .pbo import PBOResult, run_pbo_analysis
from .sensitivity import SensitivityResult, run_sensitivity_analysis
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
    "run_pbo_analysis",
    "PBOResult",
    "run_sensitivity_analysis",
    "SensitivityResult",
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
