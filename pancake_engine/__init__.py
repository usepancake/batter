"""Pancake Engine 0.3 — deterministic Python research engine over EvidenceDataset.

Engine 0.3 is correctness-first, not TS parity. Known TS divergences are documented
in pancake-production/docs/research/pancake-engine-0.3-ts-divergences.md.

PR-0 shipped the canonicalization substrate.
PR-1 ships the event-time ledger runner, validation, metrics, warnings, and CLI.
"""

from .__version__ import ENGINE, ENGINE_MODE, ENGINE_VERSION, __version__
from .canonical import canonical_string, canonicalize
from .config import BacktestConfig
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
from .runner import run_backtest
from .types import EvidenceDataset, EvidenceSpec
from .validate import ValidationVerdict
from .warnings import Severity, Warning, WarningCode

__all__ = [
    "ENGINE",
    "ENGINE_MODE",
    "ENGINE_VERSION",
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
]
