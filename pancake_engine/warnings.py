"""Structured warning system for Pancake Engine 0.3.

Every skip / clip / guard / credibility signal surfaces as a ``Warning`` with
a stable ``code`` (string enum), a ``severity``, a human-readable ``message``,
and a structured ``context`` dict.

Engine 0.3 is correctness-first, not TS parity. The TS evidence-runner skipped
silently on several paths (cash clip, out-of-range entry price, etc.); Engine
0.3 makes every skip visible via a warning.

Warning ``code`` and ``severity`` are included in ``result_hash`` (deterministic).
Warning ``message`` and ``context`` are excluded from ``result_hash`` (may carry
floating timestamps, debug strings, etc., across runs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = ["Severity", "WarningCode", "Warning"]


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class WarningCode(str, Enum):
    """Stable code identifiers for every warning Engine 0.3 emits.

    Codes are sorted into three groups for clarity:
        1. Runtime warnings (row skips, sizing clips, range guards)
        2. Credibility warnings (sample size, plausibility checks)
        3. Operational info (observation_time auto-derive, etc.)
    """

    # --- Runtime ---
    SIZING_CLIPPED = "SIZING_CLIPPED"
    SIZING_ZERO = "SIZING_ZERO"
    ENTRY_PRICE_OUT_OF_RANGE = "ENTRY_PRICE_OUT_OF_RANGE"
    FILL_PRICE_OUT_OF_RANGE = "FILL_PRICE_OUT_OF_RANGE"
    FUTURE_ROW_SKIPPED = "FUTURE_ROW_SKIPPED"
    UNRESOLVED_ROW_SKIPPED = "UNRESOLVED_ROW_SKIPPED"

    # --- Credibility ---
    LOW_SAMPLE_SIZE = "LOW_SAMPLE_SIZE"            # n < 30
    MICRO_SAMPLE_SIZE = "MICRO_SAMPLE_SIZE"        # n < 10 (severity=error, run still completes)
    IMPLAUSIBLY_HIGH_SHARPE = "IMPLAUSIBLY_HIGH_SHARPE"
    IMPLAUSIBLY_HIGH_RETURN = "IMPLAUSIBLY_HIGH_RETURN"
    DEGENERATE_HIT_RATE = "DEGENERATE_HIT_RATE"
    SINGLE_MARKET_RESULT = "SINGLE_MARKET_RESULT"
    TIME_CLUSTERED_TRADES = "TIME_CLUSTERED_TRADES"
    MARK_AT_COST_DRAWDOWN_MUTED = "MARK_AT_COST_DRAWDOWN_MUTED"
    CAGR_LOW_DUTY_CYCLE = "CAGR_LOW_DUTY_CYCLE"
    RUINED = "RUINED"
    NO_TRADES_GENERATED = "NO_TRADES_GENERATED"
    NO_TRADES_NO_CI = "NO_TRADES_NO_CI"
    BRIER_NOT_APPLICABLE = "BRIER_NOT_APPLICABLE"

    # --- Operational ---
    OBSERVATION_TIME_DERIVED = "OBSERVATION_TIME_DERIVED"

    # --- Walk-forward (PR-2) ---
    EMPTY_FOLD = "EMPTY_FOLD"
    LOW_TRADES_IN_FOLD = "LOW_TRADES_IN_FOLD"
    UNEQUAL_FOLD_SIZE = "UNEQUAL_FOLD_SIZE"
    WALKFORWARD_DISPERSION_HIGH = "WALKFORWARD_DISPERSION_HIGH"
    WALKFORWARD_SIGN_FLIP = "WALKFORWARD_SIGN_FLIP"
    WALKFORWARD_SINGLE_FOLD_CARRIES = "WALKFORWARD_SINGLE_FOLD_CARRIES"
    OVERHANG_SKIPPED = "OVERHANG_SKIPPED"
    FEATURE_LOOKAHEAD_UNCHECKED = "FEATURE_LOOKAHEAD_UNCHECKED"
    OVERRIDE_MIN_FOLD_COUNT = "OVERRIDE_MIN_FOLD_COUNT"


@dataclass(frozen=True)
class Warning:
    """One structured warning entry on a ``BacktestResult``."""

    code: WarningCode
    severity: Severity
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
        }

    def hashable_pair(self) -> dict[str, str]:
        """Only ``code`` + ``severity`` are included in ``result_hash``."""
        return {"code": self.code.value, "severity": self.severity.value}
