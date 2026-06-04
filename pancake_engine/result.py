"""Engine-native result types and the ``result_hash`` contract.

Engine 0.3 is correctness-first, not TS parity. Known TS divergences are
documented in docs/math-audit-0.4.md.

Hash field policy (architecture §Idempotency / result-hash contract):

  IN:  engine, engine_version, engine_mode, compiled_spec_hash, schema_sha256,
       rows_sha256, config_hash, metrics, equity_curve, drawdown_curve,
       monthly_returns, trades, warnings[*].{code, severity},
       validation.future_rows_count
  OUT: duration_ms, warnings[*].{message, context}, engine_runtime_*, meta.*
       except future_rows_count
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .hash import sha256_canonical
from .validate.verdict import ValidationVerdict
from .warnings import Warning

__all__ = [
    "EquityPoint",
    "DrawdownPoint",
    "MonthlyReturn",
    "MetricsStandard",
    "MetricsPM",
    "Metrics",
    "BacktestResult",
    "compute_result_hash",
]


@dataclass(frozen=True)
class EquityPoint:
    t: int
    equity: float


@dataclass(frozen=True)
class DrawdownPoint:
    t: int
    drawdown: float


@dataclass(frozen=True)
class MonthlyReturn:
    year: int
    month: int
    return_pct: float


@dataclass(frozen=True)
class MetricsStandard:
    total_return: float
    cagr: float | None   # None when CAGR_EXTRAPOLATION_OVERFLOW fires
    sharpe: float | None
    sortino: float | None
    max_drawdown: float
    win_rate: float | None
    num_trades: int
    starting_capital: float
    ending_capital: float
    # Engine 0.4: bootstrap CI + permutation test (additive; default to safe sentinels
    # for backward-compat with 0.3 callers and old fixtures that predate 0.4).
    cagr_ci: tuple[float | None, float | None] = (None, None)
    sharpe_ci: tuple[float | None, float | None] = (None, None)
    sortino_ci: tuple[float | None, float | None] = (None, None)
    sharpe_p_value: float | None = None


@dataclass(frozen=True)
class MetricsPM:
    """Prediction-market-native strategy-level metrics.

    Per-trade PM fields (implied_prob, payoff_per_unit, etc.) live on ``Trade``;
    this struct holds the strategy-level aggregates.
    """

    win_rate_ci95_low: float | None
    win_rate_ci95_high: float | None
    mean_return_pct: float | None
    std_return_pct: float | None
    sharpe_trade_level: float | None
    sharpe_equity_curve: float | None
    brier_strategy: float | None    # null in PR-1 (rule-based spec)
    brier_crowd: float | None
    brier_skill_score: float | None
    mean_edge: float | None          # null in PR-1 (no fair_probability column)


@dataclass(frozen=True)
class Metrics:
    standard: MetricsStandard
    pm: MetricsPM


@dataclass
class BacktestResult:
    engine: str
    engine_version: str
    engine_mode: str

    compiled_spec_hash: str
    schema_sha256: str
    rows_sha256: str
    config_hash: str
    result_hash: str  # filled in by compute_result_hash after construction

    metrics: Metrics
    equity_curve: list[EquityPoint]
    drawdown_curve: list[DrawdownPoint]
    monthly_returns: list[MonthlyReturn]
    trades: list[Any]  # Trade dataclass — avoid circular import
    warnings: list[Warning]
    validation: ValidationVerdict

    meta: dict[str, Any] = field(default_factory=dict)

    # --- serialization ---

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "engine_version": self.engine_version,
            "engine_mode": self.engine_mode,
            "hashes": {
                "compiled_spec_hash": self.compiled_spec_hash,
                "schema_sha256": self.schema_sha256,
                "rows_sha256": self.rows_sha256,
                "config_hash": self.config_hash,
                "result_hash": self.result_hash,
            },
            "metrics": {
                "standard": asdict(self.metrics.standard),
                "pm": asdict(self.metrics.pm),
            },
            "equity_curve": [asdict(p) for p in self.equity_curve],
            "drawdown_curve": [asdict(p) for p in self.drawdown_curve],
            "monthly_returns": [asdict(p) for p in self.monthly_returns],
            "trades": [t.to_dict() for t in self.trades],
            "warnings": [w.to_dict() for w in self.warnings],
            "validation": self.validation.to_dict(),
            "meta": self.meta,
        }


def compute_result_hash(
    *,
    engine: str,
    engine_version: str,
    engine_mode: str,
    compiled_spec_hash: str,
    schema_sha256: str,
    rows_sha256: str,
    config_hash: str,
    metrics: Metrics,
    equity_curve: list[EquityPoint],
    drawdown_curve: list[DrawdownPoint],
    monthly_returns: list[MonthlyReturn],
    trades: list[Any],
    warnings: list[Warning],
    future_rows_count: int,
) -> str:
    """Compute ``result_hash`` over the hash-policy fields only."""
    payload: dict[str, Any] = {
        "engine": engine,
        "engine_version": engine_version,
        "engine_mode": engine_mode,
        "compiled_spec_hash": compiled_spec_hash,
        "schema_sha256": schema_sha256,
        "rows_sha256": rows_sha256,
        "config_hash": config_hash,
        "metrics": {
            "standard": asdict(metrics.standard),
            "pm": asdict(metrics.pm),
        },
        "equity_curve": [asdict(p) for p in equity_curve],
        "drawdown_curve": [asdict(p) for p in drawdown_curve],
        "monthly_returns": [asdict(p) for p in monthly_returns],
        "trades": [t.to_dict() for t in trades],
        "warnings": [w.hashable_pair() for w in warnings],
        "validation": {"future_rows_count": future_rows_count},
    }
    return sha256_canonical(payload)
