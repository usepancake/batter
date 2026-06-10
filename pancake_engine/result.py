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
    # Engine 0.8: CIs are now STATIONARY block-bootstrap (Politis-Romano) — they
    # preserve serial correlation, so they are wider/honester than the 0.4 IID CIs.
    cagr_ci: tuple[float | None, float | None] = (None, None)
    sharpe_ci: tuple[float | None, float | None] = (None, None)
    sortino_ci: tuple[float | None, float | None] = (None, None)
    sharpe_p_value: float | None = None
    # Engine 0.8: credibility signals (Bailey & López de Prado). psr = probability the
    # true Sharpe exceeds 0 given sample length + skew/kurtosis; min_track_record_length
    # = observations needed for the Sharpe to be significant at 95%. Both None when
    # with_inference is skipped (the sweep) or undefined (n<2 / zero variance).
    psr: float | None = None
    min_track_record_length: float | None = None


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
    # 0.9.0 (HASHED — deliberate result_hash break): Expected Calibration Error
    # over 10 fixed bins on the traded-side (implied_prob_at_entry,
    # realized_outcome_for_trade) pairs.  None when num_trades < 10.
    calibration_ece: float | None = None


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
    # 0.8: transaction-cost sensitivity curve + break-even multiplier (additive;
    # NOT in result_hash — compute_result_hash omits it). None when no trades or
    # with_inference is skipped (the sweep).
    cost_sensitivity: dict[str, Any] | None = None
    # 0.8: buy-and-hold baseline block (spec v0.2 subset; no-filter convention).
    # The REQUEST is hashed (baseline field lives in the spec → compiled_spec_hash);
    # this OUTPUT block is additive/non-hashed until the 0.9.0 break folds it in.
    baseline: dict[str, Any] | None = None
    # 0.9: deflated Sharpe block (additive; NOT in result_hash — execution argument
    # trial_history is not part of config/config_hash). None when trial_history is
    # not supplied, with_inference is skipped, or daily returns are undefined.
    # Shape: {"dsr": float|None, "n_trials": int, "source": str, "own_sharpe_included": True}
    deflated: dict[str, Any] | None = None
    # 0.9: calibration reliability curve (additive; NOT in result_hash — cost_sensitivity
    # pattern; scalar headline calibration_ece IS hashed via MetricsPM). None when
    # num_trades < 10 or with_inference is skipped (fast path).
    # Shape: {"bins": [{"bin_low", "bin_high", "n", "confidence", "accuracy"}, …], "n_trades": int}
    calibration_bins: dict[str, Any] | None = None

    # 0.9.x Wave A: book_replay@1 provenance fields.  Set only when the spec uses
    # fill_model book_replay@1 AND a book_dataset was supplied to run_backtest.
    # These are included in result_hash ONLY when book replay ran (conditional payload
    # key "book" in compute_result_hash) — specs that don't use book_replay hash
    # byte-identically to pre-Wave-A receipts.
    book_dataset_id: str | None = None
    book_rows_sha256: str | None = None
    book_schema_sha256: str | None = None

    # 0.10.0 Wave E: regime-stability block (additive; NOT in result_hash — same
    # pattern as calibration_bins / cost_sensitivity / baseline). None when
    # with_inference is skipped or the run has <8 trades / <4 equity timestamps.
    # Shape: {"quartiles": [{quartile, t_start, t_end, num_trades, total_return,
    #                        max_drawdown}, ...], "stability": {return_sign_consistency,
    #                        worst_quartile_return}}
    regime: dict[str, Any] | None = None

    # --- serialization ---

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
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
            "cost_sensitivity": self.cost_sensitivity,
            "baseline": self.baseline,
            "deflated": self.deflated,
            "calibration_bins": self.calibration_bins,
            "regime": self.regime,
        }
        if self.book_dataset_id is not None:
            d["book_dataset_id"] = self.book_dataset_id
            d["book_rows_sha256"] = self.book_rows_sha256
            d["book_schema_sha256"] = self.book_schema_sha256
        return d


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
    book_dataset_id: str | None = None,
    book_rows_sha256: str | None = None,
    book_schema_sha256: str | None = None,
) -> str:
    """Compute ``result_hash`` over the hash-policy fields only.

    Hash-conditionality for book_replay@1 (0.9.x Wave A):
    When ``book_dataset_id`` is supplied (i.e. book_replay ran), the payload
    gains a ``"book"`` key containing the three book provenance identifiers.
    When absent (all other fill models), the ``"book"`` key is not present and
    the payload is byte-identical to pre-Wave-A receipts — no hash break for
    existing specs.
    """
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
    # Absent key for all non-book-replay specs keeps existing hashes byte-identical.
    if book_dataset_id is not None:
        payload["book"] = {
            "book_dataset_id": book_dataset_id,
            "book_rows_sha256": book_rows_sha256,
            "book_schema_sha256": book_schema_sha256,
        }
    return sha256_canonical(payload)
