"""Crypto-OHLCV result types + result_hash (ADR-0043 §2.3, P3b).

Reuses the engine-native series/metric primitives (`EquityPoint`,
`DrawdownPoint`, `MonthlyReturn`, `MetricsStandard`) but defines a crypto-native
`OhlcvTrade` (dollar prices, long/short, no settle 0/1) and a crypto result with
its own `engine_mode = "crypto_ohlcv_v1"`. The PM metrics block (brier, edge,
implied-prob) is intentionally absent — those are prediction-market concepts.

`result_hash` covers: engine identity (incl. ENGINE_VERSION), compiled_spec_hash,
dataset_hash, metrics, equity/drawdown/monthly curves, trades. Same inputs → same
hash; the `engine_mode` string partitions crypto receipts from evidence receipts
in the hash space.

`warnings` are intentionally EXCLUDED from the hash. They are free-text and a
pure function of the hashed inputs (so they never disambiguate two runs), and
hashing message text would make a future wording edit silently invalidate every
receipt. (The evidence path hashes only structured (code, severity) pairs for
the same reason; crypto v1 has no warning codes, so it hashes none.)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..hash import sha256_canonical
from ..result import DrawdownPoint, EquityPoint, MetricsStandard, MonthlyReturn
from ..validate.verdict import ValidationVerdict

__all__ = [
    "OhlcvTrade",
    "CryptoOhlcvResult",
    "compute_crypto_result_hash",
]


@dataclass(frozen=True)
class OhlcvTrade:
    side: str  # "long" | "short"
    entry_t: int  # bar-open epoch of the fill bar
    exit_t: int
    entry_price: float  # post-slippage fill
    entry_price_quote: float  # pre-slippage quote (the bar open)
    exit_price: float  # post-slippage fill
    exit_price_quote: float  # pre-slippage quote
    exit_reason: str  # "signal" | "end_of_data"
    shares: float  # absolute units (> 0)
    notional: float  # sizing basis at entry (= cash * fraction)
    entry_fee: float
    exit_fee: float
    pnl: float
    return_pct: float  # pnl / notional
    bars_held: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CryptoOhlcvResult:
    engine: str
    engine_version: str
    engine_mode: str

    compiled_spec_hash: str
    dataset_hash: str
    result_hash: str  # "" when blocked (validation failed)

    metrics: MetricsStandard
    equity_curve: list[EquityPoint]
    drawdown_curve: list[DrawdownPoint]
    monthly_returns: list[MonthlyReturn]
    trades: list[OhlcvTrade]
    warnings: list[str]
    validation: ValidationVerdict

    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "engine_version": self.engine_version,
            "engine_mode": self.engine_mode,
            "hashes": {
                "compiled_spec_hash": self.compiled_spec_hash,
                "dataset_hash": self.dataset_hash,
                "result_hash": self.result_hash,
            },
            "metrics": asdict(self.metrics),
            "equity_curve": [asdict(p) for p in self.equity_curve],
            "drawdown_curve": [asdict(p) for p in self.drawdown_curve],
            "monthly_returns": [asdict(p) for p in self.monthly_returns],
            "trades": [t.to_dict() for t in self.trades],
            "warnings": list(self.warnings),
            "validation": self.validation.to_dict(),
            "meta": self.meta,
        }


def compute_crypto_result_hash(
    *,
    engine: str,
    engine_version: str,
    engine_mode: str,
    compiled_spec_hash: str,
    dataset_hash: str,
    metrics: MetricsStandard,
    equity_curve: list[EquityPoint],
    drawdown_curve: list[DrawdownPoint],
    monthly_returns: list[MonthlyReturn],
    trades: list[OhlcvTrade],
) -> str:
    payload: dict[str, Any] = {
        "engine": engine,
        "engine_version": engine_version,
        "engine_mode": engine_mode,
        "compiled_spec_hash": compiled_spec_hash,
        "dataset_hash": dataset_hash,
        "metrics": asdict(metrics),
        "equity_curve": [asdict(p) for p in equity_curve],
        "drawdown_curve": [asdict(p) for p in drawdown_curve],
        "monthly_returns": [asdict(p) for p in monthly_returns],
        "trades": [t.to_dict() for t in trades],
    }
    return sha256_canonical(payload)
