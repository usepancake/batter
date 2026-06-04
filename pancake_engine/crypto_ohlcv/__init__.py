"""Crypto-OHLCV strategy family (Paper Trading v2 / ADR-0043 §2.3).

A time-series strategy family for batter, parallel to (and isolated from) the
prediction-market `pancake-evidence-spec` family. It runs a minimal rules-based
strategy over an OHLCV bar series and produces a deterministic `result_hash`
receipt on the same 0.5.0 engine.

Locked design picks (ADR-0043 §2.3): fill = next-bar-open · v1 source =
price_bars 1-min · DSL = minimal (SMA/EMA/RSI + cross/threshold + fixed-fraction).

P1 ships the I/O contract only (this package's `types`). The compiler (P2) and
runner (P3) consume these models; neither exists yet.
"""

from __future__ import annotations

from .types import (
    Condition,
    CryptoOhlcvSpec,
    CryptoOhlcvStrategy,
    Indicator,
    OhlcvBar,
    OhlcvCosts,
    OhlcvDataset,
    OhlcvSizing,
    Operand,
)

__all__ = [
    "Condition",
    "CryptoOhlcvSpec",
    "CryptoOhlcvStrategy",
    "Indicator",
    "OhlcvBar",
    "OhlcvCosts",
    "OhlcvDataset",
    "OhlcvSizing",
    "Operand",
]
