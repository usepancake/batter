"""Crypto-OHLCV strategy family (Paper Trading v2 / ADR-0043 §2.3).

A time-series strategy family for batter, parallel to (and isolated from) the
prediction-market `pancake-evidence-spec` family. It runs a minimal rules-based
strategy over an OHLCV bar series and produces a deterministic `result_hash`
receipt on the same 0.5.0 engine.

Locked design picks (ADR-0043 §2.3): fill = next-bar-open · v1 source =
price_bars 1-min · DSL = minimal (SMA/EMA/RSI + cross/threshold + fixed-fraction).

P1 ships the I/O contract (`types`). P2 adds the compiler (`compile`): a frozen
`CompiledCryptoOhlcvSpec` with a `compiled_spec_hash` + compiled entry/exit
conditions. The runner (P3) consumes the compiled form; it does not exist yet.
"""

from __future__ import annotations

from .compile import (
    CompiledCryptoOhlcvSpec,
    compile_condition,
    compile_crypto_ohlcv_spec,
    compile_operand,
)
from .indicators import compute_indicator, ema, rsi, sma
from .result import CryptoOhlcvResult, OhlcvTrade
from .run import run_crypto_ohlcv
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
    "CompiledCryptoOhlcvSpec",
    "Condition",
    "CryptoOhlcvResult",
    "CryptoOhlcvSpec",
    "CryptoOhlcvStrategy",
    "Indicator",
    "OhlcvBar",
    "OhlcvCosts",
    "OhlcvDataset",
    "OhlcvSizing",
    "OhlcvTrade",
    "Operand",
    "compile_condition",
    "compile_crypto_ohlcv_spec",
    "compile_operand",
    "compute_indicator",
    "ema",
    "rsi",
    "run_crypto_ohlcv",
    "sma",
]
