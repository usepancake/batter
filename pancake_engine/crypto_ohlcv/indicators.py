"""Technical indicators for the crypto-OHLCV family (ADR-0043 §2.3, P3a).

Pure, deterministic, no look-ahead. Each function takes a source series
(`list[float]`, e.g. the close prices) and returns a list of the SAME length,
with `None` during the warm-up window (insufficient history) and a float once
the indicator is defined. Index `i` of the output uses ONLY `values[0..i]` — no
future bar ever influences a past value. The runner (P3b) aligns these outputs
to bars positionally.

These values feed `result_hash`, so the computation semantics are a frozen
contract. The choices below are the standard conventions (TradingView / ta-lib
`adjust=False`); changing them changes every crypto receipt's hash.

- **SMA(period)**: arithmetic mean of the trailing `period` values. Defined from
  index `period-1`. `None` before.
- **EMA(period)**: seeded with the SMA of the first `period` values at index
  `period-1`; thereafter `ema[i] = v[i]*alpha + ema[i-1]*(1-alpha)` with
  `alpha = 2/(period+1)`. `None` before index `period-1`. (SMA-seeded EMA — the
  ta-lib / TradingView / pandas-ta `adjust=False` convention.)
- **RSI(period)**: Wilder's smoothing (the canonical RSI). Close-to-close deltas;
  the first average gain/loss at index `period` is the simple mean of the first
  `period` deltas; thereafter Wilder's recursive smoothing
  `avg = (avg_prev*(period-1) + cur)/period`. `RSI = 100 - 100/(1+RS)` with
  `RS = avg_gain/avg_loss`; `RSI = 100.0` when `avg_loss == 0`. Defined from
  index `period`. `None` before.
"""

from __future__ import annotations

__all__ = ["sma", "ema", "rsi", "compute_indicator"]


def sma(values: list[float], period: int) -> list[float | None]:
    n = len(values)
    out: list[float | None] = [None] * n
    if period <= 0 or n < period:
        return out
    window = sum(values[:period])
    out[period - 1] = window / period
    for i in range(period, n):
        window += values[i] - values[i - period]
        out[i] = window / period
    return out


def ema(values: list[float], period: int) -> list[float | None]:
    n = len(values)
    out: list[float | None] = [None] * n
    if period <= 0 or n < period:
        return out
    alpha = 2.0 / (period + 1)
    prev = sum(values[:period]) / period  # SMA seed
    out[period - 1] = prev
    for i in range(period, n):
        prev = values[i] * alpha + prev * (1 - alpha)
        out[i] = prev
    return out


def rsi(values: list[float], period: int) -> list[float | None]:
    n = len(values)
    out: list[float | None] = [None] * n
    if period <= 0 or n <= period:
        return out

    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):  # first `period` deltas → seed average
        delta = values[i] - values[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses += -delta
    avg_gain = gains / period
    avg_loss = losses / period
    out[period] = _rsi_from_avgs(avg_gain, avg_loss)

    for i in range(period + 1, n):
        delta = values[i] - values[i - 1]
        gain = delta if delta > 0 else 0.0
        loss = -delta if delta < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = _rsi_from_avgs(avg_gain, avg_loss)
    return out


def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def compute_indicator(kind: str, values: list[float], period: int) -> list[float | None]:
    """Dispatch by `Indicator.kind` (sma|ema|rsi). The compiler/runner uses this
    so the kind→function mapping lives in one place."""
    if kind == "sma":
        return sma(values, period)
    if kind == "ema":
        return ema(values, period)
    if kind == "rsi":
        return rsi(values, period)
    raise ValueError(f"E_CRYPTO_OHLCV_SPEC_INVALID: unknown indicator kind {kind!r}")
