"""Crypto-OHLCV backtest runner (ADR-0043 §2.3, P3b).

Pure function `(spec, dataset) -> CryptoOhlcvResult`. No clock, no I/O, no
network — same inputs → same `result_hash`. Engine-native / correctness-first
(there is no TS OHLCV runner to parity against).

Execution model (the determinism contract):
- One position at a time; sizing = `fixed_fraction × cash` at entry.
- **Next-bar-open fill**: a signal evaluated from bars ≤ i executes at bar i+1's
  OPEN. An entry signal on the last bar never fills (no next bar). Any position
  still open after the last bar is force-closed at the last bar's CLOSE
  (`exit_reason="end_of_data"`).
- Slippage (multiplicative bps): a BUY pays `open·(1+slip)`, a SELL gets
  `open·(1−slip)`. Long = buy→sell; short = sell→buy.
- Fees (bps): entry fee on notional; exit fee on exit gross.
- Equity is marked to each bar's CLOSE: `equity = cash + qty·close` (qty signed:
  + long, − short). This series drives the metrics.
- Accounting (verified for both sides; equity at entry = cash₀ − entry_fee):
    long  → shares = (notional−fee)/entry_fill; cash −= notional;        qty = +shares
    short → shares =  notional/entry_fill;       cash += (notional−fee);  qty = −shares
    pnl   → dir·shares·(exit_fill − entry_fill) − entry_fee − exit_fee   (dir = +1/−1)
"""

from __future__ import annotations

from ..__version__ import ENGINE, ENGINE_VERSION
from ..metrics import (
    build_drawdown_curve,
    build_monthly_returns,
    compute_standard,
    daily_returns_carry_forward,
)
from ..result import DrawdownPoint, EquityPoint
from .compile import Ctx, compile_crypto_ohlcv_spec
from .indicators import compute_indicator
from .result import CryptoOhlcvResult, OhlcvTrade, compute_crypto_result_hash
from .types import CryptoOhlcvSpec, OhlcvDataset

__all__ = ["run_crypto_ohlcv", "ENGINE_MODE"]

ENGINE_MODE = "crypto_ohlcv_v1"
BPS_DIVISOR = 10_000
_ERR = "E_CRYPTO_OHLCV_RUN_INVALID"


def run_crypto_ohlcv(spec: CryptoOhlcvSpec, dataset: OhlcvDataset) -> CryptoOhlcvResult:
    if spec.instrument_id != dataset.instrument_id:
        raise ValueError(
            f"{_ERR}: spec.instrument_id {spec.instrument_id!r} != "
            f"dataset.instrument_id {dataset.instrument_id!r}"
        )

    compiled = compile_crypto_ohlcv_spec(spec)
    bars = dataset.bars
    n = len(bars)  # dataset validator guarantees n >= 1

    # 1. indicators per id over their source series
    ind_series: dict[str, list[float | None]] = {}
    for ind in spec.strategy.indicators:
        series = [getattr(b, ind.source) for b in bars]
        ind_series[ind.id] = compute_indicator(ind.kind, series, ind.period)

    # 2. per-bar evaluation context (price fields + defined indicator values)
    contexts: list[Ctx] = []
    for i, b in enumerate(bars):
        ctx: Ctx = {"open": b.open, "high": b.high, "low": b.low, "close": b.close}
        for iid, vals in ind_series.items():
            v = vals[i]
            if v is not None:
                ctx[iid] = v
        contexts.append(ctx)

    side = compiled.side
    slip = compiled.slippage_bps / BPS_DIVISOR
    fee_rate = compiled.fee_bps / BPS_DIVISOR
    direction = 1.0 if side == "long" else -1.0

    cash = float(compiled.starting_capital)
    qty = 0.0  # signed position size; 0 = flat
    # open-position fields (meaningful only while qty != 0)
    o_entry_t = 0
    o_entry_fill = 0.0
    o_entry_quote = 0.0
    o_notional = 0.0
    o_entry_fee = 0.0
    o_entry_idx = 0

    trades: list[OhlcvTrade] = []
    equity_curve: list[EquityPoint] = []
    warnings: list[str] = []
    pending: str | None = None

    def close_at(quote: float, exit_t: int, exit_idx: int, reason: str) -> None:
        nonlocal cash, qty
        shares = abs(qty)
        if side == "long":
            fill = quote * (1 - slip)  # sell
            gross = shares * fill
            exit_fee = gross * fee_rate
            cash += gross - exit_fee
        else:  # short → buy back
            fill = quote * (1 + slip)
            gross = shares * fill
            exit_fee = gross * fee_rate
            cash -= gross + exit_fee
        pnl = direction * shares * (fill - o_entry_fill) - o_entry_fee - exit_fee
        trades.append(
            OhlcvTrade(
                side=side,
                entry_t=o_entry_t,
                exit_t=exit_t,
                entry_price=o_entry_fill,
                entry_price_quote=o_entry_quote,
                exit_price=fill,
                exit_price_quote=quote,
                exit_reason=reason,
                shares=shares,
                notional=o_notional,
                entry_fee=o_entry_fee,
                exit_fee=exit_fee,
                pnl=pnl,
                return_pct=(pnl / o_notional if o_notional else 0.0),
                bars_held=exit_idx - o_entry_idx,
            )
        )
        qty = 0.0

    for i, b in enumerate(bars):
        # 1. execute the action decided on the previous bar, at THIS bar's open
        if pending == "enter" and qty == 0:
            notional = cash * compiled.sizing_value
            if notional <= 0:
                warnings.append(f"notional <= 0 at t={b.t}; entry skipped")
            else:
                fee = notional * fee_rate
                if side == "long":
                    fill = b.open * (1 + slip)
                    qty = (notional - fee) / fill
                    cash -= notional
                else:  # short
                    fill = b.open * (1 - slip)
                    if fill <= 0:
                        fill = 0.0
                        warnings.append(f"non-positive short fill at t={b.t}; entry skipped")
                    else:
                        qty = -(notional / fill)
                        cash += notional - fee
                if qty != 0:
                    o_entry_t, o_entry_fill, o_entry_quote = b.t, fill, b.open
                    o_notional, o_entry_fee, o_entry_idx = notional, fee, i
        elif pending == "exit" and qty != 0:
            close_at(b.open, b.t, i, "signal")
        pending = None  # consumed (signal not re-armed unless it fires again below)

        # 2. evaluate signals on this bar (prev, cur) for the NEXT bar's open
        prev = contexts[i - 1] if i > 0 else None
        cur = contexts[i]
        if qty == 0:
            if compiled.entry(prev, cur):
                pending = "enter"
        else:
            if compiled.exit(prev, cur):
                pending = "exit"

        # 3. mark equity at this bar's close
        equity_curve.append(EquityPoint(t=b.t, equity=cash + qty * b.close))

    # force-close any still-open position at the last bar's close
    if qty != 0:
        last = bars[-1]
        close_at(last.close, last.t, n - 1, "end_of_data")
        equity_curve[-1] = EquityPoint(t=last.t, equity=cash)
        warnings.append(f"open position force-closed at end of data (t={last.t})")

    # 3b. series + metrics (reuse the engine-native primitives)
    daily_rets = daily_returns_carry_forward(equity_curve)
    drawdown_curve, _ = build_drawdown_curve(equity_curve)
    if len(equity_curve) == 1:
        drawdown_curve = [DrawdownPoint(t=equity_curve[0].t, drawdown=0.0)]
    monthly_returns = build_monthly_returns(equity_curve) if len(equity_curve) >= 2 else []
    period_seconds = max(equity_curve[-1].t - equity_curve[0].t, 1)

    metrics, _ruined, _cagr_overflowed, metric_warnings = compute_standard(
        trades=trades,  # type: ignore[arg-type]  # compute_standard reads only .pnl
        equity_curve=equity_curve,
        daily_rets=daily_rets,
        starting_capital=float(compiled.starting_capital),
        period_seconds=period_seconds,
    )
    warnings.extend(w.message for w in metric_warnings)

    dataset_hash = _dataset_hash(dataset)
    result_hash = compute_crypto_result_hash(
        engine=ENGINE,
        engine_version=ENGINE_VERSION,
        engine_mode=ENGINE_MODE,
        compiled_spec_hash=compiled.compiled_spec_hash,
        dataset_hash=dataset_hash,
        metrics=metrics,
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        monthly_returns=monthly_returns,
        trades=trades,
    )

    return CryptoOhlcvResult(
        engine=ENGINE,
        engine_version=ENGINE_VERSION,
        engine_mode=ENGINE_MODE,
        compiled_spec_hash=compiled.compiled_spec_hash,
        dataset_hash=dataset_hash,
        result_hash=result_hash,
        metrics=metrics,
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        monthly_returns=monthly_returns,
        trades=trades,
        warnings=warnings,
        meta={
            "instrument_id": spec.instrument_id,
            "bar_count": n,
            "trade_count": len(trades),
            "ending_capital": cash,
            "fill_timing": compiled.fill_timing,
        },
    )


def _dataset_hash(dataset: OhlcvDataset) -> str:
    from ..hash import sha256_canonical

    return sha256_canonical(dataset.model_dump(by_alias=True, exclude_none=True, mode="python"))
