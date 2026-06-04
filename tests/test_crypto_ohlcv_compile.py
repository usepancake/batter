"""Crypto-OHLCV compiler tests (ADR-0043 §2.3, P2)."""

from pancake_engine.crypto_ohlcv import CryptoOhlcvSpec
from pancake_engine.crypto_ohlcv.compile import (
    compile_condition,
    compile_crypto_ohlcv_spec,
)
from pancake_engine.crypto_ohlcv.types import Condition, Operand


def _cond(op: str, left: Operand, right: Operand) -> Condition:
    return Condition(op=op, left=left, right=right)  # type: ignore[arg-type]


def _ind(i: str) -> Operand:
    return Operand(ref="indicator", indicator_id=i)


def _price(f: str) -> Operand:
    return Operand(ref="price", field=f)  # type: ignore[arg-type]


def _const(v: float) -> Operand:
    return Operand(ref="const", value=v)


# --- threshold ops -----------------------------------------------------------


def test_threshold_gt_reads_only_cur():
    c = compile_condition(_cond("gt", _price("close"), _const(100)))
    assert c(None, {"close": 101}) is True
    assert c(None, {"close": 100}) is False  # strict
    assert c(None, {"close": 99}) is False


def test_threshold_gte_lte_lt():
    assert (
        compile_condition(_cond("gte", _price("close"), _const(100)))(None, {"close": 100}) is True
    )
    assert (
        compile_condition(_cond("lte", _price("close"), _const(100)))(None, {"close": 100}) is True
    )
    assert (
        compile_condition(_cond("lt", _price("close"), _const(100)))(None, {"close": 100}) is False
    )


def test_missing_operand_value_is_false():
    c = compile_condition(_cond("gt", _price("close"), _const(100)))
    assert c(None, {}) is False  # no close in ctx → False, never guesses


# --- cross detection (needs prev + cur) --------------------------------------


def test_cross_above():
    c = compile_condition(_cond("cross_above", _ind("fast"), _ind("slow")))
    assert c({"fast": 9, "slow": 10}, {"fast": 11, "slow": 10}) is True  # below→above
    assert c(None, {"fast": 11, "slow": 10}) is False  # first bar, no prev → no cross
    assert c({"fast": 11, "slow": 10}, {"fast": 12, "slow": 10}) is False  # already above prev
    assert c({"fast": 9, "slow": 10}, {"fast": 9.5, "slow": 10}) is False  # didn't reach above


def test_cross_below():
    c = compile_condition(_cond("cross_below", _ind("fast"), _ind("slow")))
    assert c({"fast": 11, "slow": 10}, {"fast": 9, "slow": 10}) is True  # above→below
    assert c({"fast": 9, "slow": 10}, {"fast": 8, "slow": 10}) is False  # already below prev


# --- compiled spec -----------------------------------------------------------


def _valid_spec() -> CryptoOhlcvSpec:
    return CryptoOhlcvSpec(
        spec_family="crypto-ohlcv-spec",
        spec_version="0.1",
        name="sma-cross",
        instrument_id="crypto-spot:btc-usd",
        strategy={
            "side": "long",
            "indicators": [
                {"id": "sma_fast", "kind": "sma", "period": 10},
                {"id": "sma_slow", "kind": "sma", "period": 30},
            ],
            "entry": {
                "op": "cross_above",
                "left": _ind("sma_fast").model_dump(),
                "right": _ind("sma_slow").model_dump(),
            },
            "exit": {
                "op": "cross_below",
                "left": _ind("sma_fast").model_dump(),
                "right": _ind("sma_slow").model_dump(),
            },
            "sizing": {"mode": "fixed_fraction", "value": 0.1},
        },  # type: ignore[arg-type]
        costs={"slippage_bps": 5, "fee_bps": 2},  # type: ignore[arg-type]
        starting_capital=10_000,
    )


def test_compiled_spec_hash_deterministic_and_props():
    s = _valid_spec()
    a = compile_crypto_ohlcv_spec(s)
    b = compile_crypto_ohlcv_spec(s)
    assert a.compiled_spec_hash == b.compiled_spec_hash  # determinism
    assert len(a.compiled_spec_hash) == 64
    assert a.side == "long"
    assert a.sizing_value == 0.1
    assert a.fill_timing == "next_bar_open"
    assert a.slippage_bps == 5 and a.fee_bps == 2
    assert len(a.indicators) == 2
    # compiled entry actually fires on a fast-over-slow cross
    assert a.entry({"sma_fast": 9, "sma_slow": 10}, {"sma_fast": 11, "sma_slow": 10}) is True
