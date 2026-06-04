"""Crypto-OHLCV spec/dataset contract tests (ADR-0043 §2.3, P1)."""

import pytest
from pydantic import ValidationError

from pancake_engine.crypto_ohlcv import CryptoOhlcvSpec, OhlcvDataset


def valid_spec_dict() -> dict:
    return {
        "spec_family": "crypto-ohlcv-spec",
        "spec_version": "0.1",
        "name": "sma-cross",
        "instrument_id": "crypto-spot:btc-usd",
        "strategy": {
            "side": "long",
            "indicators": [
                {"id": "sma_fast", "kind": "sma", "period": 10},
                {"id": "sma_slow", "kind": "sma", "period": 30},
            ],
            "entry": {
                "op": "cross_above",
                "left": {"ref": "indicator", "indicator_id": "sma_fast"},
                "right": {"ref": "indicator", "indicator_id": "sma_slow"},
            },
            "exit": {
                "op": "cross_below",
                "left": {"ref": "indicator", "indicator_id": "sma_fast"},
                "right": {"ref": "indicator", "indicator_id": "sma_slow"},
            },
            "sizing": {"mode": "fixed_fraction", "value": 0.1},
        },
        "costs": {"slippage_bps": 5, "fee_bps": 2},
        "starting_capital": 10_000,
    }


def valid_bars() -> list[dict]:
    return [
        {"t": 0, "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1.0},
        {"t": 60, "open": 100.5, "high": 102, "low": 100, "close": 101.0, "volume": 2.0},
    ]


# --- spec --------------------------------------------------------------------


def test_valid_spec_parses_with_locked_fill_default():
    s = CryptoOhlcvSpec(**valid_spec_dict())
    assert s.spec_family == "crypto-ohlcv-spec"
    assert s.fill_timing == "next_bar_open"  # ADR-0043 locked pick (default)
    assert s.strategy.sizing.value == 0.1
    assert [i.id for i in s.strategy.indicators] == ["sma_fast", "sma_slow"]


@pytest.mark.parametrize("field,bad", [("slippage_bps", -1), ("fee_bps", -0.5)])
def test_negative_costs_rejected(field, bad):
    d = valid_spec_dict()
    d["costs"][field] = bad
    with pytest.raises(ValidationError, match="E_CRYPTO_OHLCV_SPEC_INVALID"):
        CryptoOhlcvSpec(**d)


@pytest.mark.parametrize("field", ["slippage_bps", "fee_bps"])
def test_excessive_costs_rejected(field):
    # >= 100% (10000 bps) breaks the fill/cash model; the bound makes it
    # unrepresentable so the runner can never destroy capital on a bad spec.
    d = valid_spec_dict()
    d["costs"][field] = 10_000
    with pytest.raises(ValidationError, match="must be <"):
        CryptoOhlcvSpec(**d)


def test_non_positive_capital_rejected():
    d = valid_spec_dict()
    d["starting_capital"] = 0
    with pytest.raises(ValidationError, match="starting_capital must be > 0"):
        CryptoOhlcvSpec(**d)


def test_sizing_fraction_out_of_range_rejected():
    d = valid_spec_dict()
    d["strategy"]["sizing"]["value"] = 1.5
    with pytest.raises(ValidationError, match="fixed_fraction"):
        CryptoOhlcvSpec(**d)


def test_indicator_period_must_be_positive():
    d = valid_spec_dict()
    d["strategy"]["indicators"][0]["period"] = 0
    with pytest.raises(ValidationError, match="period must be > 0"):
        CryptoOhlcvSpec(**d)


def test_unknown_indicator_ref_rejected():
    d = valid_spec_dict()
    d["strategy"]["entry"]["left"]["indicator_id"] = "does_not_exist"
    with pytest.raises(ValidationError, match="unknown indicator"):
        CryptoOhlcvSpec(**d)


def test_duplicate_indicator_id_rejected():
    d = valid_spec_dict()
    d["strategy"]["indicators"][1]["id"] = "sma_fast"
    with pytest.raises(ValidationError, match="duplicate indicator id"):
        CryptoOhlcvSpec(**d)


def test_const_operand_requires_value():
    d = valid_spec_dict()
    d["strategy"]["entry"]["right"] = {"ref": "const"}  # missing value
    with pytest.raises(ValidationError, match="const operand requires"):
        CryptoOhlcvSpec(**d)


# --- dataset -----------------------------------------------------------------


def test_valid_dataset_parses():
    ds = OhlcvDataset(instrument_id="crypto-spot:btc-usd", bars=valid_bars())
    assert ds.bar_period == "1m"
    assert ds.currency == "USD"
    assert len(ds.bars) == 2


def test_ohlc_invariant_rejected():
    bad = valid_bars()
    bad[0]["high"] = 98  # high < low/open/close
    with pytest.raises(ValidationError, match="high < low|OHLC invariant"):
        OhlcvDataset(instrument_id="x", bars=bad)


def test_negative_volume_rejected():
    bad = valid_bars()
    bad[0]["volume"] = -1
    with pytest.raises(ValidationError, match="volume must be >= 0"):
        OhlcvDataset(instrument_id="x", bars=bad)


def test_unsorted_bars_rejected():
    with pytest.raises(ValidationError, match="strictly increasing"):
        OhlcvDataset(instrument_id="x", bars=list(reversed(valid_bars())))


def test_empty_bars_rejected():
    with pytest.raises(ValidationError, match="no bars"):
        OhlcvDataset(instrument_id="x", bars=[])
