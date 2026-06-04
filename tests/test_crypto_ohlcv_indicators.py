"""Indicator tests (ADR-0043 §2.3, P3a). Reference values are hand-computed so a
regression in the determinism contract fails loudly."""

import pytest

from pancake_engine.crypto_ohlcv.indicators import compute_indicator, ema, rsi, sma

# --- SMA ---------------------------------------------------------------------


def test_sma_basic():
    assert sma([1, 2, 3, 4, 5], 3) == [None, None, 2.0, 3.0, 4.0]


def test_sma_warmup_and_degenerate():
    assert sma([1, 2], 3) == [None, None]  # n < period → all None
    assert sma([], 3) == []
    assert sma([1, 2, 3], 0) == [None, None, None]  # period <= 0 guarded


# --- EMA ---------------------------------------------------------------------


def test_ema_sma_seeded():
    # period 3 → alpha 0.5. seed i2 = mean(1,2,3) = 2; i3 = 10*.5 + 2*.5 = 6;
    # i4 = 10*.5 + 6*.5 = 8. (distinct from SMA, which would be 2, 5, 7.67)
    assert ema([1, 2, 3, 10, 10], 3) == [None, None, 2.0, 6.0, 8.0]


def test_ema_warmup():
    assert ema([1, 2], 3) == [None, None]
    assert ema([5], 1) == [5.0]  # period 1: alpha 1, seed = first value


# --- RSI (Wilder) ------------------------------------------------------------


def test_rsi_oscillating_handcomputed():
    # closes [1,2,1,2,1], period 2. deltas [+1,-1,+1,-1].
    # i2: avg_gain=mean(1,0)=.5, avg_loss=mean(0,1)=.5 → RS 1 → 50
    # i3: gain1 → avg_gain=(.5+1)/2=.75, avg_loss=(.5+0)/2=.25 → RS 3 → 75
    # i4: loss1 → avg_gain=(.75+0)/2=.375, avg_loss=(.25+1)/2=.625 → RS .6 → 37.5
    out = rsi([1, 2, 1, 2, 1], 2)
    assert out[0] is None and out[1] is None
    assert out[2] == pytest.approx(50.0)
    assert out[3] == pytest.approx(75.0)
    assert out[4] == pytest.approx(37.5)


def test_rsi_all_gains_is_100():
    out = rsi([1, 2, 3, 4, 5], 2)
    assert out[2:] == [pytest.approx(100.0)] * 3  # avg_loss 0 → 100


def test_rsi_all_losses_is_0():
    out = rsi([5, 4, 3, 2, 1], 2)
    assert out[2:] == [pytest.approx(0.0)] * 3  # avg_gain 0 → RS 0 → 0


def test_rsi_warmup_none_until_period():
    out = rsi([1, 2, 3], 3)  # n <= period → all None (needs period deltas)
    assert out == [None, None, None]


# --- dispatch ----------------------------------------------------------------


def test_compute_indicator_dispatch():
    assert compute_indicator("sma", [1, 2, 3, 4], 2) == sma([1, 2, 3, 4], 2)
    assert compute_indicator("ema", [1, 2, 3, 4], 2) == ema([1, 2, 3, 4], 2)
    assert compute_indicator("rsi", [1, 2, 3, 4], 2) == rsi([1, 2, 3, 4], 2)


def test_compute_indicator_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown indicator kind"):
        compute_indicator("macd", [1, 2, 3], 2)
