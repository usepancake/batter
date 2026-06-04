"""Regression: daily_returns_carry_forward must use the LAST same-day equity on
the start day (intraday updates), not the first point.

Found by the batter math-audit swarm (2026-06-04). Latent for one-point-per-UTC-day
inputs (current Polymarket receipts), but the buggy branch fires on EVERY start day
for intraday cadence (crypto 1-min bars) — so it must be fixed before extending the
engine to crypto/OHLCV strategies.
"""

import pytest

from pancake_engine.metrics.series import daily_returns_carry_forward
from pancake_engine.result import EquityPoint


def test_start_day_uses_last_same_day_equity():
    # Two equity points on day 0 (intraday); day-0 close = 1050. Day 1 = 1040.
    curve = [
        EquityPoint(t=0, equity=1000.0),       # day 0, 00:00:00 (start)
        EquityPoint(t=3_600, equity=1050.0),   # day 0, 01:00:00 (last same-day -> close)
        EquityPoint(t=86_400, equity=1040.0),  # day 1, 00:00:00
    ]
    rets = daily_returns_carry_forward(curve)
    # Correct: day-0 close 1050 -> day-1 1040 => a single return of 1040/1050 - 1.
    # Pre-fix bug returned [0.04] (used the FIRST day-0 point, 1000).
    assert len(rets) == 1
    assert rets[0] == pytest.approx(1040.0 / 1050.0 - 1.0)


def test_one_point_per_day_unchanged():
    # No-regression guard: the single-point-per-UTC-day path (existing receipts)
    # must be byte-identical after the fix.
    curve = [
        EquityPoint(t=0, equity=1000.0),
        EquityPoint(t=86_400, equity=1100.0),
        EquityPoint(t=172_800, equity=1050.0),
    ]
    rets = daily_returns_carry_forward(curve)
    assert rets[0] == pytest.approx(0.1)
    assert rets[1] == pytest.approx(1050.0 / 1100.0 - 1.0)
