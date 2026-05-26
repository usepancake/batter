"""AF-3 edge-case test: extreme compound-win scenarios that can push equity to float64 overflow.

Verifies that 100k all-wins at sizing=0.01 (a low-price entry) either:
  (a) produces a finite result_hash without raising E_NONFINITE, AND
  (b) emits EQUITY_OVERFLOW_BOUND if equity overflowed float64, OR
  (c) returns a finite ending_capital if it didn't overflow.

The test MUST NOT raise an unhandled exception. Prior to the AF-3 fix, sha256_canonical
raised ValueError("E_NONFINITE: Infinity is not representable in canonical form").
"""

from __future__ import annotations

import math
import pytest

from pancake_engine import BacktestConfig, run_backtest
from pancake_engine.warnings import WarningCode

from ._runner_helpers import make_dataset, make_spec, row

DAY = 86_400


def _build_all_wins_dataset(n: int, price: float = 0.01) -> object:
    """Build a dataset of n sequential all-win trades at the given price."""
    rows = []
    for i in range(n):
        dec_ts = i * 2 * DAY
        res_ts = dec_ts + DAY
        rows.append(row(
            mkt=f"m/{i}",
            dec_ts=dec_ts,
            res_ts=res_ts,
            price=price,
            outcome=1,
            alpha=3.0,
            target=1,
        ))
    return make_dataset(rows)


def test_af3_100k_all_wins_no_nonfinite_error() -> None:
    """AF-3: 100k all-wins at sizing=0.01 must not raise E_NONFINITE.

    Each win at price=0.01 multiplies the notional by ~100×. After enough
    compounding, equity can overflow float64. The AF-3 fix clamps the overflow
    and emits EQUITY_OVERFLOW_BOUND instead of crashing.
    """
    n = 100_000
    spec = make_spec(side="YES", sizing_value=0.01, starting_capital=1000.0)
    dataset = _build_all_wins_dataset(n, price=0.01)
    obs_time = (n - 1) * 2 * DAY + DAY + 1
    config = BacktestConfig(observation_time=obs_time)

    # Must NOT raise any exception (in particular, no ValueError: E_NONFINITE).
    r = run_backtest(spec, dataset, config)

    # result_hash must be a non-empty hex string (proves sha256_canonical succeeded).
    assert r.result_hash != "", "result_hash must not be empty"
    assert len(r.result_hash) == 64, f"expected 64-char sha256 hex, got {len(r.result_hash)}"

    # Either EQUITY_OVERFLOW_BOUND was emitted (overflow path) or ending_capital is finite.
    codes = {w.code for w in r.warnings}
    ending = r.metrics.standard.ending_capital
    if WarningCode.EQUITY_OVERFLOW_BOUND in codes:
        # Overflow path: ending_capital must be finite (clamped to float_max).
        assert math.isfinite(ending), (
            f"EQUITY_OVERFLOW_BOUND was emitted but ending_capital is not finite: {ending}"
        )
    else:
        # No overflow (possible if sizing compounds more slowly than expected).
        assert math.isfinite(ending), (
            f"No overflow warning but ending_capital is non-finite: {ending}"
        )


def test_af3_determinism_after_clamp() -> None:
    """AF-3: Two independent runs of the overflow scenario produce the same result_hash."""
    n = 100_000
    spec = make_spec(side="YES", sizing_value=0.01, starting_capital=1000.0)
    dataset = _build_all_wins_dataset(n, price=0.01)
    obs_time = (n - 1) * 2 * DAY + DAY + 1
    config = BacktestConfig(observation_time=obs_time)

    r1 = run_backtest(spec, dataset, config)
    r2 = run_backtest(spec, dataset, config)

    assert r1.result_hash != "", "result_hash must not be empty"
    assert r1.result_hash == r2.result_hash, (
        f"Non-deterministic hash after AF-3 clamp: {r1.result_hash} != {r2.result_hash}"
    )
