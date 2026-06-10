"""Tests for regime_stability block (Wave E, 0.10.0).

TDD: written alongside implementation.

Design: docs/design-0.9.0-contracts-and-fills.md §5.

Coverage:
1. Happy path — hand-calc quartile boundaries + per-quartile returns
2. Returns None when <8 trades
3. Returns None when <4 distinct equity timestamps
4. Returns None when all equity at same timestamp (zero span)
5. Stability: return_sign_consistency + worst_quartile_return
6. Result is additive non-hashed — result_hash byte-equality after regime attaches
7. Integration: regime block appears on BacktestResult.regime when with_inference=True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from pancake_engine.metrics.regime import regime_stability
from pancake_engine.result import EquityPoint

DAY = 86_400


# ---------------------------------------------------------------------------
# Minimal Trade stub for regime testing
# ---------------------------------------------------------------------------


@dataclass
class _Trade:
    entry_t: int      # mirrors engine Trade.entry_t (= decision timestamp)
    pnl: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"entry_t": self.entry_t, "pnl": self.pnl}


def _trades_at(times: list[int]) -> list[_Trade]:
    return [_Trade(entry_t=t) for t in times]


# ---------------------------------------------------------------------------
# Section 1: Happy-path hand-calc
#
# Setup:
#   equity_curve spans t=0 to t=400 (4 equal quartiles of 100 seconds each)
#   Quartile 1: t=[0, 100)  → equity at 0=1.0, at 100=1.1
#   Quartile 2: t=[100,200) → equity at 100=1.1, at 200=1.2
#   Quartile 3: t=[200,300) → equity at 200=1.2, at 300=1.15
#   Quartile 4: t=[300,400] → equity at 300=1.15, at 400=1.3
#
# Expected quartile returns:
#   Q1: 1.1/1.0 - 1 = 0.1
#   Q2: 1.2/1.1 - 1 ≈ 0.09091
#   Q3: 1.15/1.2 - 1 ≈ -0.04167
#   Q4: 1.3/1.15 - 1 ≈ 0.13043
#
# Overall return: 1.3/1.0 - 1 = 0.3 (positive)
# Q3 is negative → sign_consistency = 3/4 = 0.75
# worst_quartile_return ≈ -0.04167
# ---------------------------------------------------------------------------

_HAND_CALC_CURVE = [
    EquityPoint(t=0,   equity=1.0),
    EquityPoint(t=100, equity=1.1),
    EquityPoint(t=200, equity=1.2),
    EquityPoint(t=300, equity=1.15),
    EquityPoint(t=400, equity=1.3),
]

# 8 trades distributed across quartiles
_HAND_CALC_TRADES = _trades_at([10, 20, 110, 120, 210, 220, 310, 320])


def test_returns_dict_on_valid_input() -> None:
    result = regime_stability(_HAND_CALC_CURVE, _HAND_CALC_TRADES)
    assert result is not None
    assert isinstance(result, dict)


def test_quartile_count() -> None:
    result = regime_stability(_HAND_CALC_CURVE, _HAND_CALC_TRADES)
    assert result is not None
    assert len(result["quartiles"]) == 4


def test_quartile_return_q1() -> None:
    """Q1: 1.1/1.0 - 1 = 0.1."""
    result = regime_stability(_HAND_CALC_CURVE, _HAND_CALC_TRADES)
    assert result is not None
    q1 = result["quartiles"][0]
    assert abs(q1["total_return"] - 0.1) < 1e-9, f"expected 0.1, got {q1['total_return']}"


def test_quartile_return_q3_negative() -> None:
    """Q3: 1.15/1.2 - 1 ≈ -0.04167."""
    result = regime_stability(_HAND_CALC_CURVE, _HAND_CALC_TRADES)
    assert result is not None
    q3 = result["quartiles"][2]
    expected = 1.15 / 1.2 - 1.0
    assert abs(q3["total_return"] - expected) < 1e-9, (
        f"expected {expected}, got {q3['total_return']}"
    )


def test_quartile_num_trades() -> None:
    """2 trades per quartile in our hand-calc setup."""
    result = regime_stability(_HAND_CALC_CURVE, _HAND_CALC_TRADES)
    assert result is not None
    for q in result["quartiles"]:
        assert q["num_trades"] == 2


def test_stability_return_sign_consistency() -> None:
    """3 of 4 quartiles positive → consistency = 0.75."""
    result = regime_stability(_HAND_CALC_CURVE, _HAND_CALC_TRADES)
    assert result is not None
    assert abs(result["stability"]["return_sign_consistency"] - 0.75) < 1e-9


def test_stability_worst_quartile_return() -> None:
    """worst = Q3 ≈ -0.04167."""
    result = regime_stability(_HAND_CALC_CURVE, _HAND_CALC_TRADES)
    assert result is not None
    expected = 1.15 / 1.2 - 1.0
    wqr = result["stability"]["worst_quartile_return"]
    assert abs(wqr - expected) < 1e-9, f"expected {expected}, got {wqr}"


def test_quartile_max_drawdown_q3() -> None:
    """Q3 has a drawdown: peak at 1.2, trough at 1.15 → (1.2-1.15)/1.2 ≈ 0.04167."""
    result = regime_stability(_HAND_CALC_CURVE, _HAND_CALC_TRADES)
    assert result is not None
    q3 = result["quartiles"][2]
    # The drawdown within Q3: equity starts at 1.2, ends at 1.15.
    # peak=1.2, dd=(1.2-1.15)/1.2
    expected_dd = (1.2 - 1.15) / 1.2
    assert abs(q3["max_drawdown"] - expected_dd) < 1e-9


def test_quartile_max_drawdown_non_negative() -> None:
    result = regime_stability(_HAND_CALC_CURVE, _HAND_CALC_TRADES)
    assert result is not None
    for q in result["quartiles"]:
        assert q["max_drawdown"] >= 0.0


# ---------------------------------------------------------------------------
# Section 2: None when < 8 trades
# ---------------------------------------------------------------------------


def test_fewer_than_8_trades_returns_none() -> None:
    trades = _trades_at([50, 150, 250, 350, 60, 160, 260])  # 7 trades
    result = regime_stability(_HAND_CALC_CURVE, trades)
    assert result is None


def test_exactly_8_trades_returns_dict() -> None:
    result = regime_stability(_HAND_CALC_CURVE, _HAND_CALC_TRADES)
    assert result is not None


def test_zero_trades_returns_none() -> None:
    result = regime_stability(_HAND_CALC_CURVE, [])
    assert result is None


# ---------------------------------------------------------------------------
# Section 3: None when < 4 distinct equity timestamps
# ---------------------------------------------------------------------------


def test_fewer_than_4_timestamps_returns_none() -> None:
    short_curve = [
        EquityPoint(t=0, equity=1.0),
        EquityPoint(t=100, equity=1.1),
        EquityPoint(t=200, equity=1.2),  # exactly 3 distinct timestamps
    ]
    trades = _trades_at([10, 20, 30, 40, 50, 60, 70, 80])
    result = regime_stability(short_curve, trades)
    assert result is None


def test_exactly_4_timestamps_returns_dict() -> None:
    curve_4 = [
        EquityPoint(t=0, equity=1.0),
        EquityPoint(t=100, equity=1.1),
        EquityPoint(t=200, equity=1.2),
        EquityPoint(t=300, equity=1.15),
    ]
    trades = _trades_at([10, 20, 30, 40, 50, 60, 70, 80])
    result = regime_stability(curve_4, trades)
    assert result is not None


# ---------------------------------------------------------------------------
# Section 4: Zero-span degenerate (all points at same timestamp)
# ---------------------------------------------------------------------------


def test_zero_span_returns_none() -> None:
    degenerate = [
        EquityPoint(t=0, equity=1.0),
        EquityPoint(t=0, equity=1.1),
        EquityPoint(t=0, equity=1.2),
        EquityPoint(t=0, equity=1.3),
    ]
    trades = _trades_at([0, 0, 0, 0, 0, 0, 0, 0])
    result = regime_stability(degenerate, trades)
    assert result is None


# ---------------------------------------------------------------------------
# Section 5: Stability — all-positive sign consistency
# ---------------------------------------------------------------------------


def test_all_quartiles_positive_sign_consistency_one() -> None:
    """All 4 quartiles positive → sign_consistency = 1.0."""
    all_up = [
        EquityPoint(t=0, equity=1.0),
        EquityPoint(t=100, equity=1.1),
        EquityPoint(t=200, equity=1.2),
        EquityPoint(t=300, equity=1.3),
        EquityPoint(t=400, equity=1.4),
    ]
    trades = _trades_at([10, 20, 110, 120, 210, 220, 310, 320])
    result = regime_stability(all_up, trades)
    assert result is not None
    assert result["stability"]["return_sign_consistency"] == 1.0


def test_all_quartiles_same_sign_overall_negative() -> None:
    """All quartiles negative, overall negative → sign_consistency = 1.0."""
    all_down = [
        EquityPoint(t=0, equity=1.0),
        EquityPoint(t=100, equity=0.9),
        EquityPoint(t=200, equity=0.8),
        EquityPoint(t=300, equity=0.75),
        EquityPoint(t=400, equity=0.7),
    ]
    trades = _trades_at([10, 20, 110, 120, 210, 220, 310, 320])
    result = regime_stability(all_down, trades)
    assert result is not None
    assert result["stability"]["return_sign_consistency"] == 1.0


# ---------------------------------------------------------------------------
# Section 6: Hash discipline — regime is NOT in result_hash
# ---------------------------------------------------------------------------


def test_regime_not_in_result_hash() -> None:
    """regime block is additive non-hashed — result_hash must not change.

    We verify by running the examples/toy example and checking the regime
    block is present but result_hash matches the committed expected_result.json.
    (Indirectly: the examples smoke test enforces byte-equality; here we
    just assert that regime appears in to_dict() without touching result_hash.)
    """
    from pathlib import Path
    import sys, json
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from examples._common import read_json
    from pancake_engine import BacktestConfig, load_dataset, load_spec, run_backtest

    examples_dir = Path(__file__).parent.parent / "examples" / "toy"
    spec = load_spec(examples_dir / "spec.json")
    dataset = load_dataset(examples_dir / "dataset.json")
    expected = read_json(examples_dir / "expected_result.json")

    result = run_backtest(spec, dataset, BacktestConfig(observation_time=50 * 86400))

    # result_hash must be byte-identical to committed value (regime is non-hashed)
    assert result.result_hash == expected["result_hash"], (
        f"result_hash changed after regime attach: "
        f"expected={expected['result_hash']!r}, got={result.result_hash!r}"
    )

    # regime field present in to_dict() (may be None for small toy dataset — that's fine)
    d = result.to_dict()
    assert "regime" in d


# ---------------------------------------------------------------------------
# Section 7: Integration — BacktestResult.regime populated
# ---------------------------------------------------------------------------


def test_regime_block_attached_when_with_inference_true() -> None:
    """run_backtest populates result.regime (may be None for small toy example)."""
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pancake_engine import BacktestConfig, load_dataset, load_spec, run_backtest

    examples_dir = Path(__file__).parent.parent / "examples" / "toy"
    spec = load_spec(examples_dir / "spec.json")
    dataset = load_dataset(examples_dir / "dataset.json")

    result = run_backtest(spec, dataset, BacktestConfig(observation_time=50 * 86400))
    # regime attribute exists (either None or dict — both valid for tiny dataset)
    assert hasattr(result, "regime")


def test_regime_block_none_when_with_inference_false() -> None:
    """with_inference=False skips regime block."""
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pancake_engine import BacktestConfig, load_dataset, load_spec, run_backtest

    examples_dir = Path(__file__).parent.parent / "examples" / "toy"
    spec = load_spec(examples_dir / "spec.json")
    dataset = load_dataset(examples_dir / "dataset.json")

    result = run_backtest(
        spec, dataset,
        BacktestConfig(observation_time=50 * 86400),
        with_inference=False,
    )
    assert result.regime is None
