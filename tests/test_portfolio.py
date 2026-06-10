"""Tests for portfolio receipt (Wave E, 0.10.0).

TDD: written alongside implementation.

Design: docs/design-0.9.0-contracts-and-fills.md §5.

Coverage:
1. Happy path — 2-leg hand-calc (joint equity + correlation)
2. Weights validation (negative, zero, non-sum-1, length mismatch)
3. Legs validation (< 2, blocked leg, empty equity_curve, identity mismatch)
4. Determinism (3 identical calls produce identical portfolio_hash)
5. Hash discipline — portfolio_hash is in to_dict() output
6. Correlation degenerate cases (zero-variance leg → None)
7. to_dict() round-trip shape
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from pancake_engine.portfolio import PortfolioError, PortfolioResult, compute_portfolio
from pancake_engine.result import (
    BacktestResult,
    DrawdownPoint,
    EquityPoint,
    Metrics,
    MetricsPM,
    MetricsStandard,
    MonthlyReturn,
)
from pancake_engine.validate.verdict import ValidationVerdict
from pancake_engine.warnings import Warning


# ---------------------------------------------------------------------------
# Test fixtures helpers
# ---------------------------------------------------------------------------


def _ok_verdict() -> ValidationVerdict:
    v = ValidationVerdict()
    return v  # no errors → ok


def _blocked_verdict() -> ValidationVerdict:
    v = ValidationVerdict()
    v.add_error("E_BLOCKED", "blocked for test")
    return v


def _make_metrics(tr: float = 0.1) -> Metrics:
    std = MetricsStandard(
        total_return=tr, cagr=tr, sharpe=1.0, sortino=1.0,
        max_drawdown=0.05, win_rate=0.6, num_trades=10,
        starting_capital=1000.0, ending_capital=1000.0 * (1 + tr),
    )
    pm = MetricsPM(
        win_rate_ci95_low=None, win_rate_ci95_high=None,
        mean_return_pct=None, std_return_pct=None,
        sharpe_trade_level=None, sharpe_equity_curve=None,
        brier_strategy=None, brier_crowd=None, brier_skill_score=None,
        mean_edge=None,
    )
    return Metrics(standard=std, pm=pm)


def _make_leg(
    equity_curve: list[EquityPoint],
    result_hash: str = "abc123",
    engine: str = "pancake",
    engine_version: str = "0.9.0",
    ok: bool = True,
) -> BacktestResult:
    verdict = _ok_verdict() if ok else _blocked_verdict()
    # Create a minimal Trade mock with decision_time attribute for regime.
    return BacktestResult(
        engine=engine,
        engine_version=engine_version,
        engine_mode="backtest",
        compiled_spec_hash="csh",
        schema_sha256="ssh",
        rows_sha256="rsh",
        config_hash="confh",
        result_hash=result_hash,
        metrics=_make_metrics(),
        equity_curve=equity_curve,
        drawdown_curve=[DrawdownPoint(t=p.t, drawdown=0.0) for p in equity_curve],
        monthly_returns=[],
        trades=[],
        warnings=[],
        validation=verdict,
    )


# ---------------------------------------------------------------------------
# Section 1: Happy path — hand-calculated 2-leg portfolio
# ---------------------------------------------------------------------------

# Leg A: equity goes 1.0 → 1.1 → 1.2 at t=0, 86400, 172800
# Leg B: equity goes 2.0 → 2.4 → 2.2 at t=0, 86400, 172800
#
# After normalisation:
#   Leg A normalised: 1.0 → 1.1 → 1.2
#   Leg B normalised: 1.0 → 1.2 → 1.1
#
# Weights: 0.6 (A) + 0.4 (B)
#
# Joint at t=0:    0.6 * 1.0 + 0.4 * 1.0 = 1.0
# Joint at t=86400: 0.6 * 1.1 + 0.4 * 1.2 = 0.66 + 0.48 = 1.14
# Joint at t=172800: 0.6 * 1.2 + 0.4 * 1.1 = 0.72 + 0.44 = 1.16
#
# Portfolio total_return = 1.16 / 1.0 - 1 = 0.16
#
# Daily returns of joint: [1.14/1.0 - 1, 1.16/1.14 - 1] = [0.14, 0.01754...]
#
# Pearson correlation of leg daily returns:
#   Leg A daily returns: [(1.1-1.0)/1.0, (1.2-1.1)/1.1] = [0.1, 0.090909...]
#   Leg B daily returns: [(2.4-2.0)/2.0, (2.2-2.4)/2.4] = [0.2, -0.083333...]
#   mean_A = (0.1 + 0.090909) / 2 = 0.095455
#   mean_B = (0.2 + (-0.083333)) / 2 = 0.058333
#   cov = (0.1-0.095455)*(0.2-0.058333) + (0.090909-0.095455)*(-0.083333-0.058333)
#       = (0.004545)*(0.141667) + (-0.004545)*(-0.141667)
#       = 0.000644 + 0.000644 = 0.001288
#   var_A = (0.1-0.095455)^2 + (0.090909-0.095455)^2
#         = 0.004545^2 + 0.004545^2 = 2 * 2.066e-5 = 4.132e-5
#   var_B = (0.2-0.058333)^2 + (-0.083333-0.058333)^2
#         = 0.141667^2 + 0.141667^2 = 2 * 0.020069 = 0.040138
#   r = 0.001288 / sqrt(4.132e-5 * 0.040138)
#     = 0.001288 / sqrt(1.658e-6)
#     = 0.001288 / 0.001288 = 1.0  ← both series move +/+ then +/- perfectly anti-sym
#
# Actually let's recompute carefully:
#   cov = (0.1 - 0.095455)*(0.2 - 0.058333) + (0.090909 - 0.095455)*(-0.083333 - 0.058333)
#       = 0.004545 * 0.141667 + (-0.004545)*(-0.141667)
#       = 0.004545*0.141667 + 0.004545*0.141667
#       = 2 * 0.004545 * 0.141667 = 2 * 0.0006438 = 0.001288
#   var_A = (0.004545)^2 + (0.004545)^2 = 2 * 0.00002066 = 0.00004132
#   var_B = (0.141667)^2 + (-0.141667)^2 = 2 * 0.020069 = 0.040138
#   denom = sqrt(0.00004132 * 0.040138) = sqrt(0.000001659) = 0.001288
#   r = 0.001288 / 0.001288 ≈ 1.0  (perfect positive correlation — both move same sign)

DAY = 86_400
_LEG_A_CURVE = [
    EquityPoint(t=0, equity=1.0),
    EquityPoint(t=DAY, equity=1.1),
    EquityPoint(t=2 * DAY, equity=1.2),
]
_LEG_B_CURVE = [
    EquityPoint(t=0, equity=2.0),
    EquityPoint(t=DAY, equity=2.4),
    EquityPoint(t=2 * DAY, equity=2.2),
]


def _build_two_legs() -> tuple[list[BacktestResult], list[float]]:
    leg_a = _make_leg(_LEG_A_CURVE, result_hash="hash_a")
    leg_b = _make_leg(_LEG_B_CURVE, result_hash="hash_b")
    return [leg_a, leg_b], [0.6, 0.4]


def test_joint_equity_at_t0() -> None:
    """Joint equity at t=0 is 1.0 (weighted sum of normalised start values)."""
    legs, weights = _build_two_legs()
    result = compute_portfolio(legs, weights)
    t0 = next(p for p in result.joint_equity_curve if p["t"] == 0)
    assert abs(t0["equity"] - 1.0) < 1e-12


def test_joint_equity_at_t1() -> None:
    """Joint at t=86400: 0.6*1.1 + 0.4*1.2 = 1.14."""
    legs, weights = _build_two_legs()
    result = compute_portfolio(legs, weights)
    t1 = next(p for p in result.joint_equity_curve if p["t"] == DAY)
    assert abs(t1["equity"] - 1.14) < 1e-12, f"expected 1.14, got {t1['equity']}"


def test_joint_equity_at_t2() -> None:
    """Joint at t=172800: 0.6*1.2 + 0.4*1.1 = 1.16."""
    legs, weights = _build_two_legs()
    result = compute_portfolio(legs, weights)
    t2 = next(p for p in result.joint_equity_curve if p["t"] == 2 * DAY)
    assert abs(t2["equity"] - 1.16) < 1e-12, f"expected 1.16, got {t2['equity']}"


def test_portfolio_total_return() -> None:
    """total_return = 1.16/1.0 - 1 = 0.16."""
    legs, weights = _build_two_legs()
    result = compute_portfolio(legs, weights)
    assert abs(result.metrics["total_return"] - 0.16) < 1e-10


def test_portfolio_num_legs() -> None:
    legs, weights = _build_two_legs()
    result = compute_portfolio(legs, weights)
    assert result.metrics["num_legs"] == 2


def test_per_leg_metrics_present() -> None:
    legs, weights = _build_two_legs()
    result = compute_portfolio(legs, weights)
    assert len(result.metrics["per_leg"]) == 2
    assert result.metrics["per_leg"][0]["weight"] == 0.6
    assert result.metrics["per_leg"][1]["weight"] == 0.4


def test_per_leg_result_hash_echoed() -> None:
    legs, weights = _build_two_legs()
    result = compute_portfolio(legs, weights)
    assert result.leg_result_hashes == ["hash_a", "hash_b"]


def test_correlation_matrix_shape() -> None:
    """2×2 correlation matrix, diagonal = 1.0."""
    legs, weights = _build_two_legs()
    result = compute_portfolio(legs, weights)
    assert len(result.correlation_matrix) == 2
    assert len(result.correlation_matrix[0]) == 2
    assert result.correlation_matrix[0][0] == 1.0
    assert result.correlation_matrix[1][1] == 1.0


def test_correlation_symmetric() -> None:
    """r[0][1] == r[1][0]."""
    legs, weights = _build_two_legs()
    result = compute_portfolio(legs, weights)
    r01 = result.correlation_matrix[0][1]
    r10 = result.correlation_matrix[1][0]
    assert r01 == r10


def test_correlation_hand_calc() -> None:
    """Pearson r ≈ 1.0 for this specific pair (see comment above).

    Leg A daily returns: +0.1, +0.090909...
    Leg B daily returns: +0.2, -0.083333...
    Both start positive then opposite sign → r should NOT be 1.0.
    Let's compute carefully: the sign of the second return differs.
    mean_A ≈ 0.09545, mean_B ≈ 0.05833
    deviations A: [0.00455, -0.00455]  B: [0.14167, -0.14167]
    cov = 0.00455*0.14167 + (-0.00455)*(-0.14167) = 2*(0.000644) = 0.001288 > 0
    var_A ≈ 4.13e-5, var_B ≈ 4.01e-2
    r = 0.001288 / sqrt(4.13e-5 * 4.01e-2) ≈ 0.001288 / 0.001288 = 1.0

    Both deviations are equal in magnitude and sign, so r = 1.0 exactly.
    """
    legs, weights = _build_two_legs()
    result = compute_portfolio(legs, weights)
    r = result.correlation_matrix[0][1]
    assert r is not None
    assert abs(r - 1.0) < 1e-9, f"expected r≈1.0, got {r}"


# ---------------------------------------------------------------------------
# Section 2: Weights validation
# ---------------------------------------------------------------------------


def test_weights_sum_exactly_one_pass() -> None:
    legs, _ = _build_two_legs()
    compute_portfolio(legs, [0.5, 0.5])  # should not raise


def test_weights_sum_not_one_raises() -> None:
    legs, _ = _build_two_legs()
    with pytest.raises(PortfolioError, match="sum"):
        compute_portfolio(legs, [0.5, 0.6])


def test_weights_sum_tolerance_edge() -> None:
    """Sum deviating by >1e-12 raises; exactly 1.0 passes."""
    legs, _ = _build_two_legs()
    # 0.5 + 0.5 = exact 1.0 via fsum
    compute_portfolio(legs, [0.5, 0.5])  # ok

    # Deviation just above 1e-12: should raise.
    eps = 2e-12
    with pytest.raises(PortfolioError, match="sum"):
        compute_portfolio(legs, [0.5, 0.5 + eps])


def test_negative_weight_raises() -> None:
    legs, _ = _build_two_legs()
    with pytest.raises(PortfolioError, match="positive"):
        compute_portfolio(legs, [-0.1, 1.1])


def test_zero_weight_raises() -> None:
    legs, _ = _build_two_legs()
    with pytest.raises(PortfolioError, match="positive"):
        compute_portfolio(legs, [0.0, 1.0])


def test_weights_length_mismatch_raises() -> None:
    legs, _ = _build_two_legs()
    with pytest.raises(PortfolioError, match="length"):
        compute_portfolio(legs, [0.5, 0.3, 0.2])


def test_nan_weight_raises() -> None:
    legs, _ = _build_two_legs()
    with pytest.raises(PortfolioError):
        compute_portfolio(legs, [float("nan"), 0.5])


def test_inf_weight_raises() -> None:
    legs, _ = _build_two_legs()
    with pytest.raises(PortfolioError):
        compute_portfolio(legs, [float("inf"), 0.5])


# ---------------------------------------------------------------------------
# Section 3: Legs validation
# ---------------------------------------------------------------------------


def test_fewer_than_two_legs_raises() -> None:
    leg = _make_leg(_LEG_A_CURVE)
    with pytest.raises(PortfolioError, match="≥2"):
        compute_portfolio([leg], [1.0])


def test_blocked_leg_raises() -> None:
    leg_a = _make_leg(_LEG_A_CURVE)
    leg_b = _make_leg(_LEG_B_CURVE, ok=False)
    with pytest.raises(PortfolioError, match="blocked"):
        compute_portfolio([leg_a, leg_b], [0.5, 0.5])


def test_empty_equity_curve_raises() -> None:
    leg_a = _make_leg(_LEG_A_CURVE)
    leg_b = _make_leg([])  # empty curve
    with pytest.raises(PortfolioError, match="empty equity_curve"):
        compute_portfolio([leg_a, leg_b], [0.5, 0.5])


def test_engine_identity_mismatch_raises() -> None:
    leg_a = _make_leg(_LEG_A_CURVE, engine="pancake", engine_version="0.9.0")
    leg_b = _make_leg(_LEG_B_CURVE, engine="pancake", engine_version="0.8.0")
    with pytest.raises(PortfolioError, match="engine identity"):
        compute_portfolio([leg_a, leg_b], [0.5, 0.5])


def test_engine_name_mismatch_raises() -> None:
    leg_a = _make_leg(_LEG_A_CURVE, engine="pancake")
    leg_b = _make_leg(_LEG_B_CURVE, engine="other")
    with pytest.raises(PortfolioError, match="engine identity"):
        compute_portfolio([leg_a, leg_b], [0.5, 0.5])


# ---------------------------------------------------------------------------
# Section 4: Determinism — 3 identical calls produce identical portfolio_hash
# ---------------------------------------------------------------------------


def test_determinism_three_runs() -> None:
    legs, weights = _build_two_legs()
    r1 = compute_portfolio(legs, weights)
    r2 = compute_portfolio(legs, weights)
    r3 = compute_portfolio(legs, weights)
    assert r1.portfolio_hash == r2.portfolio_hash == r3.portfolio_hash


# ---------------------------------------------------------------------------
# Section 5: Hash discipline
# ---------------------------------------------------------------------------


def test_portfolio_hash_in_to_dict() -> None:
    legs, weights = _build_two_legs()
    result = compute_portfolio(legs, weights)
    d = result.to_dict()
    assert "portfolio_hash" in d
    assert d["portfolio_hash"] == result.portfolio_hash


def test_different_weights_different_hash() -> None:
    """Changing weights changes portfolio_hash."""
    legs, _ = _build_two_legs()
    r1 = compute_portfolio(legs, [0.6, 0.4])
    r2 = compute_portfolio(legs, [0.5, 0.5])
    assert r1.portfolio_hash != r2.portfolio_hash


def test_different_leg_hashes_different_portfolio_hash() -> None:
    """Changing a leg's result_hash propagates to portfolio_hash."""
    leg_a1 = _make_leg(_LEG_A_CURVE, result_hash="hash_a_v1")
    leg_a2 = _make_leg(_LEG_A_CURVE, result_hash="hash_a_v2")
    leg_b = _make_leg(_LEG_B_CURVE, result_hash="hash_b")
    r1 = compute_portfolio([leg_a1, leg_b], [0.5, 0.5])
    r2 = compute_portfolio([leg_a2, leg_b], [0.5, 0.5])
    assert r1.portfolio_hash != r2.portfolio_hash


# ---------------------------------------------------------------------------
# Section 6: Correlation degenerate cases
# ---------------------------------------------------------------------------


def test_correlation_zero_variance_leg_returns_none() -> None:
    """A leg with constant equity (zero variance in returns) → None correlation."""
    flat_curve = [
        EquityPoint(t=0, equity=1.0),
        EquityPoint(t=DAY, equity=1.0),
        EquityPoint(t=2 * DAY, equity=1.0),
    ]
    leg_a = _make_leg(_LEG_A_CURVE)
    leg_b = _make_leg(flat_curve)
    result = compute_portfolio([leg_a, leg_b], [0.5, 0.5])
    # Off-diagonal should be None (zero variance in leg_b daily returns)
    assert result.correlation_matrix[0][1] is None
    assert result.correlation_matrix[1][0] is None
    # Diagonal stays 1.0
    assert result.correlation_matrix[0][0] == 1.0
    assert result.correlation_matrix[1][1] == 1.0


def test_correlation_single_point_leg_returns_none() -> None:
    """A leg with only 1 equity point has no daily returns → None correlation."""
    single_curve = [EquityPoint(t=0, equity=1.0)]
    leg_a = _make_leg(_LEG_A_CURVE)
    leg_b = _make_leg(single_curve)
    result = compute_portfolio([leg_a, leg_b], [0.5, 0.5])
    assert result.correlation_matrix[0][1] is None


# ---------------------------------------------------------------------------
# Section 7: to_dict() round-trip shape
# ---------------------------------------------------------------------------


def test_to_dict_keys() -> None:
    legs, weights = _build_two_legs()
    result = compute_portfolio(legs, weights)
    d = result.to_dict()
    required_keys = {
        "format_version", "engine", "engine_version",
        "leg_result_hashes", "weights", "metrics",
        "correlation_matrix", "joint_equity_curve", "portfolio_hash",
    }
    assert required_keys.issubset(set(d.keys()))


def test_format_version() -> None:
    legs, weights = _build_two_legs()
    result = compute_portfolio(legs, weights)
    assert result.format_version == "portfolio/1"
    assert result.to_dict()["format_version"] == "portfolio/1"


def test_joint_equity_curve_dicts_have_t_and_equity() -> None:
    legs, weights = _build_two_legs()
    result = compute_portfolio(legs, weights)
    for pt in result.joint_equity_curve:
        assert "t" in pt
        assert "equity" in pt


def test_three_legs_portfolio() -> None:
    """Sanity: three equal-weight legs run without error."""
    leg_c_curve = [
        EquityPoint(t=0, equity=0.5),
        EquityPoint(t=DAY, equity=0.55),
        EquityPoint(t=2 * DAY, equity=0.6),
    ]
    leg_a = _make_leg(_LEG_A_CURVE, result_hash="ha")
    leg_b = _make_leg(_LEG_B_CURVE, result_hash="hb")
    leg_c = _make_leg(leg_c_curve, result_hash="hc")
    result = compute_portfolio([leg_a, leg_b, leg_c], [1 / 3, 1 / 3, 1 / 3])
    assert result.metrics["num_legs"] == 3
    assert len(result.correlation_matrix) == 3


def test_metrics_max_drawdown_non_negative() -> None:
    legs, weights = _build_two_legs()
    result = compute_portfolio(legs, weights)
    assert result.metrics["max_drawdown"] >= 0.0


def test_metrics_sharpe_is_number_or_none() -> None:
    legs, weights = _build_two_legs()
    result = compute_portfolio(legs, weights)
    sharpe = result.metrics["sharpe"]
    assert sharpe is None or isinstance(sharpe, float)
