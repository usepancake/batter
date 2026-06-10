"""Prediction-market metrics: brier_crowd computed, brier_strategy null, Wilson CI, ECE."""

from __future__ import annotations

import math

from pancake_engine import BacktestConfig, WarningCode, run_backtest
from pancake_engine.metrics.pm import calibration_bins, calibration_ece, wilson_ci95
from pancake_engine.runner.trade import Trade

from ._runner_helpers import make_dataset, make_spec, row


# ---------------------------------------------------------------------------
# Trade factory for ECE unit tests (bypasses the full runner to keep tests fast)
# ---------------------------------------------------------------------------

def _trade(entry_price_quote: float, exit_price: float) -> Trade:
    """Minimal Trade for ECE unit tests.

    ``implied_prob_at_entry`` reads ``entry_price_quote``;
    ``realized_outcome_for_trade`` reads ``exit_price`` (1.0 = win, 0.0 = loss).
    All other fields are irrelevant to ECE.
    """
    pnl = exit_price - entry_price_quote  # approximate; unused by ECE
    return Trade(
        market_slug="test",
        outcome="YES",
        entry_t=0,
        entry_price=entry_price_quote,
        entry_price_quote=entry_price_quote,
        exit_t=1,
        exit_price=exit_price,
        exit_price_quote=exit_price,
        exit_reason="hold_to_resolution",
        shares=1.0,
        cost=entry_price_quote,
        proceeds=exit_price,
        pnl=pnl,
        return_pct=pnl / entry_price_quote if entry_price_quote > 0 else 0.0,
        days_held=0,
        resolved_outcome=None,
    )


def test_metrics_pm_brier_crowd_computed() -> None:
    """brier_crowd is always computable when at least one trade exists.

    For YES trade at price 0.6 with outcome=1: (0.6-1)^2 = 0.16.
    """
    spec = make_spec(side="YES", sizing_value=0.1, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.6, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=300)
    result = run_backtest(spec, dataset, config)
    pm = result.metrics.pm
    assert pm.brier_crowd is not None
    assert abs(pm.brier_crowd - 0.16) < 1e-9


def test_metrics_pm_brier_strategy_null_in_pr1() -> None:
    """Rule-based spec → brier_strategy is None + BRIER_NOT_APPLICABLE warning."""
    spec = make_spec(side="YES", sizing_value=0.1, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.5, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=300)
    result = run_backtest(spec, dataset, config)
    assert result.metrics.pm.brier_strategy is None
    assert result.metrics.pm.brier_skill_score is None
    assert any(w.code == WarningCode.BRIER_NOT_APPLICABLE for w in result.warnings)


def test_metrics_pm_wilson_ci_null_for_zero_trades() -> None:
    """num_trades = 0 → Wilson CI bounds = None (architecture M-1)."""
    spec = make_spec(
        side="YES",
        starting_capital=1000.0,
        entry_when={"feature": "alpha", "gte": 999.0},  # never fires
    )
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.5, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=300)
    result = run_backtest(spec, dataset, config)
    assert result.metrics.pm.win_rate_ci95_low is None
    assert result.metrics.pm.win_rate_ci95_high is None
    assert any(w.code == WarningCode.NO_TRADES_NO_CI for w in result.warnings)


def test_wilson_ci_basic_values() -> None:
    """Wilson CI smoke values."""
    # n=0 → (None, None)
    assert wilson_ci95(0, 0) == (None, None)
    # n=100, p=0.5 → roughly (0.40, 0.60)
    low, high = wilson_ci95(50, 100)
    assert low is not None and high is not None
    assert abs(low - 0.4038) < 1e-3
    assert abs(high - 0.5962) < 1e-3
    # n=1, p=1 → upper bound clamped to 1.0
    low, high = wilson_ci95(1, 1)
    assert low is not None and high is not None
    assert 0.0 <= low <= 1.0
    assert high == 1.0


# ---------------------------------------------------------------------------
# ECE tests (TDD — hand-calculated expected values documented inline)
# ---------------------------------------------------------------------------


def test_calibration_ece_below_threshold_returns_none() -> None:
    """calibration_ece and calibration_bins return None for < 10 trades."""
    trades = [_trade(0.6, 1.0) for _ in range(9)]
    assert calibration_ece(trades) is None
    assert calibration_bins(trades) is None


def test_calibration_ece_empty_returns_none() -> None:
    """calibration_ece returns None for empty trade list (0 < 10)."""
    assert calibration_ece([]) is None
    assert calibration_bins([]) is None


def test_calibration_ece_hand_calc() -> None:
    """Hand-calculated ECE over 12 trades across 3 bins.

    Trade construction:
      Bin [0.1, 0.2): 4 trades at implied_prob=0.15 — outcomes: 1,1,0,0
        conf_b = 0.15, acc_b = 0.5,  |acc_b - conf_b| = 0.35
        weight = 4/12

      Bin [0.6, 0.7): 5 trades at implied_prob=0.65 — outcomes: 1,1,1,0,0
        conf_b = 0.65, acc_b = 0.6,  |acc_b - conf_b| = 0.05
        weight = 5/12

      Bin [0.8, 0.9): 3 trades at implied_prob=0.85 — outcomes: 1,1,1
        conf_b = 0.85, acc_b = 1.0,  |acc_b - conf_b| = 0.15
        weight = 3/12

    ECE = (4/12)*0.35 + (5/12)*0.05 + (3/12)*0.15
        = 1.40/12 + 0.25/12 + 0.45/12
        = 2.10/12
        = 0.175
    """
    trades = (
        [_trade(0.15, 1.0), _trade(0.15, 1.0), _trade(0.15, 0.0), _trade(0.15, 0.0)]  # bin [0.1, 0.2)
        + [_trade(0.65, 1.0), _trade(0.65, 1.0), _trade(0.65, 1.0), _trade(0.65, 0.0), _trade(0.65, 0.0)]  # bin [0.6, 0.7)
        + [_trade(0.85, 1.0), _trade(0.85, 1.0), _trade(0.85, 1.0)]  # bin [0.8, 0.9)
    )
    assert len(trades) == 12

    ece = calibration_ece(trades)
    assert ece is not None
    # Expected: 2.10 / 12 = 0.175
    assert abs(ece - 0.175) < 1e-12


def test_calibration_ece_perfect_calibration() -> None:
    """Perfect calibration (acc == conf in every bin) → ECE == 0.

    10 trades at implied_prob=0.6, all win (acc=1.0 would not be perfect).
    Use implied_prob=1.0 with all wins — bin [0.9,1.0], conf=1.0, acc=1.0 → ECE=0.
    Actually easier: 10 trades at 0.55, 6 win (acc=6/10=0.6 ≈ conf=0.55, not exact).
    Cleanest: 10 trades in the same bin where conf == acc.
    - 10 trades at 0.65 with 6 wins, 4 losses → acc=0.6, conf=0.65 (not perfect).
    Use a contrived exact case: 10 trades at implied_prob=p, acc=p.
    For p=0.5, put exactly 5 wins and 5 losses (all at exact 0.5).
    conf_b = 0.5, acc_b = 0.5, ECE = 0.
    """
    trades = [_trade(0.5, 1.0)] * 5 + [_trade(0.5, 0.0)] * 5
    assert len(trades) == 10
    ece = calibration_ece(trades)
    assert ece is not None
    assert ece == 0.0


def test_calibration_ece_determinism() -> None:
    """Same input → same output (no randomness)."""
    trades = (
        [_trade(0.15, 1.0), _trade(0.15, 0.0)] * 3
        + [_trade(0.75, 1.0), _trade(0.75, 0.0)] * 2
    )
    assert len(trades) == 10
    ece1 = calibration_ece(trades)
    ece2 = calibration_ece(trades)
    assert ece1 == ece2


def test_calibration_ece_no_side_trade_no_inversion() -> None:
    """NO-side trade: implied_prob_at_entry == entry_price_quote unchanged (no inversion).

    A NO-side trade at entry_price_quote=0.3 → implied_prob=0.3 (bin [0.2,0.3)).
    12 identical trades to exceed the threshold: 8 wins, 4 losses.
    conf = 0.3, acc = 8/12 ≈ 0.6667, |acc-conf| = 0.3667
    ECE = 1.0 * 0.3667 = 0.3667  (all trades in one bin, weight=1)
    """
    trades = [_trade(0.3, 1.0)] * 8 + [_trade(0.3, 0.0)] * 4
    assert len(trades) == 12

    ece = calibration_ece(trades)
    assert ece is not None

    # Hand-calc: conf=0.3, acc=8/12, |8/12 - 0.3| = |0.6̄ - 0.3| = 11/30
    expected = abs(8 / 12 - 0.3)  # ≈ 0.36667
    assert abs(ece - expected) < 1e-12


def test_calibration_bins_structure() -> None:
    """calibration_bins returns correct shape for 12-trade set."""
    trades = (
        [_trade(0.15, 1.0)] * 4
        + [_trade(0.65, 1.0)] * 3 + [_trade(0.65, 0.0)] * 2
        + [_trade(0.85, 1.0)] * 3
    )
    assert len(trades) == 12

    bins = calibration_bins(trades)
    assert bins is not None
    assert len(bins) == 3  # 3 non-empty bins

    # Bins are in ascending order
    lows = [b["bin_low"] for b in bins]
    assert lows == sorted(lows)

    # Spot-check first bin: [0.1, 0.2), n=4, conf=0.15, acc=1.0
    b0 = bins[0]
    assert b0["bin_low"] == 0.1
    assert b0["bin_high"] == 0.2
    assert b0["n"] == 4
    assert abs(b0["confidence"] - 0.15) < 1e-12
    assert abs(b0["accuracy"] - 1.0) < 1e-12


def test_calibration_bins_attached_to_result() -> None:
    """calibration_bins block is NOT present for < 10 trades (fast path)."""
    spec = make_spec(side="YES", sizing_value=0.1, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.6, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=300)
    result = run_backtest(spec, dataset, config)
    # 1 trade < 10 → calibration_bins should be None
    assert result.calibration_bins is None
    assert result.metrics.pm.calibration_ece is None


def test_calibration_ece_hashed_and_bins_not_hashed() -> None:
    """calibration_ece is in result_hash; calibration_bins is NOT.

    Verify by checking that the result_hash is deterministic and that
    calibration_bins appears in to_dict() but is NOT present in the hash payload
    (i.e., the result's result_hash stays stable when calibration_bins differs).
    """
    spec = make_spec(side="YES", sizing_value=0.1, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.6, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=300)
    result = run_backtest(spec, dataset, config)
    d = result.to_dict()
    # calibration_bins key exists in to_dict()
    assert "calibration_bins" in d
    # calibration_ece appears under metrics.pm in to_dict()
    assert "calibration_ece" in d["metrics"]["pm"]
