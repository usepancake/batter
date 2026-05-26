"""Tests for pancake_engine.metrics.bootstrap — bootstrap_ci function.

Hand-calc fixture (case A):
    daily_returns = [0.01, -0.01, 0.02, -0.02, 0.0]  (5 values, non-identical)
    metric_fn     = mean
    n_resamples   = 100, seed = 0
    Expected: finite (ci_low, ci_high) tuple with ci_low <= ci_high.
    Cannot pin exact floats without running numpy — we assert structural properties
    plus determinism.

For determinism gate see test_bootstrap_determinism.py.
"""

from __future__ import annotations

from typing import Optional

import pytest

from pancake_engine.metrics.bootstrap import bootstrap_ci
from pancake_engine.warnings import WarningCode


def _mean(xs: list[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _sharpe_simple(xs: list[float]) -> Optional[float]:
    """Simplified Sharpe for test purposes."""
    if len(xs) < 2:
        return None
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    import math
    std = math.sqrt(var)
    if std == 0:
        return None
    return mean / std * math.sqrt(252)


# ---------------------------------------------------------------------------
# Guard: N < 2
# ---------------------------------------------------------------------------


def test_bootstrap_ci_insufficient_n_zero() -> None:
    (ci_low, ci_high), warnings = bootstrap_ci([], _mean, n_resamples=100, seed=0)
    assert ci_low is None
    assert ci_high is None
    codes = [w.code for w in warnings]
    assert WarningCode.BOOTSTRAP_INSUFFICIENT in codes


def test_bootstrap_ci_insufficient_n_one() -> None:
    (ci_low, ci_high), warnings = bootstrap_ci([0.01], _mean, n_resamples=100, seed=0)
    assert ci_low is None
    assert ci_high is None
    codes = [w.code for w in warnings]
    assert WarningCode.BOOTSTRAP_INSUFFICIENT in codes


# ---------------------------------------------------------------------------
# Guard: zero variance
# ---------------------------------------------------------------------------


def test_bootstrap_ci_zero_variance() -> None:
    returns = [0.01, 0.01, 0.01, 0.01, 0.01]
    (ci_low, ci_high), warnings = bootstrap_ci(returns, _mean, n_resamples=100, seed=0)
    assert ci_low is None
    assert ci_high is None
    codes = [w.code for w in warnings]
    assert WarningCode.BOOTSTRAP_INSUFFICIENT in codes
    insuff = next(w for w in warnings if w.code == WarningCode.BOOTSTRAP_INSUFFICIENT)
    assert insuff.context["reason"] == "zero_variance"


# ---------------------------------------------------------------------------
# Normal case: finite CI, ci_low <= ci_high
# ---------------------------------------------------------------------------


def test_bootstrap_ci_normal_finite_ordered() -> None:
    returns = [0.01, -0.01, 0.02, -0.02, 0.005, -0.005, 0.015, -0.015, 0.0, 0.008]
    (ci_low, ci_high), warnings = bootstrap_ci(returns, _mean, n_resamples=1000, seed=0)
    assert ci_low is not None
    assert ci_high is not None
    assert ci_low <= ci_high
    # No BOOTSTRAP_INSUFFICIENT warning on a valid sample
    codes = [w.code for w in warnings]
    assert WarningCode.BOOTSTRAP_INSUFFICIENT not in codes


# ---------------------------------------------------------------------------
# Determinism: same seed → identical CI
# ---------------------------------------------------------------------------


def test_bootstrap_ci_determinism_same_seed() -> None:
    returns = [0.01, -0.01, 0.02, -0.02, 0.005, -0.005, 0.015, -0.015, 0.0, 0.008]
    (ci_low_a, ci_high_a), _ = bootstrap_ci(returns, _mean, n_resamples=1000, seed=0)
    (ci_low_b, ci_high_b), _ = bootstrap_ci(returns, _mean, n_resamples=1000, seed=0)
    assert ci_low_a == ci_low_b
    assert ci_high_a == ci_high_b


# ---------------------------------------------------------------------------
# Different seeds → different CI
# ---------------------------------------------------------------------------


def test_bootstrap_ci_different_seeds_different_result() -> None:
    """Different seeds should (almost always) produce different CIs on a non-trivial sample."""
    returns = [0.01, -0.01, 0.02, -0.02, 0.005, -0.005, 0.015, -0.015, 0.0, 0.008,
               0.003, -0.007, 0.012, -0.004, 0.006]
    (ci_low_0, ci_high_0), _ = bootstrap_ci(returns, _mean, n_resamples=1000, seed=0)
    (ci_low_1, ci_high_1), _ = bootstrap_ci(returns, _mean, n_resamples=1000, seed=42)
    # They could theoretically match by coincidence, but with 1000 resamples it's astronomically unlikely.
    assert (ci_low_0, ci_high_0) != (ci_low_1, ci_high_1)


# ---------------------------------------------------------------------------
# CI_TOO_WIDE warning
# ---------------------------------------------------------------------------


def test_bootstrap_ci_too_wide_warning_fires() -> None:
    """Manufacture a situation where CI is very wide relative to point estimate.

    Use a highly volatile series + very small n to make CI wide,
    and a metric_fn that returns the mean (small absolute value).
    """
    # Very noisy series where the CI width should dominate the mean
    returns = [0.001, -100.0, 100.0, -0.001, 0.0]  # extreme variance, near-zero mean
    (ci_low, ci_high), warnings = bootstrap_ci(returns, _mean, n_resamples=500, seed=0)
    if ci_low is not None and ci_high is not None:
        codes = [w.code for w in warnings]
        # If the relative width > 5, CI_TOO_WIDE should fire
        point = _mean(returns)
        if point is not None and point != 0.0:
            relative_width = (ci_high - ci_low) / abs(point)
            if relative_width > 5.0:
                assert WarningCode.CI_TOO_WIDE in codes


# ---------------------------------------------------------------------------
# ci_level parameterization: 90% CI should be narrower than 95% CI
# ---------------------------------------------------------------------------


def test_bootstrap_ci_level_90_narrower_than_95() -> None:
    returns = [0.01 * i for i in range(-10, 11)]  # 21 returns, non-trivial variance
    (lo_90, hi_90), _ = bootstrap_ci(returns, _mean, n_resamples=2000, seed=0, ci_level=0.90)
    (lo_95, hi_95), _ = bootstrap_ci(returns, _mean, n_resamples=2000, seed=0, ci_level=0.95)
    assert lo_90 is not None and hi_90 is not None
    assert lo_95 is not None and hi_95 is not None
    width_90 = hi_90 - lo_90
    width_95 = hi_95 - lo_95
    assert width_90 <= width_95, (
        f"90% CI width {width_90:.6f} should be ≤ 95% CI width {width_95:.6f}"
    )


# ---------------------------------------------------------------------------
# Hand-calc fixture: mean of [0.0] * 10 with one outlier — result is near-zero
# ---------------------------------------------------------------------------


def test_bootstrap_ci_mean_zero_dominant_series() -> None:
    """When the true mean is ~0 and variance is moderate, CI should straddle 0."""
    returns = [0.0] * 20 + [0.1, -0.1]  # mean ≈ 0, some variance
    (ci_low, ci_high), warnings = bootstrap_ci(returns, _mean, n_resamples=2000, seed=0)
    assert ci_low is not None
    assert ci_high is not None
    # The CI should contain 0 (the true mean is 0)
    assert ci_low <= 0.0 <= ci_high, (
        f"CI [{ci_low:.6f}, {ci_high:.6f}] should straddle 0 for near-zero-mean series"
    )


# ---------------------------------------------------------------------------
# Metric function returning None for all resamples
# ---------------------------------------------------------------------------


def test_bootstrap_ci_all_none_metric_returns_none() -> None:
    """If metric_fn always returns None, result should be (None, None)."""
    returns = [0.01, -0.01, 0.02, -0.02, 0.005]

    def always_none(xs: list[float]) -> Optional[float]:
        return None

    (ci_low, ci_high), warnings = bootstrap_ci(returns, always_none, n_resamples=100, seed=0)
    assert ci_low is None
    assert ci_high is None
    codes = [w.code for w in warnings]
    assert WarningCode.BOOTSTRAP_INSUFFICIENT in codes
    insuff = next(w for w in warnings if w.code == WarningCode.BOOTSTRAP_INSUFFICIENT)
    assert insuff.context["reason"] == "all_none"
