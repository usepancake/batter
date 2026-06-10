"""Probabilistic Sharpe Ratio (Bailey & López de Prado 2012).

Cross-checked against a fully independent SciPy path (dev-only; the engine itself
uses math.erf and has no SciPy runtime dependency).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from pancake_engine.metrics.psr import (
    probabilistic_sharpe_ratio,
    psr_sharpe_hat,
    return_moments,
)

# A non-symmetric series so skew/kurtosis genuinely move the result.
RETURNS = [0.01, -0.02, 0.03, 0.005, -0.01, 0.02, 0.015, -0.008, 0.012, 0.0, 0.04, -0.015]


def _ref_psr(returns: list[float], sr_benchmark: float = 0.0) -> float:
    """Independent reference: SciPy moments + SciPy normal CDF."""
    a = np.asarray(returns, dtype=float)
    n = a.size
    sr = a.mean() / a.std(ddof=1)
    skew = stats.skew(a, bias=True)                       # population g1
    kurt = stats.kurtosis(a, fisher=False, bias=True)     # population, non-excess
    var_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(var_term)
    return float(stats.norm.cdf(z))


def test_psr_matches_independent_scipy_reference() -> None:
    assert probabilistic_sharpe_ratio(RETURNS) == pytest.approx(_ref_psr(RETURNS), rel=1e-9, abs=1e-12)
    # also at a non-zero benchmark
    assert probabilistic_sharpe_ratio(RETURNS, sr_benchmark=0.1) == pytest.approx(
        _ref_psr(RETURNS, 0.1), rel=1e-9, abs=1e-12
    )


def test_moments_match_scipy() -> None:
    sr_hat, skew, kurt, n = return_moments(RETURNS)
    a = np.asarray(RETURNS, dtype=float)
    assert n == len(RETURNS)
    assert sr_hat == pytest.approx(a.mean() / a.std(ddof=1), rel=1e-12)
    assert skew == pytest.approx(stats.skew(a, bias=True), rel=1e-12)
    assert kurt == pytest.approx(stats.kurtosis(a, fisher=False, bias=True), rel=1e-12)


def test_psr_half_at_own_sharpe() -> None:
    # benchmark == observed per-period Sharpe → z = 0 → PSR = 0.5 exactly.
    sr = psr_sharpe_hat(RETURNS)
    assert probabilistic_sharpe_ratio(RETURNS, sr_benchmark=sr) == pytest.approx(0.5, abs=1e-12)


def test_psr_increases_with_sample_length() -> None:
    # Tiling preserves all moments but multiplies n → more confidence (sr_hat > 0).
    assert psr_sharpe_hat(RETURNS) > 0
    short = probabilistic_sharpe_ratio(RETURNS)
    longer = probabilistic_sharpe_ratio(RETURNS * 4)
    assert longer > short


def test_psr_in_unit_interval() -> None:
    for srb in (-0.5, 0.0, 0.05, 0.5):
        psr = probabilistic_sharpe_ratio(RETURNS, sr_benchmark=srb)
        assert 0.0 <= psr <= 1.0


def test_psr_none_when_undefined() -> None:
    assert probabilistic_sharpe_ratio([]) is None
    assert probabilistic_sharpe_ratio([0.01]) is None          # n < 2
    assert probabilistic_sharpe_ratio([0.02, 0.02, 0.02]) is None  # zero variance
    assert return_moments([0.02, 0.02]) is None                # zero variance
