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
    deflated_sharpe_ratio,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    psr_sharpe_hat,
    return_moments,
)
from pancake_engine.metrics.psr import _norm_ppf  # probit, cross-checked vs scipy

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


# --- inverse normal CDF (probit) -------------------------------------------


def test_norm_ppf_matches_scipy() -> None:
    for p in (0.001, 0.01, 0.025, 0.1, 0.5, 0.9, 0.95, 0.975, 0.99, 0.999):
        assert _norm_ppf(p) == pytest.approx(float(stats.norm.ppf(p)), rel=1e-10, abs=1e-10)


# --- Deflated Sharpe Ratio --------------------------------------------------

TRIAL_SHARPES = [0.02, 0.05, -0.01, 0.08, 0.03, 0.06, 0.00, 0.04, 0.07, -0.02]  # per-period


def _ref_dsr(returns: list[float], trial_sharpes: list[float]) -> float:
    sr = np.asarray(trial_sharpes, dtype=float)
    n = sr.size
    v = sr.var(ddof=1)
    g = 0.5772156649015329
    emax = math.sqrt(v) * (
        (1 - g) * stats.norm.ppf(1 - 1 / n) + g * stats.norm.ppf(1 - 1 / (n * math.e))
    )
    return _ref_psr(returns, emax)  # fully independent (scipy throughout)


def test_dsr_matches_independent_reference() -> None:
    assert deflated_sharpe_ratio(RETURNS, TRIAL_SHARPES) == pytest.approx(
        _ref_dsr(RETURNS, TRIAL_SHARPES), rel=1e-9, abs=1e-12
    )


def test_dsr_is_below_psr_at_zero() -> None:
    # Deflation raises the benchmark above 0 → DSR <= PSR(SR* = 0).
    assert deflated_sharpe_ratio(RETURNS, TRIAL_SHARPES) <= probabilistic_sharpe_ratio(RETURNS, 0.0)


def test_dsr_decreases_with_more_trials() -> None:
    # Same per-trial variance, more configurations → higher expected-max bar → lower DSR.
    small = [0.05, 0.10, 0.15]
    large = small * 20  # ~same variance, N=60
    assert deflated_sharpe_ratio(RETURNS, large) < deflated_sharpe_ratio(RETURNS, small)


def test_dsr_none_when_undefined() -> None:
    assert deflated_sharpe_ratio(RETURNS, [0.05]) is None              # <2 trials
    assert deflated_sharpe_ratio(RETURNS, [0.05, 0.05, 0.05]) is None  # zero trial variance


# --- Minimum Track Record Length -------------------------------------------


def _ref_mintrl(returns: list[float], conf: float = 0.95, srb: float = 0.0) -> float:
    a = np.asarray(returns, dtype=float)
    sr = a.mean() / a.std(ddof=1)
    skew = stats.skew(a, bias=True)
    kurt = stats.kurtosis(a, fisher=False, bias=True)
    var_term = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    return 1 + var_term * (stats.norm.ppf(conf) / (sr - srb)) ** 2


def test_mintrl_matches_independent_reference() -> None:
    assert min_track_record_length(RETURNS, 0.95) == pytest.approx(_ref_mintrl(RETURNS, 0.95), rel=1e-9)


def test_mintrl_increases_with_confidence() -> None:
    assert min_track_record_length(RETURNS, 0.99) > min_track_record_length(RETURNS, 0.90)


def test_mintrl_none_when_sharpe_below_benchmark() -> None:
    losing = [-0.02, -0.01, -0.03, -0.015, -0.008, -0.02]  # negative mean → SR < 0
    assert min_track_record_length(losing, 0.95) is None
    assert min_track_record_length([0.01], 0.95) is None  # moments undefined (n<2)
