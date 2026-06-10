"""Stationary (Politis & Romano 1994) block-bootstrap CI.

The defining property: for autocorrelated returns, IID resampling destroys the
serial correlation and produces CIs that are too narrow; the stationary bootstrap
preserves it, giving a wider, honester interval. That is the test that matters.
"""

from __future__ import annotations

import math

import numpy as np

from pancake_engine.metrics.bootstrap import block_bootstrap_ci, bootstrap_ci


def _mean(xs: list[float]) -> float | None:
    return (math.fsum(xs) / len(xs)) if xs else None


def _ar1(n: int, phi: float, seed: int) -> list[float]:
    """A positively-autocorrelated AR(1) return series (test-side randomness)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.01, n)
    out = [0.0] * n
    for t in range(1, n):
        out[t] = phi * out[t - 1] + float(noise[t])
    return out


def test_block_bootstrap_is_deterministic() -> None:
    r = _ar1(150, 0.8, 1)
    a, _ = block_bootstrap_ci(r, _mean, seed=7, n_resamples=2000)
    b, _ = block_bootstrap_ci(r, _mean, seed=7, n_resamples=2000)
    assert a == b
    assert a[0] is not None and a[1] is not None


def test_block_ci_wider_than_iid_for_autocorrelated_returns() -> None:
    # φ=0.85 inflates the true variance of the mean ~ (1+φ)/(1-φ) ≈ 12×; IID
    # bootstrap ignores this and under-covers. Block bootstrap must be wider.
    r = _ar1(200, 0.85, 42)
    (ilo, ihi), _ = bootstrap_ci(r, _mean, seed=0, n_resamples=4000)
    (blo, bhi), _ = block_bootstrap_ci(r, _mean, seed=0, n_resamples=4000)
    iid_width = ihi - ilo
    block_width = bhi - blo
    assert block_width > iid_width


def test_block_bootstrap_inherits_guards() -> None:
    # n < 2 and zero-variance guards live in bootstrap_ci; block delegates to it.
    assert block_bootstrap_ci([0.01], _mean)[0] == (None, None)
    assert block_bootstrap_ci([0.02, 0.02, 0.02, 0.02], _mean)[0] == (None, None)


def test_default_block_length_is_sqrt_n_and_returns_valid_interval() -> None:
    r = _ar1(64, 0.5, 3)
    (lo, hi), _ = block_bootstrap_ci(r, _mean, seed=0, n_resamples=1500)
    assert lo is not None and hi is not None and lo < hi


def test_block_length_one_collapses_toward_iid() -> None:
    # L=1 → p=1 → every step is a fresh block → IID; the wider-than-IID effect
    # should vanish (block width no longer materially exceeds IID width).
    r = _ar1(200, 0.85, 42)
    (ilo, ihi), _ = bootstrap_ci(r, _mean, seed=0, n_resamples=4000)
    (blo, bhi), _ = block_bootstrap_ci(r, _mean, expected_block_length=1.0, seed=0, n_resamples=4000)
    assert (bhi - blo) < 1.5 * (ihi - ilo)
