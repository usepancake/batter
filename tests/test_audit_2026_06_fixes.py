"""Regression tests for the 2026-06-04 batter audit fixes (engine 0.6.0).

Each test pins a behavior the audit corrected so it cannot silently regress.
"""

from __future__ import annotations

import random

import pytest

from pancake_engine.metrics.bootstrap import _MAX_RESAMPLES, bootstrap_ci
from pancake_engine.metrics.permutation import (
    _ANNUALIZATION_DAYS,
    _MAX_PERMUTATIONS,
    permutation_p_sharpe,
)
from pancake_engine.metrics.standard import ANNUALIZATION_DAYS, sharpe_ratio


# --- #2 permutation p-value ------------------------------------------------


def test_permutation_p_never_zero_on_strong_signal() -> None:
    rng = random.Random(7)
    strong = [0.10 + rng.gauss(0, 0.001) for _ in range(50)]
    p, _warns = permutation_p_sharpe(strong, n_permutations=1000, seed=0)
    assert p is not None
    assert p > 0.0
    # No permutation beats an overwhelming signal → count=0 → minimum p=1/(n+1).
    assert p == pytest.approx(1.0 / (1000 + 1))


def test_permutation_n_permutations_capped() -> None:
    with pytest.raises(ValueError, match="n_permutations must be in"):
        permutation_p_sharpe([0.01] * 12, n_permutations=_MAX_PERMUTATIONS + 1)
    with pytest.raises(ValueError, match="n_permutations must be in"):
        permutation_p_sharpe([0.01] * 12, n_permutations=0)


def test_permutation_annualization_matches_standard() -> None:
    # The inner-loop Sharpe must annualize identically to the reported Sharpe,
    # or the permutation null is computed under a different statistic.
    assert _ANNUALIZATION_DAYS == ANNUALIZATION_DAYS


# --- #6 bootstrap zero-width CI --------------------------------------------


def test_bootstrap_zero_width_ci_returns_none() -> None:
    # [0.01, -0.01]: every non-degenerate Sharpe resample is exactly 0 → the CI
    # collapses to (0, 0). That must surface as insufficient, not a (0.0, 0.0)
    # that reads like infinite confidence.
    ci, warns = bootstrap_ci([0.01, -0.01], sharpe_ratio, n_resamples=1000, seed=0)
    assert ci == (None, None)
    assert any(w.context.get("reason") == "zero_width_ci" for w in warns)


def test_bootstrap_n_resamples_capped() -> None:
    with pytest.raises(ValueError, match="n_resamples must be in"):
        bootstrap_ci([0.01, -0.02, 0.03], sharpe_ratio, n_resamples=_MAX_RESAMPLES + 1)


# --- determinism guard (Python 3.12 float accumulation) --------------------


def test_metric_mean_uses_fsum() -> None:
    # Durable-hash guard (3a): the float-sum hashed path uses math.fsum
    # (correctly-rounded, identical on 3.11/3.12/3.13+), not builtin sum()
    # (whose accumulation drifts between interpreter versions). Catches a
    # regression back to builtin sum.
    import math

    from pancake_engine.metrics.standard import _mean

    xs = [0.1] * 10 + [0.2] * 10 + [0.3] * 10
    assert _mean(xs) == math.fsum(xs) / len(xs)


def test_sum_float_accumulation_is_312_stable() -> None:
    # Python 3.12 sums homogeneous floats with compensated accumulation
    # (== 0.2 exactly); 3.11 gives 0.20000000000000004, which would change
    # result_hash. The package refuses <3.12 at import; this guards the premise.
    assert sum([0.01] * 20) == 0.2
