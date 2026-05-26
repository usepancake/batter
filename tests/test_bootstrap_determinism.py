"""Determinism gate for Engine 0.4 bootstrap CI.

Load-bearing test: same inputs + seed=0 MUST produce byte-identical (ci_low, ci_high)
tuples across independent calls. A failure here means the RNG is not seeded correctly
or the metric_fn is non-deterministic.

This test is the "cross-platform stability" gate referenced in the directive.
Full cross-platform matrix (macOS / Linux / Windows via GH Actions) is deferred
to PR-B4 when the public repo is created. See docs/math-audit-0.4.md §cross-platform.
"""

from __future__ import annotations

from pancake_engine.metrics.bootstrap import bootstrap_ci
from pancake_engine.metrics.permutation import permutation_p_sharpe
from pancake_engine.metrics.standard import sharpe_ratio, sortino_ratio


# ---------------------------------------------------------------------------
# Canonical test series (fixed, not generated at test time)
# ---------------------------------------------------------------------------

# 30-day return series with a mild positive drift
CANONICAL_RETURNS = [
    0.012, -0.008, 0.015, -0.003, 0.007, 0.009, -0.011, 0.004, 0.018, -0.006,
    0.010, -0.013, 0.002, 0.021, -0.009, 0.005, 0.013, -0.007, 0.008, 0.016,
    -0.005, 0.011, -0.014, 0.003, 0.019, -0.010, 0.006, 0.014, -0.008, 0.009,
]


def test_bootstrap_sharpe_ci_determinism() -> None:
    """Running bootstrap_ci twice with seed=0 produces identical results."""
    (ci_low_a, ci_high_a), _ = bootstrap_ci(
        CANONICAL_RETURNS, sharpe_ratio, n_resamples=10_000, seed=0
    )
    (ci_low_b, ci_high_b), _ = bootstrap_ci(
        CANONICAL_RETURNS, sharpe_ratio, n_resamples=10_000, seed=0
    )
    assert ci_low_a is not None, "Expected finite ci_low for canonical returns"
    assert ci_high_a is not None, "Expected finite ci_high for canonical returns"
    assert ci_low_a == ci_low_b, (
        f"Sharpe CI low not deterministic: {ci_low_a} != {ci_low_b}"
    )
    assert ci_high_a == ci_high_b, (
        f"Sharpe CI high not deterministic: {ci_high_a} != {ci_high_b}"
    )


def test_bootstrap_sortino_ci_determinism() -> None:
    """Running bootstrap_ci for sortino twice with seed=0 produces identical results."""
    (ci_low_a, ci_high_a), _ = bootstrap_ci(
        CANONICAL_RETURNS, sortino_ratio, n_resamples=10_000, seed=0
    )
    (ci_low_b, ci_high_b), _ = bootstrap_ci(
        CANONICAL_RETURNS, sortino_ratio, n_resamples=10_000, seed=0
    )
    assert ci_low_a is not None
    assert ci_high_a is not None
    assert ci_low_a == ci_low_b
    assert ci_high_a == ci_high_b


def test_permutation_p_sharpe_determinism() -> None:
    """Running permutation_p_sharpe twice with seed=0 produces identical p-value."""
    p_a, _ = permutation_p_sharpe(CANONICAL_RETURNS, n_permutations=10_000, seed=0)
    p_b, _ = permutation_p_sharpe(CANONICAL_RETURNS, n_permutations=10_000, seed=0)
    assert p_a is not None, "Expected p-value for canonical returns (N=30 ≥ 10)"
    assert p_a == p_b, f"Permutation p-value not deterministic: {p_a} != {p_b}"


def test_bootstrap_ci_result_sanity_canonical() -> None:
    """Structural sanity for the canonical series with 10k resamples.

    Not a pinned value (would need to embed exact floats). Asserts:
    - ci_low < ci_high (valid interval)
    - Both values are finite
    - The 95% CI contains the observed Sharpe (not guaranteed statistically,
      but very likely for 10k resamples on a 30-sample series)
    """
    import math

    observed_sharpe = sharpe_ratio(CANONICAL_RETURNS)
    (ci_low, ci_high), _ = bootstrap_ci(
        CANONICAL_RETURNS, sharpe_ratio, n_resamples=10_000, seed=0
    )
    assert ci_low is not None and ci_high is not None
    assert math.isfinite(ci_low) and math.isfinite(ci_high)
    assert ci_low < ci_high, f"CI is inverted: [{ci_low}, {ci_high}]"
    # The observed Sharpe should fall within the CI (percentile method property)
    assert observed_sharpe is not None
    assert ci_low <= observed_sharpe <= ci_high, (
        f"Observed Sharpe {observed_sharpe:.4f} is outside CI [{ci_low:.4f}, {ci_high:.4f}]. "
        "This is possible but very rare with 10k resamples."
    )
