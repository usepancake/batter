"""Permutation test for Sharpe ratio significance — Pancake Engine 0.4.

**Method**: Two-tailed sign-permutation test for Sharpe ratio.
- Null hypothesis: Sharpe = 0 (returns are exchangeable with sign flips).
- Permute the sign of each return uniformly at random, recompute the Sharpe
  ratio per permutation, return the p-value as the fraction of permuted
  |Sharpe| values ≥ |observed Sharpe|.
- This is the standard "randomisation test for Sharpe" described in Good (2005)
  §4.2 and implemented widely in quantitative finance (e.g., Bailey & López de
  Prado 2012, "The Sharpe Ratio Efficient Frontier").

**Why sign permutation and not full row shuffle?**
Sign permutation preserves the marginal distribution of |r_t| while scrambling
direction — it is the maximum-entropy null under the symmetry assumption
(returns are equally likely to be positive or negative). Full row shuffle is
equivalent when returns are i.i.d. but breaks down for autocorrelated series;
sign permutation is more conservative and appropriate here.

**RNG choice**: ``numpy.random.default_rng(seed)`` with PCG64. Same RNG as
``bootstrap.py`` — cross-platform byte-stable. See bootstrap.py module
docstring for rationale.

**References**:
- Good, P. I. (2005). *Permutation, Parametric and Bootstrap Tests of
  Hypotheses*, 3rd ed. Springer. §4.2 (randomisation tests for financial data).
- Bailey, D. H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient
  Frontier." *Journal of Risk*, 15(2), 3–44.
"""

from __future__ import annotations

import math

import numpy as np

from ..warnings import Severity, Warning, WarningCode

__all__ = ["permutation_p_sharpe"]


# Minimum sample size for a meaningful permutation test.
# With N < 10 the maximum achievable p-value is 2/2^10 ≈ 0.002 and the minimum
# is 1.0, so there are only 2^N possible sign configurations — far too coarse
# to estimate a continuous p-value.
_MIN_N = 10


def permutation_p_sharpe(
    daily_returns: list[float],
    n_permutations: int = 10_000,
    seed: int = 0,
) -> tuple[float | None, list[Warning]]:
    """Sign-permutation p-value for Sharpe ratio under the null Sharpe = 0.

    Args:
        daily_returns: Sequence of daily portfolio returns. Must have at least
            ``_MIN_N`` (10) values; otherwise returns ``(None, [warning])``.
        n_permutations: Number of permutations. Default 10 000.
        seed: Integer seed for ``numpy.random.default_rng`` (PCG64).

    Returns:
        ``(p_value, warnings)`` where:
        - ``p_value``: fraction of permuted |Sharpe| ≥ |observed Sharpe|.
          A value ≤ 0.05 suggests the observed Sharpe is unlikely under the
          null of zero skill.
        - ``warnings``: list of ``Warning`` objects emitted by this call.
          The caller is responsible for collecting and propagating them.

    Side-notes:
        - ``PERMUTATION_P_HIGH`` is emitted when p > 0.10 (signal weak vs
          random). This is calibrated as the standard "weak evidence" threshold.
        - If the observed Sharpe is ``None`` (n<2 or std=0), ``p_value=None``
          is returned (cannot test a non-finite statistic).
        - The minimum achievable p-value is ``1 / n_permutations`` (one permutation
          exactly matched or exceeded |observed Sharpe|).

    Hand-calc fixture (documented in docs/math-audit-0.4.md §permutation):
        daily_returns = [0.01] * 10
        All returns are identical positive → std = 0 → Sharpe = None
        → p_value = None (no test, not PERMUTATION_P_HIGH)
    """
    warnings: list[Warning] = []

    if len(daily_returns) < _MIN_N:
        warnings.append(Warning(
            code=WarningCode.PERMUTATION_P_HIGH,
            severity=Severity.WARN,
            message=(
                f"permutation_p_sharpe: N={len(daily_returns)} < {_MIN_N}; "
                "too few observations for a meaningful permutation test. "
                "Returning None."
            ),
            context={"n": len(daily_returns), "min_n": _MIN_N},
        ))
        return None, warnings

    observed_sharpe = _sharpe(daily_returns)
    if observed_sharpe is None:
        # Cannot test a non-finite or undefined statistic.
        return None, warnings

    arr = np.asarray(daily_returns, dtype=np.float64)
    rng = np.random.default_rng(seed)
    obs_abs = abs(observed_sharpe)

    count_ge = 0
    for _ in range(n_permutations):
        # Draw random signs: +1 or −1 with equal probability.
        signs = rng.integers(0, 2, size=len(arr)) * 2 - 1  # {-1, +1}
        permuted = (arr * signs).tolist()
        perm_sharpe = _sharpe(permuted)
        if perm_sharpe is not None and abs(perm_sharpe) >= obs_abs:
            count_ge += 1

    p_value = count_ge / n_permutations

    if p_value > 0.10:
        warnings.append(Warning(
            code=WarningCode.PERMUTATION_P_HIGH,
            severity=Severity.WARN,
            message=(
                f"permutation_p_sharpe: p={p_value:.4f} > 0.10. "
                "Observed Sharpe ratio is not statistically distinguishable from "
                "noise at the 10% level."
            ),
            context={
                "p_value": p_value,
                "observed_sharpe": observed_sharpe,
                "n": len(daily_returns),
            },
        ))

    return p_value, warnings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sharpe(daily_returns: list[float]) -> float | None:
    """Annualized Sharpe with rf=0, Bessel-corrected std. Returns None if n<2 or std=0."""
    if len(daily_returns) < 2:
        return None
    n = len(daily_returns)
    mean = sum(daily_returns) / n
    var = sum((r - mean) ** 2 for r in daily_returns) / (n - 1)
    std = math.sqrt(var)
    if std == 0.0:
        return None
    return (mean / std) * math.sqrt(252)
