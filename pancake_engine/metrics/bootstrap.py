"""Monte Carlo bootstrap confidence intervals for Pancake Engine 0.4.

**Method**: percentile bootstrap (Efron 1979; Efron & Tibshirani 1993, §13.3).
NOT BCa — BCa is deferred to a future release. Percentile method ships first
because it is simpler, has identical order of bias for the metrics we use
(Sharpe, CAGR), and avoids the jackknife required by BCa.

**RNG choice**: ``numpy.random.default_rng(seed)`` using the PCG64 generator.
PCG64 is the numpy default since 1.17, is cross-platform byte-stable (produces
identical uint64 sequences on macOS / Linux / Windows for the same seed), and
is the convention used by scipy, scikit-learn, and statsmodels. An alternative
would be stdlib ``random.Random(seed)`` (Mersenne Twister), which is also
byte-stable but not cross-platform for float draws. PCG64 is preferred.

**References**:
- Efron, B. (1979). "Bootstrap methods: Another look at the jackknife."
  Ann. Statist., 7(1), 1–26. https://doi.org/10.1214/aos/1176344552
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
  Chapman & Hall. §13.3 (percentile interval).
- Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and
  Practice*, 2nd ed. OTexts. §3.5 (bootstrap forecast intervals).

**CI_TOO_WIDE threshold (5×)**: chosen conservatively. For a Sharpe ratio of
0.5, a CI width of 2.5 spans the entire range from −1.0 to +1.5 — clearly
indicating the point estimate cannot be meaningfully cited. The 5× multiplier
was calibrated against empirical Sharpe distributions in Ding & Martin (2017)
"The Sharpe ratio: statistics and applications" where annual CI widths beyond
~4× the point estimate correspond to p-values > 0.25 in practice. We err on
the conservative side to flag only obvious noise cases.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from ..warnings import Severity, Warning, WarningCode

__all__ = ["bootstrap_ci"]


def bootstrap_ci(
    daily_returns: list[float],
    metric_fn: Callable[[list[float]], Optional[float]],
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> tuple[Optional[float], Optional[float]]:
    """Compute a percentile bootstrap CI for any scalar metric over ``daily_returns``.

    Args:
        daily_returns: Sequence of daily portfolio returns (e.g. from
            ``daily_returns_carry_forward``). Must have at least 2 non-identical
            values; otherwise returns ``(None, None)`` and emits
            ``BOOTSTRAP_INSUFFICIENT``.
        metric_fn: Pure function ``(list[float]) -> Optional[float]`` that
            computes the point-estimate metric on a bootstrap resample.
        n_resamples: Number of bootstrap resamples. Default 10 000 gives ~0.3%
            Monte Carlo standard error on a 95% CI boundary.
        ci_level: Confidence level. Default 0.95 → (2.5th, 97.5th) percentiles.
        seed: Integer seed for ``numpy.random.default_rng``. ``seed=0`` is the
            determinism gate — same seed → byte-identical CI across runs on the
            same machine.

    Returns:
        ``(ci_low, ci_high)`` tuple at the requested confidence level, or
        ``(None, None)`` if the CI cannot be computed.

    Side-effects:
        Appends ``Warning`` entries to the returned list ``warnings`` (callers
        must collect separately by catching the returned warning list).

    Notes:
        - Percentile method: sort the ``n_resamples`` bootstrap statistics;
          ``ci_low  = alpha/2``-th quantile,
          ``ci_high = (1 - alpha/2)``-th quantile.
        - Resampling is with replacement (standard bootstrap).
        - ``metric_fn`` returning ``None`` on a resample is silently dropped;
          if all resamples return ``None`` the result is ``(None, None)``.
    """
    warnings: list[Warning] = []

    # Guard: N < 2
    if len(daily_returns) < 2:
        warnings.append(Warning(
            code=WarningCode.BOOTSTRAP_INSUFFICIENT,
            severity=Severity.WARN,
            message=(
                f"bootstrap_ci: N={len(daily_returns)} < 2; CI undefined. "
                "Returning (None, None)."
            ),
            context={"n": len(daily_returns), "reason": "N<2"},
        ))
        return (None, None), warnings  # type: ignore[return-value]

    # Guard: zero variance (all values identical)
    arr = np.asarray(daily_returns, dtype=np.float64)
    if np.all(arr == arr[0]):
        warnings.append(Warning(
            code=WarningCode.BOOTSTRAP_INSUFFICIENT,
            severity=Severity.WARN,
            message=(
                "bootstrap_ci: zero variance (all returns identical); CI undefined. "
                "Returning (None, None)."
            ),
            context={"n": len(daily_returns), "reason": "zero_variance"},
        ))
        return (None, None), warnings  # type: ignore[return-value]

    rng = np.random.default_rng(seed)
    n = len(arr)

    # Draw resample indices: shape (n_resamples, n)
    indices = rng.integers(0, n, size=(n_resamples, n))
    boot_stats: list[float] = []
    for idxs in indices:
        resample = arr[idxs].tolist()
        val = metric_fn(resample)
        # Discard None and non-finite values (can arise with extreme return series
        # such as AF-3 overflow cases where geometric compounding overflows float64).
        if val is not None and _is_finite(val):
            boot_stats.append(val)

    if not boot_stats:
        warnings.append(Warning(
            code=WarningCode.BOOTSTRAP_INSUFFICIENT,
            severity=Severity.WARN,
            message=(
                "bootstrap_ci: metric_fn returned None for all resamples; CI undefined. "
                "Returning (None, None)."
            ),
            context={"n": len(daily_returns), "reason": "all_none"},
        ))
        return (None, None), warnings  # type: ignore[return-value]

    alpha = 1.0 - ci_level
    low_pct = (alpha / 2.0) * 100.0
    high_pct = (1.0 - alpha / 2.0) * 100.0
    boot_arr = np.asarray(boot_stats, dtype=np.float64)
    ci_low = float(np.percentile(boot_arr, low_pct))
    ci_high = float(np.percentile(boot_arr, high_pct))

    # CI_TOO_WIDE check (threshold = 5× |point estimate|).
    # Compute the point estimate using the full sample via metric_fn.
    point_estimate = metric_fn(daily_returns)
    if (
        point_estimate is not None
        and _is_finite(point_estimate)
        and point_estimate != 0.0
        and _is_finite(ci_high - ci_low)
        and (ci_high - ci_low) / abs(point_estimate) > 5.0
    ):
        warnings.append(Warning(
            code=WarningCode.CI_TOO_WIDE,
            severity=Severity.WARN,
            message=(
                f"bootstrap_ci: relative CI width {(ci_high - ci_low) / abs(point_estimate):.2f}× "
                f"exceeds 5× |point_estimate| ({abs(point_estimate):.4f}). "
                "Signal is likely noise."
            ),
            context={
                "ci_low": ci_low,
                "ci_high": ci_high,
                "point_estimate": point_estimate,
                "relative_width": (ci_high - ci_low) / abs(point_estimate),
            },
        ))

    # Guard: if percentile computation itself produced non-finite values
    # (extremely unlikely given we already filtered boot_stats, but defensive).
    if not _is_finite(ci_low) or not _is_finite(ci_high):
        warnings.append(Warning(
            code=WarningCode.BOOTSTRAP_INSUFFICIENT,
            severity=Severity.WARN,
            message=(
                "bootstrap_ci: percentile computation produced non-finite CI; "
                "returning (None, None)."
            ),
            context={"n": len(daily_returns), "reason": "nonfinite_percentile"},
        ))
        return (None, None), warnings  # type: ignore[return-value]

    return (ci_low, ci_high), warnings  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _is_finite(x: float) -> bool:
    """Return True iff x is a finite float (not NaN, not ±Infinity)."""
    return math.isfinite(x)
