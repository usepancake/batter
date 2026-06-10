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
from collections.abc import Callable

import numpy as np

from ..warnings import Severity, Warning, WarningCode

__all__ = ["bootstrap_ci", "block_bootstrap_ci"]

# Type alias for a (ci_low, ci_high) tuple
CITuple = tuple[float | None, float | None]

# An index-maker draws a (n_resamples × n) integer index matrix from ``rng``.
# Default (None) is IID-with-replacement; block_bootstrap_ci passes a stationary
# (Politis-Romano) generator. This is the resampling Seam — everything else
# (guards, percentile method, CI_TOO_WIDE / degenerate-CI handling) is shared.
IndexMaker = Callable[[np.random.Generator, int, int], np.ndarray]

# Upper bound on n_resamples: a pathological caller (public API) could otherwise
# allocate an enormous (n_resamples × n) index matrix. Audit 2026-06-04 #7.
_MAX_RESAMPLES = 1_000_000


def bootstrap_ci(
    daily_returns: list[float],
    metric_fn: Callable[[list[float]], float | None],
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 0,
    *,
    make_indices: IndexMaker | None = None,
) -> tuple[CITuple, list[Warning]]:
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

    if not 1 <= n_resamples <= _MAX_RESAMPLES:
        raise ValueError(
            f"E_EVIDENCE_SPEC_INVALID: n_resamples must be in "
            f"[1, {_MAX_RESAMPLES}], got {n_resamples}"
        )

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
        return (None, None), warnings

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
        return (None, None), warnings

    rng = np.random.default_rng(seed)
    n = len(arr)

    # Draw resample indices: shape (n_resamples, n). Default is IID with
    # replacement; the make_indices Seam lets block_bootstrap_ci substitute a
    # stationary (Politis-Romano) resampler that preserves serial correlation.
    indices = (
        rng.integers(0, n, size=(n_resamples, n))
        if make_indices is None
        else make_indices(rng, n, n_resamples)
    )
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
        return (None, None), warnings

    alpha = 1.0 - ci_level
    low_pct = (alpha / 2.0) * 100.0
    high_pct = (1.0 - alpha / 2.0) * 100.0
    boot_arr = np.asarray(boot_stats, dtype=np.float64)
    ci_low = float(np.percentile(boot_arr, low_pct))
    ci_high = float(np.percentile(boot_arr, high_pct))

    # Degenerate CI: all (finite) resamples produced the same metric value, so
    # ci_low == ci_high. A zero-width interval is not a meaningful confidence
    # statement (it reads as infinite precision) — surface it as insufficient
    # rather than emit a misleading (v, v). Audit 2026-06-04 finding #6.
    if ci_low == ci_high:
        warnings.append(Warning(
            code=WarningCode.BOOTSTRAP_INSUFFICIENT,
            severity=Severity.WARN,
            message=(
                f"bootstrap_ci: zero-width CI ({ci_low}); all resamples produced an "
                "identical metric value. CI is degenerate; returning (None, None)."
            ),
            context={"n": len(daily_returns), "reason": "zero_width_ci"},
        ))
        return (None, None), warnings

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
        return (None, None), warnings

    return (ci_low, ci_high), warnings


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _is_finite(x: float) -> bool:
    """Return True iff x is a finite float (not NaN, not ±Infinity)."""
    return math.isfinite(x)


def _stationary_indices(
    rng: np.random.Generator, n: int, n_resamples: int, expected_block_length: float
) -> np.ndarray:
    """Politis & Romano (1994) stationary-bootstrap indices, shape (n_resamples, n).

    Random geometric block lengths (mean = ``expected_block_length``) with circular
    wrap, so resamples preserve the serial correlation IID resampling destroys.
    ``p = 1/L`` is the per-step probability of starting a fresh block. Deterministic
    given ``rng`` — fixed draw order (``random()`` then ``integers()`` each step) —
    and PCG64 byte-stable across platforms, like the IID path.
    """
    p = 1.0 / expected_block_length
    idx = np.empty((n_resamples, n), dtype=np.int64)
    cur = rng.integers(0, n, size=n_resamples)
    idx[:, 0] = cur
    for t in range(1, n):
        new_block = rng.random(n_resamples) < p
        fresh = rng.integers(0, n, size=n_resamples)
        cur = np.where(new_block, fresh, (cur + 1) % n)
        idx[:, t] = cur
    return idx


def block_bootstrap_ci(
    daily_returns: list[float],
    metric_fn: Callable[[list[float]], float | None],
    expected_block_length: float | None = None,
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> tuple[CITuple, list[Warning]]:
    """Stationary (Politis & Romano 1994) block-bootstrap CI.

    Preserves the serial correlation IID resampling destroys, so the interval is
    not artificially narrow for autocorrelated return series (volatility clustering,
    momentum, trending equity). Delegates to :func:`bootstrap_ci`, swapping only the
    resampler — same percentile method and degenerate / CI_TOO_WIDE / insufficient
    guards, same determinism contract.

    ``expected_block_length`` defaults to ``sqrt(n)`` (Politis-White rule of thumb).
    A block length of 1 degenerates to (a different draw order of) IID resampling.
    """
    n = len(daily_returns)
    block_len = (
        expected_block_length
        if (expected_block_length is not None and expected_block_length > 0)
        else max(1.0, math.sqrt(n))
    )
    return bootstrap_ci(
        daily_returns,
        metric_fn,
        n_resamples=n_resamples,
        ci_level=ci_level,
        seed=seed,
        make_indices=lambda rng, nn, m: _stationary_indices(rng, nn, m, block_len),
    )
