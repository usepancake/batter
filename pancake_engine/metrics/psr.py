"""Probabilistic Sharpe Ratio (Bailey & López de Prado 2012).

``PSR(SR*)`` estimates ``P(true SR > SR*)`` given the observed Sharpe and the
higher moments of the return distribution. It corrects the naïve Sharpe for
sample length, skewness and (non-excess) kurtosis: a strategy with the same
Sharpe but negative skew, fat tails, or fewer observations is less credible,
and PSR says so. This is the primary significance signal for batter 0.8.0,
complementing (not replacing) the existing sign-permutation ``sharpe_p_value``.

Reference variance of the Sharpe estimator (Mertens 2002 / Lo 2002):

    Var(SR_hat) ≈ (1 - γ3·SR_hat + ((γ4 - 1)/4)·SR_hat²) / (n - 1)

so for normal returns (γ3=0, γ4=3) this collapses to the familiar
``(1 + SR²/2)/(n-1)``. Then

    PSR(SR*) = Φ( (SR_hat - SR*) / sqrt(Var(SR_hat)) )

Conventions (documented so an independent quant reproduces the number exactly):

- ``SR_hat`` is the PER-PERIOD (non-annualised) Sharpe = ``mean / std`` with
  Bessel's correction (ddof=1) — identical to ``metrics.standard.sharpe_ratio``
  before the ``sqrt(252)`` annualisation. ``sr_benchmark`` is on the same scale.
- ``γ3`` (skew) and ``γ4`` (kurtosis) are the POPULATION standardised central
  moments (divide by ``n``), matching ``scipy.stats.skew(bias=True)`` and
  ``scipy.stats.kurtosis(fisher=False, bias=True)``. ``γ4`` is NON-excess
  (normal == 3).
- Every sum uses ``math.fsum`` (correctly rounded) so the result is identical
  across interpreters — PSR enters ``result_hash`` (0.8.0 is a deliberate break).
- ``Φ`` / ``Φ⁻¹`` are computed in 50-digit ``decimal`` (libmpdec) — NOT libm.
  0.8.0 used ``math.erf``, whose last-ULP rounding differs between glibc and
  Apple libm: psr diverged by 2 ULP between ubuntu and macOS, breaking
  cross-platform ``result_hash`` equality (the receipt contract). Hashed values
  may only come from operations correctly rounded BY SPECIFICATION everywhere;
  IEEE-754 guarantees that for +,-,*,/,sqrt — and the General Decimal
  Arithmetic spec guarantees it for ``decimal`` arithmetic, ``sqrt`` and
  ``exp``. erf is computed by its Maclaurin series at 50 digits with pinned
  constants; the probit refines an Acklam float seed with Decimal-Newton
  (quadratic convergence erases the seed's libm ULPs), so the single final
  float rounding is identical on every platform. (0.8.1 determinism hotfix.)
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

__all__ = [
    "probabilistic_sharpe_ratio",
    "psr_sharpe_hat",
    "return_moments",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "min_track_record_length",
]

_EULER_MASCHERONI = 0.5772156649015329

# 50-digit pinned constants for the decimal paths (deterministic by construction).
_PREC = 50
_D_SQRT2 = Decimal("1.41421356237309504880168872420969807856967187537695")
_D_2_OVER_SQRTPI = Decimal("1.12837916709551257389615890312154517168810125865800")
_D_SQRT_2PI = Decimal("2.50662827463100050241576528481104525300698674060994")


def _erf_dec(x: "Decimal") -> "Decimal":
    """erf(x) by Maclaurin series in the CURRENT decimal context (prec=_PREC).

    erf(x) = 2/√π · Σ_{n≥0} (-1)^n x^(2n+1) / (n!·(2n+1)), term recurrence
    c_{n+1} = c_n·(-x²)/(n+1). At |x| ≤ 8 the alternating series loses ≤ ~28
    digits to cancellation — prec 50 leaves ≥ 22 good digits, far beyond the
    17 needed for an exact float64 rounding. All operations are libmpdec
    (correctly rounded per spec) → byte-identical on every platform.
    """
    x2 = x * x
    c = x
    s = Decimal(0)
    n = 0
    while True:
        t = c / (2 * n + 1)
        s += t
        if abs(t) < Decimal("1e-45") * (abs(s) + Decimal("1e-30")):
            break
        n += 1
        c = c * (-x2) / n
        if n > 500:  # unreachable for |x| ≤ 8; defensive
            break
    return _D_2_OVER_SQRTPI * s


def _phi_dec(z: "Decimal") -> "Decimal":
    """Standard normal CDF in the current decimal context."""
    return (1 + _erf_dec(z / _D_SQRT2)) / 2


def _phi(z: float) -> float:
    """Standard normal CDF, byte-identical across platforms (see module note).

    |z| ≥ 11 is clamped symmetrically to 0.0 / 1.0 — a deliberate CONVENTION,
    not a claim that both bounds are the correctly-rounded float64:

    - Upper tail (z ≥ 11): Φ(11) ≈ 1 − 2e-28. The distance to 1.0 is ~2e-28,
      which is far below half an ULP of 1.0 (≈ 1.1e-16), so 1.0 IS the
      correctly-rounded value here.
    - Lower tail (z ≤ −11): Φ(−11) ≈ 2e-28. float64 CAN represent values as
      small as ~6.6e-31 (the minimum positive subnormal), so 0.0 is NOT the
      correctly-rounded result — the true value is representable. We return 0.0
      anyway for SYMMETRY with the upper bound: PSR values below ~2e-28 are
      statistically indistinguishable from zero in any practical context, and a
      symmetric clamp keeps the contract simple. This is a shipped, hashed
      convention; behaviour will not change.
    """
    if math.isnan(z):
        raise ValueError("_phi domain error: z is NaN")
    if z >= 11.0:
        return 1.0
    if z <= -11.0:
        return 0.0
    with localcontext() as ctx:
        ctx.prec = _PREC
        return float(_phi_dec(Decimal(z)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (probit), byte-identical across platforms.

    Acklam's rational approximation (float) seeds a Newton iteration carried
    out in 50-digit decimal against the same series-based Φ used by ``_phi``,
    with the normal pdf via the correctly-rounded ``Decimal.exp``. Quadratic
    convergence (seed error ~1e-9 → ~1e-36 in two steps) makes the converged
    value — and hence the single final float rounding — independent of any
    libm ULP differences in the seed. Domain (0, 1); |z| > 10 unsupported
    (raises — quantiles that extreme have no engine use)."""
    if not (0.0 < p < 1.0):
        raise ValueError(f"_norm_ppf domain error: p must be in (0, 1), got {p}")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = ((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
             / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    elif p <= 1.0 - p_low:
        q = p - 0.5
        r = q * q
        x = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
             / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0))
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
              / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    # Decimal-Newton refinement against the deterministic series Φ. The float
    # Acklam seed (above) is pure rational arithmetic in the central region but
    # uses libm log/sqrt in the tails — irrelevant either way: Newton's quadratic
    # convergence to the 50-digit root erases any ULP-level seed differences, so
    # the final float depends only on correctly-rounded decimal operations.
    if abs(x) > 10.0:
        raise ValueError(
            f"_norm_ppf domain error: quantile |z|>10 unsupported (p={p!r})"
        )
    with localcontext() as ctx:
        ctx.prec = _PREC
        dp = Decimal(p)
        dx = Decimal(x)
        for _ in range(4):
            err = _phi_dec(dx) - dp
            pdf = (-(dx * dx) / 2).exp() / _D_SQRT_2PI
            if pdf == 0:
                break
            dx = dx - err / pdf
        return float(dx)


def return_moments(returns: list[float]) -> tuple[float, float, float, int] | None:
    """``(sr_hat, skew, kurtosis, n)`` or ``None`` if undefined.

    ``sr_hat`` uses sample std (ddof=1); ``skew`` / ``kurtosis`` are population
    standardised central moments (kurtosis non-excess). ``None`` when ``n < 2``
    or the return series has zero variance (no Sharpe to speak of).
    """
    n = len(returns)
    if n < 2:
        return None
    mean = math.fsum(returns) / n
    m2 = math.fsum((r - mean) ** 2 for r in returns) / n
    if m2 <= 0.0:
        return None
    m3 = math.fsum((r - mean) ** 3 for r in returns) / n
    m4 = math.fsum((r - mean) ** 4 for r in returns) / n
    std_sample = math.sqrt(m2 * n / (n - 1))  # ddof=1, matches sharpe_ratio
    if std_sample == 0.0:
        return None
    sr_hat = mean / std_sample
    skew = m3 / (m2 ** 1.5)
    kurtosis = m4 / (m2 ** 2)  # non-excess (normal == 3)
    return sr_hat, skew, kurtosis, n


def psr_sharpe_hat(returns: list[float]) -> float | None:
    """The per-period (non-annualised) Sharpe PSR is built on. ``None`` if undefined."""
    moments = return_moments(returns)
    return None if moments is None else moments[0]


def probabilistic_sharpe_ratio(
    returns: list[float], sr_benchmark: float = 0.0
) -> float | None:
    """``PSR(sr_benchmark)`` in ``[0, 1]``, or ``None`` if undefined.

    ``sr_benchmark`` is a PER-PERIOD Sharpe (same scale as :func:`psr_sharpe_hat`),
    default ``0.0`` (probability the strategy's true Sharpe beats zero). Returns
    ``None`` when the moments are undefined (n<2 / zero variance) or the estimator
    variance is non-positive (a pathological skew/kurtosis combination).
    """
    moments = return_moments(returns)
    if moments is None:
        return None
    sr_hat, skew, kurtosis, n = moments
    var_term = 1.0 - skew * sr_hat + ((kurtosis - 1.0) / 4.0) * sr_hat * sr_hat
    if var_term <= 0.0:
        return None
    z = (sr_hat - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(var_term)
    return _phi(z)


from typing import Sequence


def expected_max_sharpe(
    trial_sharpes: Sequence[float],
    *,
    sharpes_annualized: bool = False,
    periods_per_year: int = 252,
) -> float | None:
    """Expected maximum Sharpe under the null (Bailey & López de Prado 2014, eq. 8).

        SR*₀ = sqrt(Var(trial Sharpes)) · [ (1-γ)·Z⁻¹(1 - 1/N) + γ·Z⁻¹(1 - 1/(N·e)) ]

    Used as the benchmark for the Deflated Sharpe Ratio.

    ``trial_sharpes`` are the PER-PERIOD Sharpes of the tested configs (pass
    ``sharpes_annualized=True`` to supply annualised Sharpes — they are divided by
    ``sqrt(periods_per_year)`` to match the per-period scale PSR uses).

    Returns ``None`` when:
    - fewer than 2 trials (the formula is undefined for N < 2), OR
    - zero variance across trials (all trials identical → ``max(sr) == min(sr)``
      or sample variance ≤ 0 after floating-point arithmetic; the formula would
      return 0 but that is indistinguishable from a degenerate input, so we
      signal the caller explicitly).
    """
    n_trials = len(trial_sharpes)
    if n_trials < 2:
        return None
    sr: list[float] = (
        [s / math.sqrt(periods_per_year) for s in trial_sharpes]
        if sharpes_annualized
        else list(trial_sharpes)
    )
    if max(sr) == min(sr):  # zero dispersion (float-robust; var > 0 would be fp noise)
        return None
    mean = math.fsum(sr) / n_trials
    var = math.fsum((s - mean) ** 2 for s in sr) / (n_trials - 1)  # sample var of trial SRs
    if var <= 0.0:
        return None
    return math.sqrt(var) * (
        (1.0 - _EULER_MASCHERONI) * _norm_ppf(1.0 - 1.0 / n_trials)
        + _EULER_MASCHERONI * _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    )


def deflated_sharpe_ratio(
    returns: list[float],
    trial_sharpes: list[float],
    *,
    sharpes_annualized: bool = False,
    periods_per_year: int = 252,
) -> float | None:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    ``DSR = PSR(SR*₀)`` where ``SR*₀`` is the expected MAXIMUM Sharpe under the null
    across the ``N`` configurations tested — so a strategy selected as best-of-many
    must clear a higher bar. This is the multiple-testing-aware significance signal.

        SR*₀ = sqrt(Var(trial Sharpes)) · [ (1-γ)·Z⁻¹(1 - 1/N) + γ·Z⁻¹(1 - 1/(N·e)) ]

    ``trial_sharpes`` are the PER-PERIOD Sharpes of the tested configs (pass
    ``sharpes_annualized=True`` to supply annualised Sharpes — they are divided by
    ``sqrt(periods_per_year)`` to match the per-period scale PSR uses). Returns DSR
    in ``[0, 1]``, or ``None`` if fewer than 2 trials, zero trial-variance, or the
    return moments are undefined.
    """
    emax = expected_max_sharpe(
        trial_sharpes, sharpes_annualized=sharpes_annualized, periods_per_year=periods_per_year
    )
    if emax is None:
        return None
    return probabilistic_sharpe_ratio(returns, sr_benchmark=emax)


def min_track_record_length(
    returns: list[float], target_confidence: float = 0.95, sr_benchmark: float = 0.0
) -> float | None:
    """Minimum Track Record Length (Bailey & López de Prado 2012).

    The smallest number of observations for the observed (per-period) Sharpe to be
    significant above ``sr_benchmark`` at ``target_confidence``:

        MinTRL = 1 + (1 - γ3·SR + ((γ4-1)/4)·SR²) · (Z_c / (SR - SR*))²

    (the ``n`` at which the PSR z-statistic equals ``Z_c``). Returns the float
    observation count, or ``None`` if the moments are undefined or the Sharpe does
    not exceed the benchmark (significance is then unreachable).
    """
    if not (0.0 < target_confidence < 1.0):
        raise ValueError(
            f"E_MINTRL_INVALID_CONFIDENCE: target_confidence must be in (0, 1), got {target_confidence}"
        )
    moments = return_moments(returns)
    if moments is None:
        return None
    sr_hat, skew, kurtosis, _n = moments
    if sr_hat <= sr_benchmark:
        return None
    var_term = 1.0 - skew * sr_hat + ((kurtosis - 1.0) / 4.0) * sr_hat * sr_hat
    if var_term <= 0.0:
        return None
    z = _norm_ppf(target_confidence)
    return 1.0 + var_term * (z / (sr_hat - sr_benchmark)) ** 2
