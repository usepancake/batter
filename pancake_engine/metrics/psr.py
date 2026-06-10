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
- ``Φ`` uses ``math.erf`` (stdlib); no SciPy runtime dependency.
"""

from __future__ import annotations

import math

__all__ = ["probabilistic_sharpe_ratio", "psr_sharpe_hat", "return_moments"]


def _phi(x: float) -> float:
    """Standard normal CDF via ``math.erf`` (no SciPy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


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
