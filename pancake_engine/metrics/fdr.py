"""False-discovery-rate control (Benjamini-Yekutieli / Benjamini-Hochberg).

When a strategy is evaluated across many configurations — e.g. the 0.7.0
robustness sweep's 7×7 entry×sizing grid — reporting the single best raw p-value
as "significant" is the multiple-testing trap. FDR control adjusts for the number
of hypotheses tested and reports how many survive.

- ``method="by"`` (default): Benjamini & Yekutieli (2001). Valid under ARBITRARY
  dependence — the right default here because sweep cells computed over one dataset
  are strongly correlated. Uses the harmonic correction factor ``c(m)=Σ_{k=1}^m 1/k``.
- ``method="bh"``: Benjamini & Hochberg (1995). Assumes independence / positive
  regression dependence (PRDS); less conservative.

Pure + deterministic (no RNG; ``math.fsum`` for the harmonic factor) so it is
``result_hash``-safe when wired in Phase B.

References:
- Benjamini, Y., & Hochberg, Y. (1995). JRSS-B 57(1), 289–300.
- Benjamini, Y., & Yekutieli, D. (2001). Ann. Statist. 29(4), 1165–1188.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["fdr_control", "FDRResult"]


@dataclass(frozen=True)
class FDRResult:
    method: str
    n: int
    alpha: float
    correction_factor: float            # c(m): 1.0 for BH, harmonic sum for BY
    adjusted_p_values: list[float]      # original order
    significant: list[bool]             # original order; adjusted <= alpha
    n_significant: int
    min_raw_p: float | None
    min_adjusted_p: float | None


def fdr_control(
    p_values: list[float], alpha: float = 0.05, method: str = "by"
) -> FDRResult:
    """Step-up FDR control. Returns adjusted p-values (original order) + the
    significant set at ``alpha``. ``method`` is ``"by"`` (default, arbitrary
    dependence) or ``"bh"`` (independence/PRDS)."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"E_FDR_INVALID_ALPHA: alpha must be in (0, 1), got {alpha}")
    if method not in ("by", "bh"):
        raise ValueError(f"E_FDR_UNKNOWN_METHOD: {method!r}; use 'by' or 'bh'")

    m = len(p_values)
    if m == 0:
        return FDRResult(method, 0, alpha, 1.0, [], [], 0, None, None)

    for p in p_values:
        if (
            isinstance(p, bool)
            or not isinstance(p, (int, float))
            or math.isnan(p)
            or not (0.0 <= p <= 1.0)
        ):
            raise ValueError(f"E_FDR_INVALID_P: p-values must be finite and in [0, 1], got {p!r}")

    factor = math.fsum(1.0 / k for k in range(1, m + 1)) if method == "by" else 1.0

    order = sorted(range(m), key=lambda i: p_values[i])  # ascending by p-value
    adjusted = [0.0] * m
    running_min = 1.0
    # Step-up: walk from the largest p-value down, enforcing monotone non-decreasing
    # adjusted values via the running minimum.
    for rank in range(m, 0, -1):  # rank is 1-based position in the sorted order
        oi = order[rank - 1]
        val = min(running_min, min(1.0, p_values[oi] * m * factor / rank))
        running_min = val
        adjusted[oi] = val

    significant = [a <= alpha for a in adjusted]
    return FDRResult(
        method=method,
        n=m,
        alpha=alpha,
        correction_factor=factor,
        adjusted_p_values=adjusted,
        significant=significant,
        n_significant=sum(significant),
        min_raw_p=min(p_values),
        min_adjusted_p=min(adjusted),
    )
