"""FDR control (Benjamini-Yekutieli / Benjamini-Hochberg).

Verified against hand calculations (BH and BY diverge cleanly on a linear p-ramp)
and an independent vectorized reference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pancake_engine.metrics.fdr import fdr_control


def _ref_adjusted(p_values: list[float], method: str) -> list[float]:
    """Independent vectorized step-up reference (reverse cumulative min)."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    factor = float(np.sum(1.0 / np.arange(1, m + 1))) if method == "by" else 1.0
    order = np.argsort(p, kind="stable")
    ranks = np.arange(1, m + 1)
    p_sorted = p[order]
    adj_sorted = np.minimum.accumulate((p_sorted * m * factor / ranks)[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    out = np.empty(m)
    out[order] = adj_sorted
    return out.tolist()


def test_bh_hand_calc_linear_ramp() -> None:
    # p_(i) = 0.01·i, m=5. BH adjusted = p_(i)·m/i = 0.05 for every i → all sig at 0.05.
    r = fdr_control([0.01, 0.02, 0.03, 0.04, 0.05], alpha=0.05, method="bh")
    assert all(a == pytest.approx(0.05) for a in r.adjusted_p_values)
    assert r.n_significant == 5
    assert all(r.significant)


def test_by_hand_calc_linear_ramp() -> None:
    # Same ramp, BY: c(5)=2.283333, adjusted = 0.05·c(5) ≈ 0.11417 → none sig at 0.05.
    r = fdr_control([0.01, 0.02, 0.03, 0.04, 0.05], alpha=0.05, method="by")
    c5 = 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 5
    assert r.correction_factor == pytest.approx(c5)
    assert all(a == pytest.approx(0.05 * c5) for a in r.adjusted_p_values)
    assert r.n_significant == 0
    assert not any(r.significant)


def test_matches_independent_reference() -> None:
    rng = np.random.default_rng(7)
    pv = sorted(rng.uniform(0.0, 0.2, 20).tolist())  # mostly-small p-values, ties unlikely
    for method in ("bh", "by"):
        r = fdr_control(pv, alpha=0.05, method=method)
        ref = _ref_adjusted(pv, method)
        for got, exp in zip(r.adjusted_p_values, ref):
            assert got == pytest.approx(exp, rel=1e-12, abs=1e-15)


def test_adjusted_is_monotone_in_raw_order() -> None:
    pv = [0.001, 0.7, 0.02, 0.2, 0.04, 0.9]
    r = fdr_control(pv, method="by")
    # sort by raw p; adjusted along that order must be non-decreasing
    by_raw = [a for _, a in sorted(zip(pv, r.adjusted_p_values))]
    assert all(by_raw[i] <= by_raw[i + 1] + 1e-12 for i in range(len(by_raw) - 1))


def test_by_is_more_conservative_than_bh() -> None:
    pv = [0.001, 0.01, 0.02, 0.03, 0.2, 0.5, 0.8]
    bh = fdr_control(pv, method="bh")
    by = fdr_control(pv, method="by")
    assert by.n_significant <= bh.n_significant
    for a_by, a_bh in zip(by.adjusted_p_values, bh.adjusted_p_values):
        assert a_by >= a_bh - 1e-12


def test_edge_cases_and_validation() -> None:
    assert fdr_control([], method="by").n == 0
    assert fdr_control([], method="by").n_significant == 0
    single = fdr_control([0.03], method="by")
    assert single.adjusted_p_values == [pytest.approx(0.03)] and single.n_significant == 1
    with pytest.raises(ValueError, match="E_FDR_INVALID_ALPHA"):
        fdr_control([0.01], alpha=0.0)
    with pytest.raises(ValueError, match="E_FDR_UNKNOWN_METHOD"):
        fdr_control([0.01], method="bonferroni")
    with pytest.raises(ValueError, match="E_FDR_INVALID_P"):
        fdr_control([0.5, 1.5])
    with pytest.raises(ValueError, match="E_FDR_INVALID_P"):
        fdr_control([0.5, float("nan")])


def test_min_fields() -> None:
    pv = [0.2, 0.001, 0.05]
    r = fdr_control(pv, method="by")
    assert r.min_raw_p == pytest.approx(0.001)
    assert r.min_adjusted_p == pytest.approx(min(r.adjusted_p_values))
