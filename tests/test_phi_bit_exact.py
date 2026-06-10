"""Bit-exact cross-platform contract for the deterministic Φ / Φ⁻¹ (0.8.1).

These hex values are the pinned truth: Φ and the probit are computed in 50-digit
decimal (libmpdec — correctly rounded per the General Decimal Arithmetic spec on
every platform), so the SAME float64 must come out on ubuntu / macOS / windows,
bit for bit. 0.8.0 used math.erf (libm): psr diverged by 2 ULP between glibc and
Apple libm, breaking cross-platform result_hash equality. If this test fails on
any platform, the receipt contract is broken — do NOT loosen it to tolerances.
"""

from __future__ import annotations

import pytest

from pancake_engine.metrics.psr import _norm_ppf, _phi

PHI = [
    (-3.7, "0x1.c425151b7d9e4p-14"),
    (-1.0, "0x1.44ed0bb7cb20bp-3"),
    (0.0, "0x1.0000000000000p-1"),
    (0.5, "0x1.62075e232ac77p-1"),
    (1.96, "0x1.f33379d3bd367p-1"),
    (4.2, "0x1.fffe4030e38d2p-1"),
    (8.0, "0x1.ffffffffffffap-1"),
    (11.5, "0x1.0000000000000p+0"),
    (-11.5, "0x0.0p+0"),
]

PPF = [
    (0.025, "-0x1.f5c0331eeff85p+0"),
    (0.5, "0x0.0p+0"),
    (0.9, "0x1.4813c36e26d33p+0"),
    (0.95, "0x1.a515209676abbp+0"),
    (0.975, "0x1.f5c0331eeff83p+0"),
    (0.99, "0x1.29c5c4630ff0ep+1"),
    (1 - 1 / 49, "0x1.05cf5f3884f8ap+1"),
    (1 - 1 / (49 * 2.718281828459045), "0x1.374bf327f6c0cp+1"),
]


@pytest.mark.parametrize("z,expected_hex", PHI)
def test_phi_bit_exact(z: float, expected_hex: str) -> None:
    assert _phi(z).hex() == expected_hex


@pytest.mark.parametrize("p,expected_hex", PPF)
def test_ppf_bit_exact(p: float, expected_hex: str) -> None:
    assert _norm_ppf(p).hex() == expected_hex


def test_phi_guards() -> None:
    with pytest.raises(ValueError):
        _phi(float("nan"))
    with pytest.raises(ValueError):
        _norm_ppf(0.0)
    with pytest.raises(ValueError):
        _norm_ppf(1.0)
    # |z| > 10 quantiles are out of the supported domain (no engine use)
    with pytest.raises(ValueError):
        _norm_ppf(1e-300)
