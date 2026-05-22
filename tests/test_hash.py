"""SHA-256 over canonical bytes.

Engine 0.3 is correctness-first, not TS parity. The hash is the determinism
gate — same inputs must produce the same hex digest on every supported
runtime in the CI matrix.
"""

from __future__ import annotations

import hashlib

from pancake_engine.hash import sha256_canonical


def test_hash_stable_simple() -> None:
    """Key order in the input dict does not affect the hash (canonical sorts keys)."""
    assert sha256_canonical({"a": 1, "b": 2}) == sha256_canonical({"b": 2, "a": 1})


def test_hash_format() -> None:
    """SHA-256 output is 64 lowercase hex chars."""
    h = sha256_canonical({"a": 1})
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_hash_different_inputs_differ() -> None:
    assert sha256_canonical({"a": 1}) != sha256_canonical({"a": 2})
    assert sha256_canonical([1, 2, 3]) != sha256_canonical([3, 2, 1])
    # str "1" vs int 1 canonicalize differently
    assert sha256_canonical("1") != sha256_canonical(1)


def test_hash_int_and_float_equivalent() -> None:
    """``0`` and ``0.0`` canonicalize identically; same for ``1`` and ``1.0``."""
    assert sha256_canonical(0) == sha256_canonical(0.0)
    assert sha256_canonical(0) == sha256_canonical(-0.0)
    assert sha256_canonical(1) == sha256_canonical(1.0)


def test_hash_known_value_drift_sentinel() -> None:
    """Hardcoded sentinel: canonicalize({"a": 1}) == b'{"a":1}'.

    If this drifts, the canonical form has changed — investigate before rebaselining.
    """
    expected = hashlib.sha256(b'{"a":1}').hexdigest()
    assert sha256_canonical({"a": 1}) == expected
    assert sha256_canonical({}) == hashlib.sha256(b"{}").hexdigest()
    assert sha256_canonical([]) == hashlib.sha256(b"[]").hexdigest()
    assert sha256_canonical(None) == hashlib.sha256(b"null").hexdigest()
