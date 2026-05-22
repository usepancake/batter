"""Canonical serialization tests.

Engine 0.3 is correctness-first, not TS parity. The 50-case byte-equality test
against V8 JSON.stringify is the substrate gate for every downstream hash.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pancake_engine.canonical import canonical_string, canonicalize

CANONICAL_FIXTURES = Path(__file__).parent / "fixtures" / "canonical"


def _load_cases() -> list[dict]:
    return json.loads((CANONICAL_FIXTURES / "cases.json").read_text(encoding="utf-8"))


def _load_expected() -> dict[str, str]:
    path = CANONICAL_FIXTURES / "expected_bytes.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["name"]: row["canonical"] for row in data}


def test_canonical_byte_equality_v8() -> None:
    """50-case numeric fixture: Python canonicalize matches V8 JSON.stringify byte-for-byte.

    If ``expected_bytes.json`` is absent, run ``node tests/fixtures/canonical/v8_oracle.js``
    to regenerate.
    """
    expected = _load_expected()
    if not expected:
        pytest.skip(
            "expected_bytes.json missing; run `node tests/fixtures/canonical/v8_oracle.js`"
        )
    cases = _load_cases()
    failures: list[str] = []
    for case in cases:
        name = case["name"]
        value = case["value"]
        if name not in expected:
            failures.append(f"{name}: no V8 oracle output for this case")
            continue
        try:
            got = canonicalize(value).decode("utf-8")
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}: Python raised {e!r}, expected {expected[name]!r}")
            continue
        if got != expected[name]:
            failures.append(
                f"{name}: got {got!r}, expected {expected[name]!r} (value={value!r})"
            )
    assert not failures, "\n".join(["", *failures])


def test_canonical_rejects_nan_inf() -> None:
    with pytest.raises(ValueError, match="E_NONFINITE"):
        canonicalize(float("nan"))
    with pytest.raises(ValueError, match="E_NONFINITE"):
        canonicalize(float("inf"))
    with pytest.raises(ValueError, match="E_NONFINITE"):
        canonicalize(float("-inf"))


def test_canonical_rejects_lone_surrogate() -> None:
    with pytest.raises(ValueError, match="E_LONE_SURROGATE"):
        canonicalize("\ud800")


def test_canonical_rejects_integer_above_2_to_53() -> None:
    with pytest.raises(ValueError, match="E_INTEGER_TOO_LARGE"):
        canonicalize(2**53 + 1)
    with pytest.raises(ValueError, match="E_INTEGER_TOO_LARGE"):
        canonicalize(-(2**53 + 1))
    # 2**53 itself is exactly representable as a float; allowed.
    assert canonicalize(2**53) == b"9007199254740992"
    assert canonicalize(-(2**53)) == b"-9007199254740992"


def test_canonical_normalizes_negative_zero() -> None:
    """``canonicalize(-0.0)`` returns ``b"0"`` — matches V8 normalization of ``-0`` to ``"0"``."""
    assert canonicalize(-0.0) == b"0"
    assert canonicalize(0.0) == b"0"
    assert canonicalize(0) == b"0"


def test_canonical_array_order_preserved() -> None:
    """Arrays preserve insertion order; never sorted."""
    assert canonicalize([3, 1, 2]) == b"[3,1,2]"
    assert canonicalize(["b", "a"]) == b'["b","a"]'
    assert canonicalize([]) == b"[]"


def test_canonical_object_keys_sorted_codepoint() -> None:
    """Object keys sort by Unicode codepoint, not locale, recursively."""
    # "a" (U+0061) < "b" (U+0062) < "ä" (U+00E4)
    assert canonicalize({"ä": 3, "b": 2, "a": 1}) == b'{"a":1,"b":2,"\xc3\xa4":3}'
    # "1" (U+0031) < "2" (U+0032) — string-sort, not numeric
    assert canonicalize({"10": 1, "2": 2}) == b'{"10":1,"2":2}'
    # Nested sorting
    obj = {"b": [1, 2, {"d": 4, "c": 3}], "a": "x"}
    assert canonicalize(obj) == b'{"a":"x","b":[1,2,{"c":3,"d":4}]}'


def test_canonical_unicode_nfc() -> None:
    """Composed and decomposed Unicode forms canonicalize identically."""
    composed = "ä"       # ä as single codepoint U+00E4
    decomposed = "ä"    # a + combining diaeresis U+0308
    assert canonicalize(composed) == canonicalize(decomposed)


def test_canonical_control_chars_escaped() -> None:
    """Control characters and special chars are JSON-escaped."""
    assert canonicalize("\x01") == b'"\\u0001"'
    assert canonicalize("\n") == b'"\\n"'
    assert canonicalize("\t") == b'"\\t"'
    assert canonicalize("\r") == b'"\\r"'
    assert canonicalize("\b") == b'"\\b"'
    assert canonicalize("\f") == b'"\\f"'
    assert canonicalize('"') == b'"\\""'
    assert canonicalize("\\") == b'"\\\\"'


def test_canonical_basic_types() -> None:
    assert canonicalize(None) == b"null"
    assert canonicalize(True) == b"true"
    assert canonicalize(False) == b"false"
    assert canonicalize(0) == b"0"
    assert canonicalize(1) == b"1"
    assert canonicalize(-1) == b"-1"
    assert canonicalize("hello") == b'"hello"'
    assert canonicalize([]) == b"[]"
    assert canonicalize({}) == b"{}"


def test_canonical_unsupported_type() -> None:
    with pytest.raises(ValueError, match="E_UNSUPPORTED_TYPE"):
        canonicalize({"k": object()})


def test_canonical_non_string_keys_rejected() -> None:
    with pytest.raises(ValueError, match="E_NON_STRING_KEY"):
        canonicalize({1: "v"})


def test_canonical_string_helper_matches_canonicalize() -> None:
    """``canonical_string(x).encode('utf-8')`` equals ``canonicalize(x)``."""
    for obj in [None, True, False, 0, 1.5, "hello", [1, 2], {"a": 1}]:
        assert canonical_string(obj).encode("utf-8") == canonicalize(obj)
