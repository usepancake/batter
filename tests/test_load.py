"""JSON load with duplicate-key detection at parse time.

Engine 0.3 is correctness-first, not TS parity. Python's default ``json.loads``
silently drops duplicate keys — we must reject at parse time via
``object_pairs_hook``.
"""

from __future__ import annotations

import pytest

from pancake_engine.io.load import parse_json


def test_canonical_rejects_duplicate_keys() -> None:
    text = '{"a": 1, "a": 2}'
    with pytest.raises(ValueError, match="E_DUPLICATE_KEY"):
        parse_json(text)


def test_canonical_rejects_nested_duplicate_keys() -> None:
    text = '{"outer": {"a": 1, "a": 2}}'
    with pytest.raises(ValueError, match="E_DUPLICATE_KEY"):
        parse_json(text)


def test_canonical_rejects_duplicate_in_array_of_objects() -> None:
    text = '[{"a": 1}, {"b": 1, "b": 2}]'
    with pytest.raises(ValueError, match="E_DUPLICATE_KEY"):
        parse_json(text)


def test_load_normal_object() -> None:
    assert parse_json('{"a": 1, "b": 2}') == {"a": 1, "b": 2}


def test_load_array() -> None:
    assert parse_json("[1, 2, 3]") == [1, 2, 3]


def test_load_nested_object() -> None:
    text = '{"outer": {"a": 1, "b": [1, 2, {"c": 3}]}}'
    assert parse_json(text) == {"outer": {"a": 1, "b": [1, 2, {"c": 3}]}}
