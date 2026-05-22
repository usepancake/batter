"""Condition AST compile + evaluator."""

from __future__ import annotations

import pytest

from pancake_engine.compile import compile_condition


def test_feature_gte() -> None:
    c = compile_condition({"feature": "alpha", "gte": 2.0})
    assert c({"alpha": 2.0}) is True
    assert c({"alpha": 3.5}) is True
    assert c({"alpha": 1.999}) is False


def test_feature_lte() -> None:
    c = compile_condition({"feature": "alpha", "lte": 5.0})
    assert c({"alpha": 5.0}) is True
    assert c({"alpha": 4.0}) is True
    assert c({"alpha": 5.001}) is False


def test_feature_eq() -> None:
    c = compile_condition({"feature": "alpha", "eq": 3})
    assert c({"alpha": 3}) is True
    assert c({"alpha": 3.0}) is True
    assert c({"alpha": 2.999}) is False


def test_feature_range_both() -> None:
    c = compile_condition({"feature": "alpha", "gte": 2.0, "lte": 5.0})
    assert c({"alpha": 3.0}) is True
    assert c({"alpha": 2.0}) is True
    assert c({"alpha": 5.0}) is True
    assert c({"alpha": 1.99}) is False
    assert c({"alpha": 5.01}) is False


def test_feature_non_number_short_circuits_false() -> None:
    c = compile_condition({"feature": "alpha", "gte": 0.0})
    assert c({"alpha": "string"}) is False
    assert c({"alpha": None}) is False
    assert c({"alpha": True}) is False  # bool excluded


def test_feature_equal() -> None:
    c = compile_condition({"feature_equal": {"a": "target", "b": "outcome"}})
    assert c({"target": 1, "outcome": 1}) is True
    assert c({"target": 1, "outcome": 0}) is False
    assert c({"target": "x", "outcome": "x"}) is True


def test_all_of() -> None:
    c = compile_condition({"all_of": [
        {"feature": "alpha", "gte": 2.0},
        {"feature": "target", "eq": 1},
    ]})
    assert c({"alpha": 3.0, "target": 1}) is True
    assert c({"alpha": 1.0, "target": 1}) is False
    assert c({"alpha": 3.0, "target": 0}) is False


def test_any_of() -> None:
    c = compile_condition({"any_of": [
        {"feature": "alpha", "gte": 5.0},
        {"feature": "target", "eq": 1},
    ]})
    assert c({"alpha": 1.0, "target": 1}) is True
    assert c({"alpha": 6.0, "target": 0}) is True
    assert c({"alpha": 1.0, "target": 0}) is False


def test_any_of_empty() -> None:
    c = compile_condition({"any_of": []})
    assert c({"x": 1}) is False


def test_not() -> None:
    c = compile_condition({"not": {"feature": "alpha", "gte": 5.0}})
    assert c({"alpha": 4.0}) is True
    assert c({"alpha": 6.0}) is False


def test_nested() -> None:
    c = compile_condition({"all_of": [
        {"any_of": [
            {"feature": "alpha", "gte": 5.0},
            {"feature": "target", "eq": 1},
        ]},
        {"not": {"feature": "alpha", "gte": 10.0}},
    ]})
    assert c({"alpha": 5.0, "target": 0}) is True
    assert c({"alpha": 11.0, "target": 1}) is False  # blocked by not


def test_unknown_node_keys_raises() -> None:
    with pytest.raises(ValueError, match="E_EVIDENCE_SPEC_INVALID"):
        compile_condition({"unknown_op": "x"})


def test_feature_string_name_required() -> None:
    with pytest.raises(ValueError, match="E_EVIDENCE_SPEC_INVALID"):
        compile_condition({"feature": 1, "gte": 0})
