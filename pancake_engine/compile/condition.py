"""Condition AST → callable evaluator.

Mirrors the TS condition AST shape:

- ``{"feature": "<col>", "gte"?: N, "lte"?: N, "eq"?: N}``
- ``{"feature_equal": {"a": "<col>", "b": "<col>"}}``
- ``{"all_of": [<node>, ...]}``
- ``{"any_of": [<node>, ...]}``
- ``{"not": <node>}``

Evaluator returns ``bool`` for a given ``row``. Numeric comparisons require
the row value to be a finite number — non-number values cause the condition
to short-circuit to ``False`` (matching TS behavior at L451).
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["compile_condition", "Condition"]

Row = dict[str, Any]
Condition = Callable[[Row], bool]


def compile_condition(node: dict[str, Any]) -> Condition:
    """Compile a condition AST node into a callable ``(row) -> bool``."""
    if not isinstance(node, dict):
        raise ValueError(f"E_EVIDENCE_SPEC_INVALID: condition node must be a dict, got {type(node).__name__}")

    if "all_of" in node:
        children = [compile_condition(c) for c in node["all_of"]]
        return lambda row: all(c(row) for c in children)

    if "any_of" in node:
        children = [compile_condition(c) for c in node["any_of"]]
        if not children:
            # any_of with empty list is False by convention
            return lambda _row: False
        return lambda row: any(c(row) for c in children)

    if "not" in node:
        inner = compile_condition(node["not"])
        return lambda row: not inner(row)

    if "feature" in node:
        col = node["feature"]
        if not isinstance(col, str):
            raise ValueError(f"E_EVIDENCE_SPEC_INVALID: feature must be a string column name, got {col!r}")
        gte = node.get("gte")
        lte = node.get("lte")
        eq = node.get("eq")
        return _make_feature_check(col, gte=gte, lte=lte, eq=eq)

    if "feature_equal" in node:
        pair = node["feature_equal"]
        if not isinstance(pair, dict) or "a" not in pair or "b" not in pair:
            raise ValueError(
                f"E_EVIDENCE_SPEC_INVALID: feature_equal requires {{'a': str, 'b': str}}, got {pair!r}"
            )
        a, b = pair["a"], pair["b"]
        return lambda row: row.get(a) == row.get(b)

    raise ValueError(f"E_EVIDENCE_SPEC_INVALID: unknown condition node keys: {sorted(node.keys())!r}")


def _make_feature_check(
    col: str, *, gte: Any = None, lte: Any = None, eq: Any = None
) -> Condition:
    """Numeric comparison on row[col]. Non-number value → False (matches TS L451)."""

    def check(row: Row) -> bool:
        v = row.get(col)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return False
        if eq is not None:
            return v == eq
        if gte is not None and v < gte:
            return False
        if lte is not None and v > lte:
            return False
        return True

    return check
