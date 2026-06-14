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

from collections.abc import Callable
from typing import Any

__all__ = ["compile_condition", "extract_referenced_columns", "lint_condition", "Condition"]

Row = dict[str, Any]
Condition = Callable[[Row], bool]


def extract_referenced_columns(node: dict[str, Any]) -> set[str]:
    """Walk a condition AST and return all column names referenced by feature predicates.

    Returns the union of:
    - ``node["feature"]`` for every ``{"feature": col, ...}`` node
    - ``node["feature_equal"]["a"]`` and ``["b"]`` for every ``{"feature_equal": ...}`` node

    Does NOT include system-role columns (entry_price, resolution_time, etc.) because
    those are only accessed directly by the runner, not through the condition AST.
    """
    cols: set[str] = set()
    if not isinstance(node, dict):
        return cols

    if "all_of" in node:
        for child in node["all_of"]:
            cols |= extract_referenced_columns(child)
    elif "any_of" in node:
        for child in node["any_of"]:
            cols |= extract_referenced_columns(child)
    elif "not" in node:
        cols |= extract_referenced_columns(node["not"])
    elif "feature" in node:
        col = node["feature"]
        if isinstance(col, str):
            cols.add(col)
    elif "feature_equal" in node:
        pair = node.get("feature_equal", {})
        if isinstance(pair, dict):
            if isinstance(pair.get("a"), str):
                cols.add(pair["a"])
            if isinstance(pair.get("b"), str):
                cols.add(pair["b"])

    return cols


def lint_condition(node: Any) -> list[str]:
    """Walk a condition AST and return all defects that ``compile_condition`` would raise on.

    Returns a list of human-readable error strings (empty = valid). Never raises.
    Covers:
    - Non-dict node
    - empty ``all_of`` (vacuously always-true)
    - ``feature`` node with unknown operator keys (typos like ``gt``)
    - ``feature`` node with no gte/lte/eq (bare feature reference)
    - ``feature_equal`` missing ``a``/``b`` or non-string values
    - Unknown top-level node keys

    Recurses into ``all_of``, ``any_of``, and ``not`` children.
    """
    errors: list[str] = []
    _lint_node(node, errors)
    return errors


def _lint_node(node: Any, errors: list[str]) -> None:
    """Recursive helper — appends error strings to *errors*."""
    if not isinstance(node, dict):
        errors.append(
            f"E_EVIDENCE_SPEC_INVALID: condition node must be a dict, got {type(node).__name__!r}"
        )
        return

    if "all_of" in node:
        children = node["all_of"]
        if not isinstance(children, list) or len(children) == 0:
            errors.append(
                "E_EVIDENCE_SPEC_INVALID: all_of requires at least one child condition"
            )
        else:
            for child in children:
                _lint_node(child, errors)
        return

    if "any_of" in node:
        for child in node.get("any_of") or []:
            _lint_node(child, errors)
        return

    if "not" in node:
        _lint_node(node["not"], errors)
        return

    if "feature" in node:
        col = node["feature"]
        if not isinstance(col, str):
            errors.append(
                f"E_EVIDENCE_SPEC_INVALID: feature must be a string column name, got {col!r}"
            )
        unknown = set(node.keys()) - _FEATURE_NODE_KEYS
        if unknown:
            errors.append(
                f"E_EVIDENCE_SPEC_INVALID: unknown operator key(s) in feature node: "
                f"{sorted(unknown)!r}; valid operators are gte, lte, eq"
            )
        elif node.get("gte") is None and node.get("lte") is None and node.get("eq") is None:
            errors.append(
                "E_EVIDENCE_SPEC_INVALID: feature node requires at least one of gte/lte/eq "
                "(a bare feature reference matches every numeric row)"
            )
        return

    if "feature_equal" in node:
        pair = node["feature_equal"]
        if not isinstance(pair, dict) or "a" not in pair or "b" not in pair:
            errors.append(
                f"E_EVIDENCE_SPEC_INVALID: feature_equal requires {{'a': str, 'b': str}}, got {pair!r}"
            )
        else:
            a, b = pair["a"], pair["b"]
            if not isinstance(a, str) or not isinstance(b, str):
                errors.append(
                    f"E_EVIDENCE_SPEC_INVALID: feature_equal 'a' and 'b' must be string column "
                    f"names, got a={a!r}, b={b!r}"
                )
        return

    errors.append(
        f"E_EVIDENCE_SPEC_INVALID: unknown condition node keys: {sorted(node.keys())!r}"
    )


_FEATURE_NODE_KEYS = frozenset({"feature", "gte", "lte", "eq"})


def compile_condition(node: dict[str, Any]) -> Condition:
    """Compile a condition AST node into a callable ``(row) -> bool``."""
    if not isinstance(node, dict):
        raise ValueError(f"E_EVIDENCE_SPEC_INVALID: condition node must be a dict, got {type(node).__name__}")

    if "all_of" in node:
        children = [compile_condition(c) for c in node["all_of"]]
        if not children:
            # all([]) is vacuously True → a silent always-true entry condition.
            # An empty all_of is a spec error; reject it, consistent with the
            # unknown-operator guard below and the 0.6.0 always-true fix. (any_of
            # empty is a coherent False sentinel; empty all_of is not.)
            raise ValueError(
                "E_EVIDENCE_SPEC_INVALID: all_of requires at least one child condition"
            )
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
        # Reject typo'd / unknown operator keys ("gt", "GTE", "lt", ...). Without
        # this, an unrecognised key is silently ignored and the node degenerates
        # to an always-true numeric-existence check — a strategy that "enters on
        # everything" with no error. See audit 2026-06-04 finding #1.
        unknown = set(node.keys()) - _FEATURE_NODE_KEYS
        if unknown:
            raise ValueError(
                f"E_EVIDENCE_SPEC_INVALID: unknown operator key(s) in feature node: "
                f"{sorted(unknown)!r}; valid operators are gte, lte, eq"
            )
        gte = node.get("gte")
        lte = node.get("lte")
        eq = node.get("eq")
        if gte is None and lte is None and eq is None:
            raise ValueError(
                "E_EVIDENCE_SPEC_INVALID: feature node requires at least one of gte/lte/eq "
                "(a bare feature reference matches every numeric row)"
            )
        return _make_feature_check(col, gte=gte, lte=lte, eq=eq)

    if "feature_equal" in node:
        pair = node["feature_equal"]
        if not isinstance(pair, dict) or "a" not in pair or "b" not in pair:
            raise ValueError(
                f"E_EVIDENCE_SPEC_INVALID: feature_equal requires {{'a': str, 'b': str}}, got {pair!r}"
            )
        a, b = pair["a"], pair["b"]
        if not isinstance(a, str) or not isinstance(b, str):
            raise ValueError(
                f"E_EVIDENCE_SPEC_INVALID: feature_equal 'a' and 'b' must be string column names, "
                f"got a={a!r}, b={b!r}"
            )
        # Require both sides present: two absent columns must NOT match
        # (None == None → True would be a spurious always-enter). Audit #3.
        return lambda row: (av := row.get(a)) is not None and av == row.get(b)

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
            return bool(v == eq)
        if gte is not None and v < gte:
            return False
        if lte is not None and v > lte:
            return False
        return True

    return check
