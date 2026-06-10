"""Spec-level validation beyond pydantic shape checks.

Most ``EvidenceSpec`` validators (``starting_capital > 0``, costs ≥ 0)
live as pydantic field-validators on :mod:`pancake_engine.types`. This
module adds the remaining business rules that aren't expressible in the
pydantic schema directly.

Engine 0.3 is correctness-first, not TS parity. See architecture §Validation.
"""

from __future__ import annotations

from typing import Any

from ..compile.condition import extract_referenced_columns
from ..types import EvidenceSpec
from .verdict import ValidationVerdict

__all__ = ["validate_spec"]

SUPPORTED_SIZING_MODES = {"fixed_fraction"}  # PR-1 ships fixed_fraction only


def _when(node: Any) -> dict[str, Any]:
    """The ``when`` condition AST of an entry/yes_payoff block, or ``{}``."""
    if isinstance(node, dict):
        when = node.get("when")
        if isinstance(when, dict):
            return when
    return {}


def validate_spec(spec: EvidenceSpec) -> ValidationVerdict:
    """Business-rule checks on an already-pydantic-loaded ``EvidenceSpec``."""
    v = ValidationVerdict()

    # Entry / yes_payoff predicates may only reference columns DECLARED in
    # schema_requirements. A predicate on an undeclared column silently evaluates
    # False at runtime (compile/condition.py: a missing/non-number value short-
    # circuits to False) — for a NO-side yes_payoff that means the strategy "wins"
    # every trade and emits astronomical garbage (~2e+68) with no error. This was
    # the 2026-06-10 "fade the favorite" P1 (agent wrote the semantic-role name
    # `resolved_outcome_numeric` instead of the actual column). Reject it here so
    # run_backtest returns a clean blocked verdict instead of a garbage success.
    declared = {req.name for req in spec.schema_requirements.required_columns}
    referenced = (
        extract_referenced_columns(_when(spec.strategy.entry))
        | extract_referenced_columns(_when(spec.strategy.yes_payoff))
    )
    undeclared = sorted(referenced - declared)
    if undeclared:
        v.add_error(
            "E_EVIDENCE_SPEC_INVALID",
            f"entry/yes_payoff predicate(s) reference column(s) not declared in "
            f"schema_requirements.required_columns: {undeclared}. An undeclared "
            f"column silently evaluates False at runtime (NO-side → wins every trade).",
            field="strategy",
        )

    if spec.strategy.side not in ("YES", "NO"):
        v.add_error(
            "E_EVIDENCE_SPEC_INVALID",
            f"strategy.side must be YES or NO (got {spec.strategy.side!r})",
            field="strategy.side",
        )

    if spec.strategy.sizing.mode not in SUPPORTED_SIZING_MODES:
        v.add_error(
            "E_EVIDENCE_SPEC_INVALID",
            f"strategy.sizing.mode={spec.strategy.sizing.mode!r} is not supported in this engine "
            f"version (PR-1 ships only {sorted(SUPPORTED_SIZING_MODES)!r})",
            field="strategy.sizing.mode",
        )

    if not (0 < spec.strategy.sizing.value <= 1):
        v.add_error(
            "E_EVIDENCE_SPEC_INVALID",
            f"strategy.sizing.value must be in (0, 1] for fixed_fraction "
            f"(got {spec.strategy.sizing.value})",
            field="strategy.sizing.value",
        )

    return v
