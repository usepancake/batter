"""Spec-level validation beyond pydantic shape checks.

Most ``EvidenceSpec`` validators (``starting_capital > 0``, costs ≥ 0)
live as pydantic field-validators on :mod:`pancake_engine.types`. This
module adds the remaining business rules that aren't expressible in the
pydantic schema directly.

Engine 0.3 is correctness-first, not TS parity. See architecture §Validation.
"""

from __future__ import annotations

from ..types import EvidenceSpec
from .verdict import ValidationVerdict

__all__ = ["validate_spec"]

SUPPORTED_SIZING_MODES = {"fixed_fraction"}  # PR-1 ships fixed_fraction only


def validate_spec(spec: EvidenceSpec) -> ValidationVerdict:
    """Business-rule checks on an already-pydantic-loaded ``EvidenceSpec``."""
    v = ValidationVerdict()

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
