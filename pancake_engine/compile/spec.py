"""Compile an ``EvidenceSpec`` to a runner-ready ``CompiledSpec``.

The ``compiled_spec_hash`` is the SHA-256 of the canonicalized **raw** spec
(after pydantic re-emit, with the ``schema`` alias resolved). It is byte-equal
to the TS ``source_spec_hash`` for the same spec content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..fills.registry import FillModel
from ..fills.registry import default_model as _default_fill_model
from ..fills.registry import resolve as _resolve_fill_model
from ..hash import sha256_canonical
from ..types import EvidenceSpec
from .condition import Condition, compile_condition

__all__ = ["CompiledSpec", "compile_spec"]


@dataclass(frozen=True)
class CompiledSpec:
    """Runner-ready spec with compiled condition callables."""

    raw: EvidenceSpec
    compiled_spec_hash: str
    entry_condition: Condition
    yes_payoff_condition: Condition
    # 0.9.0 Wave 2: resolved fill model.  Always set (default = static_bps@1).
    # This field is NOT frozen's concern — it's an object reference, not part
    # of the hash.  The hash comes from the raw spec dict (which includes or
    # excludes fill_model via exclude_none).
    fill_model: FillModel = None  # type: ignore[assignment]  # set in compile_spec

    @property
    def side(self) -> str:
        return self.raw.strategy.side

    @property
    def sizing_value(self) -> float:
        return self.raw.strategy.sizing.value

    @property
    def slippage_bps(self) -> float:
        return self.raw.costs.slippage_bps

    @property
    def fee_bps(self) -> float:
        return self.raw.costs.fee_bps

    @property
    def starting_capital(self) -> float:
        return self.raw.starting_capital


def compile_spec(spec: EvidenceSpec) -> CompiledSpec:
    """Compile condition ASTs and compute ``compiled_spec_hash``.

    The hash is over the canonical form of the *raw* spec dict (post-pydantic
    re-emit, with ``schema`` alias unaliased). This matches the TS
    ``source_spec_hash`` semantics.
    """
    raw_dict = _spec_to_canonical_dict(spec)
    compiled_spec_hash = sha256_canonical(raw_dict)

    entry_when = spec.strategy.entry.get("when")
    yes_payoff_when = spec.strategy.yes_payoff.get("when")
    if not isinstance(entry_when, dict):
        raise ValueError("E_EVIDENCE_SPEC_INVALID: strategy.entry.when must be a condition node dict")
    if not isinstance(yes_payoff_when, dict):
        raise ValueError(
            "E_EVIDENCE_SPEC_INVALID: strategy.yes_payoff.when must be a condition node dict"
        )

    # Resolve fill model: use the spec's explicit reference when present,
    # otherwise default to static_bps@1.  Validation has already confirmed
    # that the name/version pair exists, so resolve() should never raise here.
    if spec.costs.fill_model is not None:
        fm_ref = spec.costs.fill_model
        fill_model = _resolve_fill_model(fm_ref.name, fm_ref.version)
    else:
        fill_model = _default_fill_model()

    return CompiledSpec(
        raw=spec,
        compiled_spec_hash=compiled_spec_hash,
        entry_condition=compile_condition(entry_when),
        yes_payoff_condition=compile_condition(yes_payoff_when),
        fill_model=fill_model,
    )


def _spec_to_canonical_dict(spec: EvidenceSpec) -> dict[str, Any]:
    """Serialize the spec to a plain dict suitable for canonicalize.

    We use ``model_dump(by_alias=True, exclude_none=True)`` so the resulting
    dict has the same keys a user would write in JSON (e.g. ``starting_capital``
    not the internal pydantic alias).
    """
    return spec.model_dump(by_alias=True, exclude_none=True, mode="python")
