"""Pydantic v2 models for the I/O boundary of Pancake Engine 0.3.

PR-0 ships minimal shapes — enough to load an ``EvidenceDataset`` and an
``EvidenceSpec`` from JSON and to enforce the cost / capital validators that
prevent the negative-slippage / negative-fee / non-positive starting-capital
classes of dishonest backtest. Deeper semantic validation (schema/spec
alignment, row invariants, condition AST) lands in PR-1
(``pancake_engine/validate/``).

Engine 0.3 is correctness-first, not TS parity. Known TS divergences are
documented in docs/math-audit-0.4.md.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "ColumnType",
    "SemanticRole",
    "EvidenceColumn",
    "EvidenceSchema",
    "EvidenceDataset",
    "EvidenceSizing",
    "EvidenceCosts",
    "EvidenceStrategy",
    "EvidenceColumnRequirement",
    "EvidenceSchemaRequirements",
    "EvidenceSpec",
]

ColumnType = Literal["string", "int", "number", "bool"]
SemanticRole = Literal[
    "market_link",
    "decision_time",
    "resolution_time",
    "entry_price",
    "resolved_outcome_numeric",
    "feature",
]


class EvidenceColumn(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    type: ColumnType
    semantic_role: SemanticRole
    range: Optional[tuple[float, float]] = None


class EvidenceSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    columns: list[EvidenceColumn]


class EvidenceDataset(BaseModel):
    """Minimal model for the TS ``EvidenceDatasetRecord`` at load boundary.

    Extra fields (``owner_id``, ``created_at``, ``provenance``, etc.) are
    accepted and preserved without strict validation in PR-0.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str
    # ``schema`` collides with BaseModel.schema() in pydantic v1; we keep the
    # field internally as ``dataset_schema`` and use a JSON alias.
    dataset_schema: EvidenceSchema = Field(alias="schema")
    schema_sha256: str
    storage_mode: Literal["inline", "pointer"]
    rows_inline: Optional[list[dict[str, Any]]] = None
    rows_sha256: str
    row_count: int


class EvidenceSizing(BaseModel):
    model_config = ConfigDict(extra="allow")

    mode: Literal["fixed_fraction", "fixed_notional"]
    value: float


class EvidenceCosts(BaseModel):
    """Cost block with non-negativity validators.

    Negative ``slippage_bps`` produces favorable fills (free money);
    negative ``fee_bps`` produces rebates (also free money). Engine 0.3
    rejects both at spec-load.
    """

    model_config = ConfigDict(extra="forbid")

    slippage_bps: float
    fee_bps: float

    @field_validator("slippage_bps")
    @classmethod
    def _slip_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(
                f"E_EVIDENCE_SPEC_INVALID: slippage_bps must be >= 0 (got {v}); "
                "negative slippage represents favorable fills (free money)"
            )
        return v

    @field_validator("fee_bps")
    @classmethod
    def _fee_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(
                f"E_EVIDENCE_SPEC_INVALID: fee_bps must be >= 0 (got {v}); "
                "negative fees represent rebates (free money)"
            )
        return v


class EvidenceStrategy(BaseModel):
    model_config = ConfigDict(extra="allow")

    side: Literal["YES", "NO"]
    # condition AST validated in PR-1
    entry: dict[str, Any]
    yes_payoff: dict[str, Any]
    sizing: EvidenceSizing
    # 0.8: optional benchmark request (spec v0.2 subset). {"kind": "buy_and_hold"}.
    # None default + exclude_none serialization → specs without it hash identically
    # to pre-0.8 specs. Convention: NO-FILTER — same side/sizing/costs on every
    # candidate row, isolating the entry condition's selection value.
    baseline: Optional[dict[str, Any]] = None


class EvidenceColumnRequirement(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    type: ColumnType
    semantic_role: SemanticRole
    range: Optional[tuple[float, float]] = None


class EvidenceSchemaRequirements(BaseModel):
    model_config = ConfigDict(extra="allow")

    required_columns: list[EvidenceColumnRequirement]


class EvidenceSpec(BaseModel):
    """Minimal model for the TS ``EvidenceSpecV01`` / ``CompiledEvidenceSpec``."""

    model_config = ConfigDict(extra="allow")

    spec_family: Literal["pancake-evidence-spec"]
    spec_version: Literal["0.1"]
    name: str
    evidence_dataset_id: Optional[str] = None
    schema_requirements: EvidenceSchemaRequirements
    strategy: EvidenceStrategy
    costs: EvidenceCosts
    starting_capital: float

    @field_validator("starting_capital")
    @classmethod
    def _starting_capital_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(
                f"E_EVIDENCE_SPEC_INVALID: starting_capital must be > 0 (got {v})"
            )
        return v
