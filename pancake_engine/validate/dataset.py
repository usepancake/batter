"""Dataset validation: schema match + row invariants.

Port of TS ``preflightSchemaMatch`` + ``preflightRowInvariants``
(``lib/evidence-runner/runner.ts``), extended with structured warnings.
"""

from __future__ import annotations

from typing import Any

from ..types import EvidenceDataset, EvidenceSpec, SemanticRole
from .verdict import ValidationVerdict

__all__ = ["validate_dataset", "RoleLookup"]

REQUIRED_UNIQUE_ROLES: tuple[SemanticRole, ...] = (
    "market_link",
    "decision_time",
    "resolution_time",
    "entry_price",
    "resolved_outcome_numeric",
)


class RoleLookup(dict[str, str]):
    """Maps each required semantic role to its column name."""


def validate_dataset(dataset: EvidenceDataset, spec: EvidenceSpec) -> tuple[ValidationVerdict, RoleLookup]:
    """Validate dataset against spec.

    Returns the verdict plus the ``role -> column_name`` lookup, which the
    runner needs to read row values. If the verdict is not ``ok``, the
    lookup may be incomplete; callers must check ``verdict.ok`` first.
    """
    v = ValidationVerdict()
    lookup = RoleLookup()

    if dataset.storage_mode != "inline":
        v.add_error(
            "E_EVIDENCE_INLINE_REQUIRED",
            f"PR-1 engine only supports inline-stored datasets; got storage_mode="
            f"{dataset.storage_mode!r}",
        )
        return v, lookup

    rows = dataset.rows_inline
    if rows is None or len(rows) == 0:
        v.add_error(
            "E_EVIDENCE_ROWS_MISSING",
            f"dataset {dataset.id!r} has zero rows; cannot run a strategy",
        )
        return v, lookup

    # --- Schema match against spec.schema_requirements ---
    dataset_cols = {c.name: c for c in dataset.dataset_schema.columns}
    for req in spec.schema_requirements.required_columns:
        have = dataset_cols.get(req.name)
        if have is None:
            v.add_error(
                "E_EVIDENCE_SCHEMA_MISMATCH",
                f"spec requires column {req.name!r} (role={req.semantic_role}); "
                "dataset does not declare it",
                column=req.name,
            )
            continue
        if have.type != req.type:
            v.add_error(
                "E_EVIDENCE_SCHEMA_MISMATCH",
                f"column {req.name!r} type mismatch: spec={req.type}, dataset={have.type}",
                column=req.name,
            )
        if have.semantic_role != req.semantic_role:
            v.add_error(
                "E_EVIDENCE_SCHEMA_MISMATCH",
                f"column {req.name!r} role mismatch: spec={req.semantic_role}, "
                f"dataset={have.semantic_role}",
                column=req.name,
            )

    # --- Each required-unique role must resolve to exactly one spec column ---
    role_to_col: dict[str, str] = {}
    for req in spec.schema_requirements.required_columns:
        if req.semantic_role == "feature":
            continue
        if req.semantic_role in role_to_col:
            v.add_error(
                "E_EVIDENCE_SCHEMA_MISMATCH",
                f"spec declares semantic_role {req.semantic_role!r} on multiple columns",
                column=req.name,
            )
        else:
            role_to_col[req.semantic_role] = req.name

    for role in REQUIRED_UNIQUE_ROLES:
        if role not in role_to_col:
            v.add_error(
                "E_EVIDENCE_SCHEMA_MISMATCH",
                f"spec must declare exactly one column with semantic_role {role!r}",
                role=role,
            )

    if not v.ok:
        # Cannot proceed to row invariants without a role lookup
        return v, lookup

    for role, col in role_to_col.items():
        lookup[role] = col

    # --- Row invariants ---
    seen_market_decisions: dict[str, set[Any]] = {}
    required_cols = spec.schema_requirements.required_columns

    for i, row in enumerate(rows):
        for req in required_cols:
            val = row.get(req.name)
            if val is None:
                # `resolved_outcome_numeric` may be null (unresolved row).
                # The runner skips such rows with UNRESOLVED_ROW_SKIPPED warning.
                # Per architecture §observation_time rule: a dataset with any null
                # resolved_outcome_numeric requires config.observation_time to be set.
                if req.semantic_role == "resolved_outcome_numeric":
                    continue
                v.add_error(
                    "E_EVIDENCE_FEATURE_MISSING",
                    f"row {i}: required column {req.name!r} is missing or null",
                    row_index=i, column=req.name,
                )
                continue
            if not _type_matches(req.type, val):
                v.add_error(
                    "E_EVIDENCE_TYPE",
                    f"row {i}: column {req.name!r} value {val!r} does not match declared "
                    f"type {req.type}",
                    row_index=i, column=req.name,
                )
                continue
            if req.range is not None and isinstance(val, (int, float)) and not isinstance(val, bool):
                low, high = req.range
                if val < low or val > high:
                    v.add_error(
                        "E_EVIDENCE_RANGE",
                        f"row {i}: column {req.name!r} value {val} outside declared range "
                        f"[{low}, {high}]",
                        row_index=i, column=req.name,
                    )

        # Lookahead invariant: decision_time < resolution_time strictly
        if "decision_time" in lookup and "resolution_time" in lookup:
            dec = row.get(lookup["decision_time"])
            res = row.get(lookup["resolution_time"])
            if isinstance(dec, (int, float)) and isinstance(res, (int, float)):
                if dec >= res:
                    v.add_error(
                        "E_EVIDENCE_LOOKAHEAD",
                        f"row {i}: decision_time ({dec}) must be strictly less than "
                        f"resolution_time ({res})",
                        row_index=i,
                    )

        # Monotonicity: no duplicate (market_link, decision_time) pairs
        if "market_link" in lookup and "decision_time" in lookup:
            mkt = row.get(lookup["market_link"])
            dec = row.get(lookup["decision_time"])
            if mkt is not None and dec is not None:
                seen = seen_market_decisions.setdefault(str(mkt), set())
                if dec in seen:
                    v.add_error(
                        "E_EVIDENCE_MONOTONICITY",
                        f"row {i}: duplicate (market_link={mkt!r}, decision_time={dec}) — "
                        "must be unique",
                        row_index=i, market_link=str(mkt), decision_time=dec,
                    )
                seen.add(dec)

    return v, lookup


def _type_matches(declared: str, value: Any) -> bool:
    if declared == "string":
        return isinstance(value, str)
    if declared == "bool":
        return isinstance(value, bool)
    if declared == "int":
        # bool is subclass of int; exclude it
        return isinstance(value, int) and not isinstance(value, bool)
    if declared == "number":
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            # int counts as number; reject NaN/Inf
            if isinstance(value, float):
                import math
                if math.isnan(value) or math.isinf(value):
                    return False
            return True
    return False
