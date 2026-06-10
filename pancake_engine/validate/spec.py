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
from ..fills.registry import resolve as _resolve_fill_model
from ..types import EvidenceSpec
from .verdict import ValidationVerdict

__all__ = ["validate_spec"]

SUPPORTED_SIZING_MODES = {"fixed_fraction"}  # PR-1 ships fixed_fraction only
_PAPER_GUARD_ALLOWED_KEYS = frozenset({"max_drawdown_pct", "max_consecutive_losses", "cooldown_bars"})


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
    exit_when: dict[str, Any] = {}
    if isinstance(spec.strategy.exit, dict):
        w = spec.strategy.exit.get("when")
        if isinstance(w, dict):
            exit_when = w
    referenced = (
        extract_referenced_columns(_when(spec.strategy.entry))
        | extract_referenced_columns(_when(spec.strategy.yes_payoff))
        | extract_referenced_columns(exit_when)
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

    # 0.8 baseline block: only the locked spec v0.2 shape is accepted.
    if spec.strategy.baseline is not None:
        kind = (
            spec.strategy.baseline.get("kind")
            if isinstance(spec.strategy.baseline, dict)
            else None
        )
        if kind != "buy_and_hold":
            v.add_error(
                "E_EVIDENCE_SPEC_INVALID",
                f"strategy.baseline.kind must be 'buy_and_hold' (got {kind!r})",
                field="strategy.baseline",
            )

    # 0.9 paper_guard block: validate shape + values.
    if spec.strategy.paper_guard is not None:
        pg = spec.strategy.paper_guard
        if not isinstance(pg, dict) or not pg:
            v.add_error(
                "E_EVIDENCE_SPEC_INVALID",
                "strategy.paper_guard must be a non-empty dict",
                field="strategy.paper_guard",
            )
        else:
            unknown_keys = sorted(set(pg.keys()) - _PAPER_GUARD_ALLOWED_KEYS)
            if unknown_keys:
                v.add_error(
                    "E_EVIDENCE_SPEC_INVALID",
                    f"strategy.paper_guard contains unknown key(s): {unknown_keys}; "
                    f"allowed: {sorted(_PAPER_GUARD_ALLOWED_KEYS)}",
                    field="strategy.paper_guard",
                )
            if "max_drawdown_pct" in pg:
                mdp = pg["max_drawdown_pct"]
                if not isinstance(mdp, (int, float)) or isinstance(mdp, bool) or not (0 < mdp <= 1):
                    v.add_error(
                        "E_EVIDENCE_SPEC_INVALID",
                        f"strategy.paper_guard.max_drawdown_pct must be a float in (0, 1] "
                        f"(got {mdp!r})",
                        field="strategy.paper_guard.max_drawdown_pct",
                    )
            if "max_consecutive_losses" in pg:
                mcl = pg["max_consecutive_losses"]
                if not isinstance(mcl, int) or isinstance(mcl, bool) or mcl < 1:
                    v.add_error(
                        "E_EVIDENCE_SPEC_INVALID",
                        f"strategy.paper_guard.max_consecutive_losses must be an int >= 1 "
                        f"(got {mcl!r})",
                        field="strategy.paper_guard.max_consecutive_losses",
                    )
            if "cooldown_bars" in pg:
                cb = pg["cooldown_bars"]
                if not isinstance(cb, int) or isinstance(cb, bool) or cb < 1:
                    v.add_error(
                        "E_EVIDENCE_SPEC_INVALID",
                        f"strategy.paper_guard.cooldown_bars must be an int >= 1 "
                        f"(got {cb!r})",
                        field="strategy.paper_guard.cooldown_bars",
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

    # 0.9.0 Wave 2: fill_model validation.
    # When present: name/version must resolve in the registry; params must be
    # valid per-model.  static_bps@1 takes NO params → reject non-empty dict.
    if spec.costs.fill_model is not None:
        fm = spec.costs.fill_model
        try:
            _resolve_fill_model(fm.name, fm.version)
        except ValueError as exc:
            v.add_error(
                "E_EVIDENCE_SPEC_INVALID",
                str(exc),
                field="costs.fill_model",
            )
        else:
            # Per-model param validation: static_bps@1 accepts no params.
            if fm.name == "static_bps" and fm.version == 1:
                if fm.params:
                    v.add_error(
                        "E_EVIDENCE_SPEC_INVALID",
                        f"costs.fill_model static_bps@1 takes no params; "
                        f"got: {sorted(fm.params.keys())}",
                        field="costs.fill_model.params",
                    )

    return v
