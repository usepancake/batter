"""Shared fixture builders for PR-1 runner tests.

Mirrors the schema used by pancake-production/tests/evidence-runner/runner.test.ts
so fixtures are portable to the TS oracle for golden parity tests.
"""

from __future__ import annotations

from typing import Any

from pancake_engine.types import EvidenceDataset, EvidenceSpec

SCHEMA_COLUMNS = [
    {"name": "mkt",     "type": "string", "semantic_role": "market_link"},
    {"name": "dec_ts",  "type": "int",    "semantic_role": "decision_time"},
    {"name": "res_ts",  "type": "int",    "semantic_role": "resolution_time"},
    {"name": "price",   "type": "number", "semantic_role": "entry_price", "range": [0, 1]},
    {"name": "outcome", "type": "int",    "semantic_role": "resolved_outcome_numeric"},
    {"name": "alpha",   "type": "number", "semantic_role": "feature"},
    {"name": "target",  "type": "int",    "semantic_role": "feature"},
]


def make_spec(
    *,
    side: str = "YES",
    sizing_value: float = 0.1,
    slip_bps: float = 0.0,
    fee_bps: float = 0.0,
    entry_when: dict | None = None,
    yes_payoff_when: dict | None = None,
    starting_capital: float = 1000.0,
    paper_guard: dict | None = None,
    exit_when: dict | None = None,
) -> EvidenceSpec:
    strategy: dict = {
        "side": side,
        "entry": {"when": entry_when or {"feature": "alpha", "gte": 2.0}},
        "yes_payoff": {"when": yes_payoff_when or {"feature_equal": {"a": "target", "b": "outcome"}}},
        "sizing": {"mode": "fixed_fraction", "value": sizing_value},
    }
    if paper_guard is not None:
        strategy["paper_guard"] = paper_guard
    if exit_when is not None:
        strategy["exit"] = {"when": exit_when}
    return EvidenceSpec.model_validate({
        "spec_family": "pancake-evidence-spec",
        "spec_version": "0.1",
        "name": "test-spec",
        "evidence_dataset_id": "ev_runner_test",
        "schema_requirements": {"required_columns": SCHEMA_COLUMNS},
        "strategy": strategy,
        "costs": {"slippage_bps": slip_bps, "fee_bps": fee_bps},
        "starting_capital": starting_capital,
    })


def make_dataset(rows: list[dict[str, Any]], *, dataset_id: str = "ds_test") -> EvidenceDataset:
    return EvidenceDataset.model_validate({
        "id": dataset_id,
        "schema": {"columns": SCHEMA_COLUMNS},
        "schema_sha256": "0" * 64,
        "storage_mode": "inline",
        "rows_inline": rows,
        "rows_sha256": "0" * 64,
        "row_count": len(rows),
    })


def row(
    *,
    mkt: str,
    dec_ts: int,
    res_ts: int,
    price: float,
    outcome: int,
    alpha: float = 3.0,
    target: int = 1,
) -> dict[str, Any]:
    return {
        "mkt": mkt, "dec_ts": dec_ts, "res_ts": res_ts,
        "price": price, "outcome": outcome,
        "alpha": alpha, "target": target,
    }
