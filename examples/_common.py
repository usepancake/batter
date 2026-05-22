"""Shared helpers for example regen / run scripts.

Examples are domain-specific by design — they live OUTSIDE ``pancake_engine/``.
The ``test_no_domain_leak`` guard enforces that engine code never imports from
``examples/`` and never references example-specific tokens.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


SCHEMA_COLUMNS = [
    {"name": "mkt",     "type": "string", "semantic_role": "market_link"},
    {"name": "dec_ts",  "type": "int",    "semantic_role": "decision_time"},
    {"name": "res_ts",  "type": "int",    "semantic_role": "resolution_time"},
    {"name": "price",   "type": "number", "semantic_role": "entry_price", "range": [0, 1]},
    {"name": "outcome", "type": "int",    "semantic_role": "resolved_outcome_numeric"},
    {"name": "alpha",   "type": "number", "semantic_role": "feature"},
    {"name": "target",  "type": "int",    "semantic_role": "feature"},
]


def base_spec(*, name: str, side: str = "YES", sizing_value: float = 0.05,
              slip_bps: float = 0, fee_bps: float = 0, starting_capital: float = 10000.0,
              entry_alpha_gte: float = 2.0) -> dict[str, Any]:
    return {
        "spec_family": "pancake-evidence-spec",
        "spec_version": "0.1",
        "name": name,
        "schema_requirements": {"required_columns": SCHEMA_COLUMNS},
        "strategy": {
            "side": side,
            "entry": {"when": {"feature": "alpha", "gte": entry_alpha_gte}},
            "yes_payoff": {"when": {"feature_equal": {"a": "target", "b": "outcome"}}},
            "sizing": {"mode": "fixed_fraction", "value": sizing_value},
        },
        "costs": {"slippage_bps": slip_bps, "fee_bps": fee_bps},
        "starting_capital": starting_capital,
    }


def base_dataset(*, dataset_id: str, rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Build a committed-shape dataset dict.

    ``label`` is a free-form note (e.g., 'synthetic', 'hybrid', 'real') stored
    under ``provenance.label`` for traceability.
    """
    return {
        "id": dataset_id,
        "schema": {"columns": SCHEMA_COLUMNS},
        "schema_sha256": "0" * 64,
        "storage_mode": "inline",
        "rows_inline": rows,
        "rows_sha256": "0" * 64,
        "row_count": len(rows),
        "provenance": {
            "label": label,
            "feature_construction_verified_no_lookahead": True,
        },
    }
