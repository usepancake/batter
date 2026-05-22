"""Pydantic v2 model loading + cost/capital validators.

Engine 0.3 is correctness-first, not TS parity. Negative slippage / fees and
non-positive starting capital are rejected at spec-load time.
"""

from __future__ import annotations

from typing import Any

import pytest

from pancake_engine.types import EvidenceCosts, EvidenceDataset, EvidenceSpec


def _minimal_spec_dict(**overrides: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "spec_family": "pancake-evidence-spec",
        "spec_version": "0.1",
        "name": "test",
        "schema_requirements": {"required_columns": []},
        "strategy": {
            "side": "YES",
            "entry": {"when": {"all_of": []}},
            "yes_payoff": {"when": {"all_of": []}},
            "sizing": {"mode": "fixed_fraction", "value": 0.05},
        },
        "costs": {"slippage_bps": 50, "fee_bps": 0},
        "starting_capital": 1000,
    }
    d.update(overrides)
    return d


def test_evidence_spec_loads() -> None:
    spec = EvidenceSpec.model_validate(_minimal_spec_dict())
    assert spec.starting_capital == 1000
    assert spec.strategy.side == "YES"
    assert spec.costs.slippage_bps == 50
    assert spec.costs.fee_bps == 0


def test_evidence_costs_reject_negative_slippage() -> None:
    with pytest.raises(ValueError, match="E_EVIDENCE_SPEC_INVALID"):
        EvidenceCosts(slippage_bps=-10, fee_bps=0)


def test_evidence_costs_reject_negative_fee() -> None:
    with pytest.raises(ValueError, match="E_EVIDENCE_SPEC_INVALID"):
        EvidenceCosts(slippage_bps=0, fee_bps=-10)


def test_evidence_costs_zero_allowed() -> None:
    c = EvidenceCosts(slippage_bps=0, fee_bps=0)
    assert c.slippage_bps == 0
    assert c.fee_bps == 0


def test_evidence_spec_reject_negative_starting_capital() -> None:
    with pytest.raises(ValueError, match="E_EVIDENCE_SPEC_INVALID"):
        EvidenceSpec.model_validate(_minimal_spec_dict(starting_capital=-100))


def test_evidence_spec_reject_zero_starting_capital() -> None:
    with pytest.raises(ValueError, match="E_EVIDENCE_SPEC_INVALID"):
        EvidenceSpec.model_validate(_minimal_spec_dict(starting_capital=0))


def test_evidence_dataset_loads() -> None:
    raw = {
        "id": "ds_1",
        "schema": {
            "columns": [
                {"name": "mkt", "type": "string", "semantic_role": "market_link"},
                {"name": "dec_ts", "type": "int", "semantic_role": "decision_time"},
            ]
        },
        "schema_sha256": "deadbeef",
        "storage_mode": "inline",
        "rows_inline": [],
        "rows_sha256": "deadbeef",
        "row_count": 0,
    }
    d = EvidenceDataset.model_validate(raw)
    assert d.id == "ds_1"
    assert d.dataset_schema.columns[0].name == "mkt"
    assert d.storage_mode == "inline"
    assert d.row_count == 0


def test_evidence_dataset_accepts_extra_fields() -> None:
    """Provenance, owner_id, and other extra fields pass through without breaking load."""
    raw = {
        "id": "ds_1",
        "owner_id": "user_42",
        "created_at": "2026-05-22T00:00:00Z",
        "provenance": {"source_type": "weather"},
        "schema": {
            "columns": [
                {"name": "mkt", "type": "string", "semantic_role": "market_link"},
            ]
        },
        "schema_sha256": "deadbeef",
        "storage_mode": "inline",
        "rows_inline": [],
        "rows_sha256": "deadbeef",
        "row_count": 0,
    }
    d = EvidenceDataset.model_validate(raw)
    assert d.id == "ds_1"
