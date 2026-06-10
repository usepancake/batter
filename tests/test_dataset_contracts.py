"""Tests for the DatasetContract Seam (Wave 2, Part B).

TDD: these tests were written BEFORE the implementation.

Hard constraints:
- validate_dataset's observable behavior (error codes, messages, warning
  emission, role_lookup) must not change — the existing validation tests
  are the refactor's safety net.
- Contract registry resolves 'prediction_market' for 'pancake-evidence-spec'.
- Unknown spec_family already fails at pydantic (Literal type); assert that
  stays true and the contract lookup is total for families that pass pydantic.
"""

from __future__ import annotations

import pytest

from pancake_engine.contracts import (
    DatasetContract,
    PredictionMarketContract,
    contract_for_spec_family,
)
from pancake_engine.types import EvidenceSpec

from ._runner_helpers import SCHEMA_COLUMNS, make_dataset, make_spec, row


# ---------------------------------------------------------------------------
# Contract registry
# ---------------------------------------------------------------------------


def test_contract_registry_pm_resolves() -> None:
    """'pancake-evidence-spec' resolves to the PredictionMarketContract."""
    contract = contract_for_spec_family("pancake-evidence-spec")
    assert contract is PredictionMarketContract
    assert contract.domain == "prediction_market"


def test_pm_contract_domain_fields() -> None:
    """PredictionMarketContract has the expected domain metadata."""
    assert PredictionMarketContract.domain == "prediction_market"
    assert PredictionMarketContract.time_model == "event_resolution"
    assert PredictionMarketContract.resolution_semantics == "binary_payout"
    assert PredictionMarketContract.fill_reference == "entry_price_col"


def test_pm_contract_required_roles_complete() -> None:
    """PredictionMarketContract declares all five PM required roles."""
    role_names = {r.name for r in PredictionMarketContract.required_roles}
    assert role_names == {
        "market_link",
        "decision_time",
        "resolution_time",
        "entry_price",
        "resolved_outcome_numeric",
    }


def test_unknown_spec_family_rejected_by_pydantic() -> None:
    """Unknown spec_family fails at pydantic (Literal type); never reaches contract lookup."""
    with pytest.raises(Exception):  # pydantic ValidationError
        EvidenceSpec.model_validate({
            "spec_family": "unknown-family-xyz",
            "spec_version": "0.1",
            "name": "x",
            "schema_requirements": {"required_columns": SCHEMA_COLUMNS},
            "strategy": {
                "side": "YES",
                "entry": {"when": {"feature": "alpha", "gte": 2.0}},
                "yes_payoff": {"when": {"feature_equal": {"a": "target", "b": "outcome"}}},
                "sizing": {"mode": "fixed_fraction", "value": 0.1},
            },
            "costs": {"slippage_bps": 0.0, "fee_bps": 0.0},
            "starting_capital": 1000.0,
        })


# ---------------------------------------------------------------------------
# Behavioral parity: validate_dataset after refactor produces identical output
# ---------------------------------------------------------------------------


def test_valid_dataset_still_passes() -> None:
    """A well-formed dataset still passes after contract extraction."""
    from pancake_engine.validate import validate_dataset

    spec = make_spec()
    dataset = make_dataset([
        row(mkt="m/1", dec_ts=100, res_ts=200, price=0.6, outcome=1),
        row(mkt="m/2", dec_ts=300, res_ts=400, price=0.4, outcome=0),
    ])
    verdict, lookup = validate_dataset(dataset, spec)
    assert verdict.ok
    assert set(lookup.keys()) >= {"market_link", "decision_time", "resolution_time",
                                   "entry_price", "resolved_outcome_numeric"}


def test_lookahead_invariant_still_caught() -> None:
    """dec_ts >= res_ts still emits E_EVIDENCE_LOOKAHEAD."""
    from pancake_engine.validate import validate_dataset

    spec = make_spec()
    dataset = make_dataset([
        row(mkt="m/1", dec_ts=200, res_ts=100, price=0.6, outcome=1),  # lookahead
    ])
    verdict, _ = validate_dataset(dataset, spec)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_LOOKAHEAD" in codes


def test_monotonicity_still_caught() -> None:
    """Duplicate (market_link, decision_time) still emits E_EVIDENCE_MONOTONICITY."""
    from pancake_engine.validate import validate_dataset

    spec = make_spec()
    dataset = make_dataset([
        row(mkt="m/1", dec_ts=100, res_ts=200, price=0.6, outcome=1),
        row(mkt="m/1", dec_ts=100, res_ts=300, price=0.5, outcome=0),  # dup
    ])
    verdict, _ = validate_dataset(dataset, spec)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_MONOTONICITY" in codes


def test_entry_price_range_invariant_still_caught() -> None:
    """entry_price outside [0,1] (no explicit range) still emits E_EVIDENCE_RANGE."""
    from pancake_engine.validate import validate_dataset

    # Use a spec/dataset without an explicit range on entry_price so the
    # contract-driven 0.7.2 fallback fires.
    spec = make_spec()
    # Replace the schema columns with one that has no range on entry_price
    from pancake_engine.types import EvidenceDataset
    no_range_cols = [
        c if c["name"] != "price" else {**c, "range": None}
        for c in SCHEMA_COLUMNS
    ]
    no_range_cols_clean = [{k: v for k, v in c.items() if v is not None} for c in no_range_cols]

    # Also adjust the spec's schema_requirements to drop the range on price
    from pancake_engine.types import EvidenceSpec
    spec_no_range = EvidenceSpec.model_validate({
        "spec_family": "pancake-evidence-spec",
        "spec_version": "0.1",
        "name": "test",
        "schema_requirements": {"required_columns": no_range_cols_clean},
        "strategy": {
            "side": "YES",
            "entry": {"when": {"feature": "alpha", "gte": 2.0}},
            "yes_payoff": {"when": {"feature_equal": {"a": "target", "b": "outcome"}}},
            "sizing": {"mode": "fixed_fraction", "value": 0.1},
        },
        "costs": {"slippage_bps": 0.0, "fee_bps": 0.0},
        "starting_capital": 1000.0,
    })

    dataset = EvidenceDataset.model_validate({
        "id": "ds_no_range",
        "schema": {"columns": no_range_cols_clean},
        "schema_sha256": "0" * 64,
        "storage_mode": "inline",
        "rows_inline": [
            {"mkt": "m/1", "dec_ts": 100, "res_ts": 200, "price": 1.7,
             "outcome": 1, "alpha": 3.0, "target": 1},
        ],
        "rows_sha256": "0" * 64,
        "row_count": 1,
    })

    verdict, _ = validate_dataset(dataset, spec_no_range)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_RANGE" in codes


def test_contract_is_frozen_dataclass() -> None:
    """DatasetContract is frozen — immutable at runtime."""
    with pytest.raises((AttributeError, TypeError)):
        PredictionMarketContract.domain = "mutated"  # type: ignore[misc]
