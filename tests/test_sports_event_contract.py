"""Tests for SportsEventContract + validate_sports_dataset (Wave E, 0.10.0).

TDD: written alongside implementation.

Pattern mirrors test_macro_signal_contract.py.

Coverage:
1. SportsEventContract — shape and domain registry
2. validate_sports_dataset — happy path
3. Missing column rejection
4. Type error rejection
5. Range rejection (entry_price out of (0,1), outcome not 0/1)
6. Empty rows rejection
7. Multiple errors accumulated
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pancake_engine.contracts import (
    DatasetContract,
    SportsEventContract,
    contract_for_domain,
)
from pancake_engine.validate.sports import validate_sports_dataset
from pancake_engine.validate.verdict import ValidationVerdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(
    market_link: str = "https://example.com/event/1",
    decision_time: int = 1000,
    resolution_time: int = 2000,
    entry_price: float = 0.6,
    resolved_outcome_numeric: int = 1,
    event_id: str = "evt_001",
    league: str = "NBA",
) -> dict[str, Any]:
    return {
        "market_link": market_link,
        "decision_time": decision_time,
        "resolution_time": resolution_time,
        "entry_price": entry_price,
        "resolved_outcome_numeric": resolved_outcome_numeric,
        "event_id": event_id,
        "league": league,
    }


def _valid_rows(n: int = 1) -> list[dict[str, Any]]:
    return [
        _row(
            market_link=f"https://example.com/event/{i}",
            decision_time=1000 + i * 100,
            resolution_time=2000 + i * 100,
            event_id=f"evt_{i:03d}",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Section 1: SportsEventContract — shape and domain registry
# ---------------------------------------------------------------------------


def test_sports_contract_domain() -> None:
    assert SportsEventContract.domain == "sports_event"


def test_sports_contract_time_model() -> None:
    """time_model is 'event_resolution' — shared with PM."""
    assert SportsEventContract.time_model == "event_resolution"


def test_sports_contract_resolution_semantics() -> None:
    """resolution_semantics is 'binary_payout'."""
    assert SportsEventContract.resolution_semantics == "binary_payout"


def test_sports_contract_fill_reference() -> None:
    assert SportsEventContract.fill_reference == "entry_price_col"


def test_sports_contract_required_roles() -> None:
    role_names = {r.name for r in SportsEventContract.required_roles}
    expected = {
        "market_link", "decision_time", "resolution_time",
        "entry_price", "resolved_outcome_numeric", "event_id", "league",
    }
    assert role_names == expected


def test_sports_contract_role_types() -> None:
    role_map = {r.name: r for r in SportsEventContract.required_roles}
    assert role_map["market_link"].col_type == "string"
    assert role_map["decision_time"].col_type == "int"
    assert role_map["resolution_time"].col_type == "int"
    assert role_map["entry_price"].col_type == "number"
    assert role_map["resolved_outcome_numeric"].col_type == "int"
    assert role_map["event_id"].col_type == "string"
    assert role_map["league"].col_type == "string"


def test_sports_contract_outcome_not_nullable() -> None:
    role_map = {r.name: r for r in SportsEventContract.required_roles}
    assert role_map["resolved_outcome_numeric"].nullable is False


def test_sports_contract_frozen() -> None:
    with pytest.raises((AttributeError, TypeError)):
        SportsEventContract.domain = "mutated"  # type: ignore[misc]


def test_sports_contract_is_dataset_contract() -> None:
    assert isinstance(SportsEventContract, DatasetContract)


def test_contract_for_domain_sports_event() -> None:
    assert contract_for_domain("sports_event") is SportsEventContract


def test_contract_for_domain_unknown_raises() -> None:
    with pytest.raises(KeyError, match="sports_unknown"):
        contract_for_domain("sports_unknown")


# ---------------------------------------------------------------------------
# Section 2: validate_sports_dataset — happy path
# ---------------------------------------------------------------------------


def test_valid_single_row_passes() -> None:
    verdict = validate_sports_dataset(_valid_rows(1))
    assert verdict.ok, verdict.errors


def test_valid_multiple_rows_pass() -> None:
    verdict = validate_sports_dataset(_valid_rows(5))
    assert verdict.ok, verdict.errors


def test_outcome_zero_passes() -> None:
    verdict = validate_sports_dataset([_row(resolved_outcome_numeric=0)])
    assert verdict.ok, verdict.errors


def test_outcome_one_passes() -> None:
    verdict = validate_sports_dataset([_row(resolved_outcome_numeric=1)])
    assert verdict.ok, verdict.errors


def test_entry_price_just_above_zero_passes() -> None:
    verdict = validate_sports_dataset([_row(entry_price=0.001)])
    assert verdict.ok, verdict.errors


def test_entry_price_just_below_one_passes() -> None:
    verdict = validate_sports_dataset([_row(entry_price=0.999)])
    assert verdict.ok, verdict.errors


# ---------------------------------------------------------------------------
# Section 3: Missing column rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing_col", [
    "market_link", "decision_time", "resolution_time",
    "entry_price", "resolved_outcome_numeric", "event_id", "league",
])
def test_missing_column_rejected(missing_col: str) -> None:
    row = _row()
    del row[missing_col]
    verdict = validate_sports_dataset([row])
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_SCHEMA_MISMATCH" in codes


def test_null_column_rejected() -> None:
    row = _row()
    row["league"] = None  # type: ignore[assignment]
    verdict = validate_sports_dataset([row])
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_SCHEMA_MISMATCH" in codes


# ---------------------------------------------------------------------------
# Section 4: Type error rejection
# ---------------------------------------------------------------------------


def test_non_string_market_link_rejected() -> None:
    verdict = validate_sports_dataset([_row(market_link=12345)])  # type: ignore[arg-type]
    assert not verdict.ok
    assert "E_EVIDENCE_TYPE" in {e.code for e in verdict.errors}


def test_bool_decision_time_rejected() -> None:
    """bool is a subclass of int — must be rejected."""
    verdict = validate_sports_dataset([_row(decision_time=True)])  # type: ignore[arg-type]
    assert not verdict.ok
    assert "E_EVIDENCE_TYPE" in {e.code for e in verdict.errors}


def test_float_decision_time_rejected() -> None:
    verdict = validate_sports_dataset([_row(decision_time=1000.5)])  # type: ignore[arg-type]
    assert not verdict.ok
    assert "E_EVIDENCE_TYPE" in {e.code for e in verdict.errors}


def test_bool_resolution_time_rejected() -> None:
    verdict = validate_sports_dataset([_row(resolution_time=False)])  # type: ignore[arg-type]
    assert not verdict.ok
    assert "E_EVIDENCE_TYPE" in {e.code for e in verdict.errors}


def test_string_entry_price_rejected() -> None:
    verdict = validate_sports_dataset([_row(entry_price="0.6")])  # type: ignore[arg-type]
    assert not verdict.ok
    assert "E_EVIDENCE_TYPE" in {e.code for e in verdict.errors}


def test_bool_entry_price_rejected() -> None:
    verdict = validate_sports_dataset([_row(entry_price=True)])  # type: ignore[arg-type]
    assert not verdict.ok
    assert "E_EVIDENCE_TYPE" in {e.code for e in verdict.errors}


def test_float_outcome_rejected() -> None:
    verdict = validate_sports_dataset([_row(resolved_outcome_numeric=1.0)])  # type: ignore[arg-type]
    assert not verdict.ok
    assert "E_EVIDENCE_TYPE" in {e.code for e in verdict.errors}


def test_bool_outcome_rejected() -> None:
    verdict = validate_sports_dataset([_row(resolved_outcome_numeric=True)])  # type: ignore[arg-type]
    assert not verdict.ok
    assert "E_EVIDENCE_TYPE" in {e.code for e in verdict.errors}


def test_non_string_event_id_rejected() -> None:
    verdict = validate_sports_dataset([_row(event_id=999)])  # type: ignore[arg-type]
    assert not verdict.ok
    assert "E_EVIDENCE_TYPE" in {e.code for e in verdict.errors}


def test_non_string_league_rejected() -> None:
    verdict = validate_sports_dataset([_row(league=42)])  # type: ignore[arg-type]
    assert not verdict.ok
    assert "E_EVIDENCE_TYPE" in {e.code for e in verdict.errors}


# ---------------------------------------------------------------------------
# Section 5: Range rejection
# ---------------------------------------------------------------------------


def test_entry_price_zero_rejected() -> None:
    verdict = validate_sports_dataset([_row(entry_price=0.0)])
    assert not verdict.ok
    assert "E_EVIDENCE_RANGE" in {e.code for e in verdict.errors}


def test_entry_price_one_rejected() -> None:
    verdict = validate_sports_dataset([_row(entry_price=1.0)])
    assert not verdict.ok
    assert "E_EVIDENCE_RANGE" in {e.code for e in verdict.errors}


def test_entry_price_above_one_rejected() -> None:
    verdict = validate_sports_dataset([_row(entry_price=1.5)])
    assert not verdict.ok
    assert "E_EVIDENCE_RANGE" in {e.code for e in verdict.errors}


def test_entry_price_negative_rejected() -> None:
    verdict = validate_sports_dataset([_row(entry_price=-0.1)])
    assert not verdict.ok
    assert "E_EVIDENCE_RANGE" in {e.code for e in verdict.errors}


def test_entry_price_nan_rejected() -> None:
    verdict = validate_sports_dataset([_row(entry_price=math.nan)])
    assert not verdict.ok
    assert "E_EVIDENCE_RANGE" in {e.code for e in verdict.errors}


def test_entry_price_inf_rejected() -> None:
    verdict = validate_sports_dataset([_row(entry_price=math.inf)])
    assert not verdict.ok
    assert "E_EVIDENCE_RANGE" in {e.code for e in verdict.errors}


def test_outcome_two_rejected() -> None:
    verdict = validate_sports_dataset([_row(resolved_outcome_numeric=2)])
    assert not verdict.ok
    assert "E_EVIDENCE_RANGE" in {e.code for e in verdict.errors}


def test_outcome_negative_one_rejected() -> None:
    verdict = validate_sports_dataset([_row(resolved_outcome_numeric=-1)])
    assert not verdict.ok
    assert "E_EVIDENCE_RANGE" in {e.code for e in verdict.errors}


# ---------------------------------------------------------------------------
# Section 6: Empty rows rejection
# ---------------------------------------------------------------------------


def test_empty_rows_rejected() -> None:
    verdict = validate_sports_dataset([])
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_ROWS_MISSING" in codes


# ---------------------------------------------------------------------------
# Section 7: Multiple errors accumulated
# ---------------------------------------------------------------------------


def test_multiple_violations_all_reported() -> None:
    """Validator accumulates all errors, not just the first."""
    rows = [
        _row(entry_price=1.5, resolved_outcome_numeric=3),  # two range errors
    ]
    verdict = validate_sports_dataset(rows)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_RANGE" in codes
    # Both entry_price and outcome should be reported
    range_errors = [e for e in verdict.errors if e.code == "E_EVIDENCE_RANGE"]
    assert len(range_errors) >= 2


def test_type_and_range_errors_both_reported() -> None:
    """Two rows: one with type error on market_link, one with range error on outcome.

    Both errors accumulate across rows (fail-slow, not fail-fast).
    A type error in one row does skip the range check for THAT row (row_ok guard),
    so we use separate rows to confirm accumulation across rows.
    """
    rows = [
        _row(resolved_outcome_numeric=5),      # row 0: range error on outcome
        {**_row(), "market_link": 999},         # row 1: type error on market_link
    ]
    verdict = validate_sports_dataset(rows)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_TYPE" in codes
    assert "E_EVIDENCE_RANGE" in codes
