"""Tests for MacroSignalContract (Wave 4, 0.9.0).

TDD: written before implementation.

MacroSignalContract is a FEATURE-PROVIDER domain.  Reference datasets
(e.g. FRED series) are validated here; the engine never runs them directly —
the platform joins macro columns into evidence rows upstream, and PM specs
reference those columns as ordinary declared features.

Invariants:
- observation_time: int (epoch sec), monotone non-decreasing per series_id
- series_id: string, non-empty
- value: number (finite — NaN / Inf rejected)
- no duplicate (series_id, observation_time) pairs

Error codes used:
  Reused:
    E_EVIDENCE_ROWS_MISSING    — zero rows (same semantics as PM)
    E_EVIDENCE_SCHEMA_MISMATCH — required column/role absent
    E_EVIDENCE_TYPE            — column value type wrong
    E_EVIDENCE_MONOTONICITY    — duplicate (series_id, observation_time)
    E_EVIDENCE_RANGE           — value is NaN or Inf (finite constraint)
  New (no existing code covers it):
    E_REFERENCE_OBSERVATION_TIME_ORDER — observation_time not monotone non-decreasing
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pancake_engine.contracts import (
    DatasetContract,
    MacroSignalContract,
    contract_for_domain,
)
from pancake_engine.validate.macro import validate_reference_dataset
from pancake_engine.validate.verdict import ValidationVerdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(series_id: str, observation_time: int, value: float) -> dict[str, Any]:
    return {"series_id": series_id, "observation_time": observation_time, "value": value}


def _rows(*args: tuple[str, int, float]) -> list[dict[str, Any]]:
    return [_row(s, t, v) for s, t, v in args]


# ---------------------------------------------------------------------------
# 1. MacroSignalContract — shape and registry
# ---------------------------------------------------------------------------


def test_macro_contract_domain() -> None:
    assert MacroSignalContract.domain == "macro_signal"


def test_macro_contract_time_model() -> None:
    """time_model is 'reference_series' — a new value; no engine runner."""
    assert MacroSignalContract.time_model == "reference_series"


def test_macro_contract_resolution_semantics_none() -> None:
    """No binary payout; macro is a feature provider."""
    assert MacroSignalContract.resolution_semantics is None


def test_macro_contract_fill_reference_none() -> None:
    """No fill reference — macro datasets are never traded directly."""
    assert MacroSignalContract.fill_reference is None


def test_macro_contract_required_roles() -> None:
    role_names = {r.name for r in MacroSignalContract.required_roles}
    assert role_names == {"observation_time", "series_id", "value"}


def test_macro_contract_role_types() -> None:
    role_map = {r.name: r for r in MacroSignalContract.required_roles}
    assert role_map["observation_time"].col_type == "int"
    assert role_map["series_id"].col_type == "string"
    assert role_map["value"].col_type == "number"


def test_macro_contract_frozen() -> None:
    with pytest.raises((AttributeError, TypeError)):
        MacroSignalContract.domain = "mutated"  # type: ignore[misc]


def test_contract_for_domain_macro() -> None:
    """contract_for_domain('macro_signal') returns MacroSignalContract."""
    assert contract_for_domain("macro_signal") is MacroSignalContract


def test_contract_for_domain_unknown_raises() -> None:
    with pytest.raises(KeyError, match="macro_unknown"):
        contract_for_domain("macro_unknown")


# ---------------------------------------------------------------------------
# 2. validate_reference_dataset — happy path
# ---------------------------------------------------------------------------


def test_valid_single_series_passes() -> None:
    verdict = validate_reference_dataset(_rows(
        ("UNRATE", 1000, 3.5),
        ("UNRATE", 2000, 3.6),
        ("UNRATE", 3000, 3.4),
    ))
    assert verdict.ok, verdict.errors


def test_valid_multi_series_passes() -> None:
    """Independent series are evaluated per-series (monotonicity, dedup)."""
    verdict = validate_reference_dataset(_rows(
        ("UNRATE", 1000, 3.5),
        ("CPIAUCSL", 500, 280.0),   # earlier time, different series — OK
        ("UNRATE", 2000, 3.6),
        ("CPIAUCSL", 600, 281.0),
    ))
    assert verdict.ok, verdict.errors


def test_equal_observation_time_in_same_series_fails() -> None:
    """Same (series_id, observation_time) twice → E_EVIDENCE_MONOTONICITY."""
    verdict = validate_reference_dataset(_rows(
        ("UNRATE", 1000, 3.5),
        ("UNRATE", 1000, 3.6),   # duplicate
    ))
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_MONOTONICITY" in codes


def test_valid_equal_observation_time_across_series() -> None:
    """Same observation_time in different series is fine."""
    verdict = validate_reference_dataset(_rows(
        ("UNRATE", 1000, 3.5),
        ("CPIAUCSL", 1000, 280.0),
    ))
    assert verdict.ok, verdict.errors


# ---------------------------------------------------------------------------
# 3. Invariant: observation_time monotone non-decreasing
# ---------------------------------------------------------------------------


def test_decreasing_observation_time_rejected() -> None:
    verdict = validate_reference_dataset(_rows(
        ("UNRATE", 2000, 3.5),
        ("UNRATE", 1000, 3.4),   # goes backward
    ))
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_REFERENCE_OBSERVATION_TIME_ORDER" in codes


def test_constant_observation_time_monotone_ok() -> None:
    """Non-decreasing allows equal — duplicate check is separate."""
    # Equal times are caught by dedup, not by the order check;
    # the order check only fires when time strictly decreases.
    verdict = validate_reference_dataset(_rows(
        ("UNRATE", 1000, 3.5),
        ("UNRATE", 1000, 3.6),
    ))
    # E_EVIDENCE_MONOTONICITY fires (dedup), NOT E_REFERENCE_OBSERVATION_TIME_ORDER
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_MONOTONICITY" in codes
    assert "E_REFERENCE_OBSERVATION_TIME_ORDER" not in codes


# ---------------------------------------------------------------------------
# 4. Invariant: value must be finite (NaN / Inf rejected)
# ---------------------------------------------------------------------------


def test_nan_value_rejected() -> None:
    verdict = validate_reference_dataset(_rows(
        ("UNRATE", 1000, math.nan),
    ))
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_RANGE" in codes


def test_positive_inf_rejected() -> None:
    verdict = validate_reference_dataset(_rows(
        ("UNRATE", 1000, math.inf),
    ))
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_RANGE" in codes


def test_negative_inf_rejected() -> None:
    verdict = validate_reference_dataset(_rows(
        ("UNRATE", 1000, -math.inf),
    ))
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_RANGE" in codes


def test_zero_value_ok() -> None:
    """Zero is a valid finite number."""
    verdict = validate_reference_dataset(_rows(("UNRATE", 1000, 0.0)))
    assert verdict.ok, verdict.errors


def test_negative_value_ok() -> None:
    """Negative values are valid (e.g. real interest rates, spreads)."""
    verdict = validate_reference_dataset(_rows(("UNRATE", 1000, -0.5)))
    assert verdict.ok, verdict.errors


# ---------------------------------------------------------------------------
# 5. Invariant: type checks
# ---------------------------------------------------------------------------


def test_wrong_type_observation_time_rejected() -> None:
    rows = [{"series_id": "UNRATE", "observation_time": "not-an-int", "value": 3.5}]
    verdict = validate_reference_dataset(rows)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_TYPE" in codes


def test_wrong_type_series_id_rejected() -> None:
    rows = [{"series_id": 12345, "observation_time": 1000, "value": 3.5}]
    verdict = validate_reference_dataset(rows)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_TYPE" in codes


def test_wrong_type_value_rejected() -> None:
    rows = [{"series_id": "UNRATE", "observation_time": 1000, "value": "not-a-number"}]
    verdict = validate_reference_dataset(rows)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_TYPE" in codes


def test_bool_not_valid_int_for_observation_time() -> None:
    """bool is a subclass of int in Python — must be rejected."""
    rows = [{"series_id": "UNRATE", "observation_time": True, "value": 3.5}]
    verdict = validate_reference_dataset(rows)
    assert not verdict.ok


def test_bool_not_valid_number_for_value() -> None:
    rows = [{"series_id": "UNRATE", "observation_time": 1000, "value": True}]
    verdict = validate_reference_dataset(rows)
    assert not verdict.ok


# ---------------------------------------------------------------------------
# 6. Missing columns
# ---------------------------------------------------------------------------


def test_missing_observation_time_rejected() -> None:
    rows = [{"series_id": "UNRATE", "value": 3.5}]
    verdict = validate_reference_dataset(rows)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_SCHEMA_MISMATCH" in codes


def test_missing_series_id_rejected() -> None:
    rows = [{"observation_time": 1000, "value": 3.5}]
    verdict = validate_reference_dataset(rows)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_SCHEMA_MISMATCH" in codes


def test_missing_value_rejected() -> None:
    rows = [{"series_id": "UNRATE", "observation_time": 1000}]
    verdict = validate_reference_dataset(rows)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_SCHEMA_MISMATCH" in codes


# ---------------------------------------------------------------------------
# 7. Empty dataset
# ---------------------------------------------------------------------------


def test_empty_rows_rejected() -> None:
    verdict = validate_reference_dataset([])
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_ROWS_MISSING" in codes


# ---------------------------------------------------------------------------
# 8. Multiple errors accumulated (not fail-fast)
# ---------------------------------------------------------------------------


def test_multiple_violations_all_reported() -> None:
    """Validator accumulates all errors, not just the first."""
    rows = [
        {"series_id": "UNRATE", "observation_time": 1000, "value": math.nan},
        {"series_id": "UNRATE", "observation_time": 500, "value": 3.0},   # order violation
    ]
    verdict = validate_reference_dataset(rows)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    # Both NaN (E_EVIDENCE_RANGE) and out-of-order (E_REFERENCE_OBSERVATION_TIME_ORDER) fire
    assert "E_EVIDENCE_RANGE" in codes
    assert "E_REFERENCE_OBSERVATION_TIME_ORDER" in codes


# ---------------------------------------------------------------------------
# 9. Docstring / platform flow (smoke only — verifies the intended join pattern)
# ---------------------------------------------------------------------------


def test_platform_flow_docstring_example() -> None:
    """
    Intended platform flow:
        1. Ingest FRED series → reference_observations rows
           (validated here with validate_reference_dataset)
        2. Platform joins reference rows into evidence rows upstream
           (left-join on observation_time ≤ decision_time per series)
        3. PM EvidenceSpec references the joined column as an ordinary
           declared feature — the engine treats it identically to any
           other feature column; no macro-awareness required

    This test validates that a well-formed FRED-style reference dataset
    passes validation — the join + feature-column path is platform-side
    and not tested here.
    """
    fred_unrate = [
        {"series_id": "UNRATE", "observation_time": 1_609_459_200, "value": 6.7},   # Jan 2021
        {"series_id": "UNRATE", "observation_time": 1_612_137_600, "value": 6.2},   # Feb 2021
        {"series_id": "UNRATE", "observation_time": 1_614_556_800, "value": 6.0},   # Mar 2021
    ]
    verdict = validate_reference_dataset(fred_unrate)
    assert verdict.ok, verdict.errors
