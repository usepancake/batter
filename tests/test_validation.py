"""Validation layer: all error codes + structured output."""

from __future__ import annotations

import pytest

from pancake_engine.types import EvidenceCosts, EvidenceSpec
from pancake_engine.validate import validate_dataset, validate_spec

from ._runner_helpers import SCHEMA_COLUMNS, make_dataset, make_spec, row


def test_validation_negative_codes_starting_capital_zero() -> None:
    with pytest.raises(ValueError, match="E_EVIDENCE_SPEC_INVALID"):
        make_spec(starting_capital=0)


def test_validation_negative_codes_starting_capital_negative() -> None:
    with pytest.raises(ValueError, match="E_EVIDENCE_SPEC_INVALID"):
        make_spec(starting_capital=-1.0)


def test_validation_negative_codes_slippage_bps_negative() -> None:
    with pytest.raises(ValueError, match="E_EVIDENCE_SPEC_INVALID"):
        EvidenceCosts(slippage_bps=-1, fee_bps=0)


def test_validation_negative_codes_fee_bps_negative() -> None:
    with pytest.raises(ValueError, match="E_EVIDENCE_SPEC_INVALID"):
        EvidenceCosts(slippage_bps=0, fee_bps=-1)


def test_validation_missing_column_in_dataset() -> None:
    spec = make_spec()
    # Dataset missing the "alpha" column declared in spec.
    dataset = make_dataset([
        {"mkt": "m/A", "dec_ts": 100, "res_ts": 200, "price": 0.5, "outcome": 1, "target": 1},
    ])
    # Strip "alpha" column from the dataset schema
    new_cols = [c for c in dataset.dataset_schema.columns if c.name != "alpha"]
    dataset.dataset_schema.columns = new_cols
    verdict, _ = validate_dataset(dataset, spec)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_SCHEMA_MISMATCH" in codes


def test_yes_payoff_undeclared_column_fails_validation() -> None:
    # 2026-06-10 fade-path P1 root cause: a predicate on an UNDECLARED column (here
    # the semantic-role name 'resolved_outcome_numeric' instead of the actual column
    # 'outcome') silently evaluates False at runtime → a NO-side yes_payoff means the
    # strategy "wins" every trade and emits astronomical garbage (~2e+68) with no error.
    spec = make_spec(side="NO", yes_payoff_when={"feature": "resolved_outcome_numeric", "gte": 1})
    verdict = validate_spec(spec)
    assert not verdict.ok
    assert "E_EVIDENCE_SPEC_INVALID" in {e.code for e in verdict.errors}


def test_entry_undeclared_column_fails_validation() -> None:
    spec = make_spec(entry_when={"feature": "nope_not_a_column", "gte": 1.0})
    verdict = validate_spec(spec)
    assert not verdict.ok
    assert "E_EVIDENCE_SPEC_INVALID" in {e.code for e in verdict.errors}


def test_undeclared_column_run_is_blocked_not_garbage() -> None:
    from pancake_engine import BacktestConfig, run_backtest

    spec = make_spec(
        side="NO", sizing_value=0.1, starting_capital=1000.0,
        yes_payoff_when={"feature": "resolved_outcome_numeric", "gte": 1},
    )
    dataset = make_dataset([
        row(mkt=f"m/{i}", dec_ts=i * 100, res_ts=i * 100 + 50, price=0.5, outcome=1, alpha=3.0, target=1)
        for i in range(1, 6)
    ])
    r = run_backtest(spec, dataset, BacktestConfig(observation_time=600))
    assert not r.validation.ok                  # blocked, not a silent success
    assert r.metrics.standard.num_trades == 0   # no garbage trades
    assert r.metrics.standard.total_return == 0.0
    assert r.result_hash == ""                  # blocked → no receipt hash


def test_declared_columns_still_validate() -> None:
    assert validate_spec(make_spec()).ok
    assert validate_spec(make_spec(side="NO")).ok
    assert validate_spec(
        make_spec(entry_when={"feature": "alpha", "gte": 2.0},
                  yes_payoff_when={"feature": "outcome", "gte": 1})
    ).ok


def test_validation_wrong_type_in_row() -> None:
    spec = make_spec()
    dataset = make_dataset([
        {"mkt": "m/A", "dec_ts": 100, "res_ts": 200, "price": "not-a-number",
         "outcome": 1, "alpha": 3.0, "target": 1},
    ])
    verdict, _ = validate_dataset(dataset, spec)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_TYPE" in codes


def test_validation_range_violation() -> None:
    spec = make_spec()
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=1.5, outcome=1),  # price > 1
    ])
    verdict, _ = validate_dataset(dataset, spec)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_RANGE" in codes


def test_validation_lookahead_violation() -> None:
    spec = make_spec()
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=300, res_ts=200, price=0.5, outcome=1),  # decision >= resolution
    ])
    verdict, _ = validate_dataset(dataset, spec)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_LOOKAHEAD" in codes


def test_validation_monotonicity_violation() -> None:
    spec = make_spec()
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.5, outcome=1),
        row(mkt="m/A", dec_ts=100, res_ts=300, price=0.5, outcome=1),  # duplicate (mkt, dec_ts)
    ])
    verdict, _ = validate_dataset(dataset, spec)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_MONOTONICITY" in codes


def test_validation_feature_missing_in_row() -> None:
    spec = make_spec()
    dataset = make_dataset([
        {"mkt": "m/A", "dec_ts": 100, "res_ts": 200, "price": 0.5,
         "outcome": 1, "target": 1},  # missing alpha
    ])
    verdict, _ = validate_dataset(dataset, spec)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_FEATURE_MISSING" in codes


def test_validation_inline_required() -> None:
    spec = make_spec()
    dataset = make_dataset([row(mkt="m/A", dec_ts=100, res_ts=200, price=0.5, outcome=1)])
    dataset.storage_mode = "pointer"
    verdict, _ = validate_dataset(dataset, spec)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_INLINE_REQUIRED" in codes


def test_validation_rows_missing() -> None:
    spec = make_spec()
    dataset = make_dataset([])
    verdict, _ = validate_dataset(dataset, spec)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_ROWS_MISSING" in codes


def test_validation_unsupported_sizing_mode_blocked_by_pydantic() -> None:
    # pydantic literal validator rejects unknown sizing.mode before our layer
    with pytest.raises(ValueError):
        make_spec()  # baseline ok
        EvidenceSpec.model_validate({
            "spec_family": "pancake-evidence-spec",
            "spec_version": "0.1",
            "name": "x",
            "schema_requirements": {"required_columns": []},
            "strategy": {
                "side": "YES",
                "entry": {"when": {"all_of": []}},
                "yes_payoff": {"when": {"all_of": []}},
                "sizing": {"mode": "kelly", "value": 0.1},  # not allowed
            },
            "costs": {"slippage_bps": 0, "fee_bps": 0},
            "starting_capital": 1000,
        })


def test_validation_sizing_value_out_of_range_blocked_by_spec_validator() -> None:
    spec = make_spec(sizing_value=1.5)  # builder will succeed via pydantic, but
    # validate_spec catches sizing.value > 1
    verdict = validate_spec(spec)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_SPEC_INVALID" in codes


def test_validation_ok_baseline() -> None:
    spec = make_spec()
    dataset = make_dataset([row(mkt="m/A", dec_ts=100, res_ts=200, price=0.5, outcome=1)])
    verdict, lookup = validate_dataset(dataset, spec)
    assert verdict.ok
    assert lookup["entry_price"] == "price"
    assert lookup["market_link"] == "mkt"
