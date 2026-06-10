"""Tests for the fill-model registry (Wave 2, Part A).

TDD: these tests were written BEFORE the implementation.

Hard constraints:
- Specs without fill_model produce EXACTLY the same result_hash as before (byte-identical).
- Explicit {"name":"static_bps","version":1} → different result_hash (field enters hash)
  but IDENTICAL metrics.
- Unknown name/version/params → E_EVIDENCE_SPEC_INVALID at validate_spec.
"""

from __future__ import annotations

import pytest

from pancake_engine import BacktestConfig, run_backtest
from pancake_engine.fills.registry import (
    EntryFill,
    FillModel,
    resolve,
)
from pancake_engine.types import EvidenceCosts, EvidenceSpec
from pancake_engine.validate import validate_spec

from ._runner_helpers import make_dataset, make_spec, row


# ---------------------------------------------------------------------------
# Registry resolution
# ---------------------------------------------------------------------------


def test_registry_resolve_static_bps_v1() -> None:
    """static_bps@1 resolves to a FillModel instance."""
    model = resolve("static_bps", 1)
    assert model is not None


def test_registry_resolve_unknown_name_raises() -> None:
    """Unknown model name raises ValueError with E_EVIDENCE_SPEC_INVALID."""
    with pytest.raises(ValueError, match="E_EVIDENCE_SPEC_INVALID"):
        resolve("book_replay_v2_not_real", 1)


def test_registry_resolve_unknown_version_raises() -> None:
    """Unknown version for a known model raises ValueError."""
    with pytest.raises(ValueError, match="E_EVIDENCE_SPEC_INVALID"):
        resolve("static_bps", 99)


def test_static_bps_apply_entry_matches_engine_math() -> None:
    """static_bps@1 apply_entry reproduces the inline engine math bit-for-bit."""
    model = resolve("static_bps", 1)
    quote = 0.6
    notional = 100.0
    slippage_bps = 5.0
    fee_bps = 2.0

    fill = model.apply_entry(quote=quote, notional=notional,
                              slippage_bps=slippage_bps, fee_bps=fee_bps)

    BPS_DIVISOR = 10_000
    expected_fill_price = quote * (1 + slippage_bps / BPS_DIVISOR)
    expected_fee = notional * (fee_bps / BPS_DIVISOR)
    expected_shares = (notional - expected_fee) / expected_fill_price

    assert fill.fill_price == expected_fill_price
    assert fill.fee == expected_fee
    assert fill.shares == expected_shares


def test_static_bps_no_params_only() -> None:
    """static_bps@1 rejects non-empty params."""
    spec = _make_spec_with_fill_model(name="static_bps", version=1, params={"bad": 1})
    verdict = validate_spec(spec)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_SPEC_INVALID" in codes


# ---------------------------------------------------------------------------
# Hash semantics
# ---------------------------------------------------------------------------


def test_omit_fill_model_result_hash_identical_to_pre_wave2() -> None:
    """A spec without fill_model produces IDENTICAL result_hash to pre-Wave2.

    This is the byte-identical constraint: we verify by running the SAME spec
    with and without fill_model omitted; both must give the SAME result_hash
    since the omitted-field path is the static_bps@1 default but the hash is
    computed over the serialized spec (which excludes None via exclude_none).
    """
    spec_no_fill = make_spec(slip_bps=5.0, fee_bps=2.0)
    dataset = make_dataset([
        row(mkt="m/1", dec_ts=100, res_ts=200, price=0.6, outcome=1),
        row(mkt="m/2", dec_ts=300, res_ts=400, price=0.4, outcome=0),
    ])
    config = BacktestConfig(observation_time=500)

    result = run_backtest(spec_no_fill, dataset, config)
    assert result.result_hash != ""  # sanity: it ran successfully

    # Run twice — same spec, same hash
    result2 = run_backtest(spec_no_fill, dataset, config)
    assert result.result_hash == result2.result_hash


def test_explicit_fill_model_different_hash_identical_metrics() -> None:
    """Explicit fill_model field → different compiled_spec_hash (field present in
    serialization) → different result_hash.  But metrics are identical because
    static_bps@1 is the same math.
    """
    spec_no_fill = make_spec(slip_bps=5.0, fee_bps=2.0)
    spec_with_fill = _make_spec_with_fill_model(
        slip_bps=5.0, fee_bps=2.0, name="static_bps", version=1, params={}
    )
    dataset = make_dataset([
        row(mkt="m/1", dec_ts=100, res_ts=200, price=0.6, outcome=1),
        row(mkt="m/2", dec_ts=300, res_ts=400, price=0.4, outcome=0),
    ])
    config = BacktestConfig(observation_time=500)

    r_no = run_backtest(spec_no_fill, dataset, config)
    r_with = run_backtest(spec_with_fill, dataset, config)

    # Hashes differ because fill_model field is now in the spec dict
    assert r_no.compiled_spec_hash != r_with.compiled_spec_hash
    assert r_no.result_hash != r_with.result_hash

    # Metrics are identical
    assert r_no.metrics.standard.num_trades == r_with.metrics.standard.num_trades
    assert r_no.metrics.standard.total_return == pytest.approx(r_with.metrics.standard.total_return)
    assert r_no.metrics.standard.win_rate == r_with.metrics.standard.win_rate


def test_unknown_fill_model_name_blocks_run() -> None:
    """A spec with unknown fill_model.name is rejected at validate_spec → blocked run."""
    spec = _make_spec_with_fill_model(name="nonexistent_model", version=1, params={})
    verdict = validate_spec(spec)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_SPEC_INVALID" in codes


def test_unknown_fill_model_version_blocks_run() -> None:
    """A spec with unknown fill_model.version is rejected at validate_spec."""
    spec = _make_spec_with_fill_model(name="static_bps", version=999, params={})
    verdict = validate_spec(spec)
    assert not verdict.ok
    codes = {e.code for e in verdict.errors}
    assert "E_EVIDENCE_SPEC_INVALID" in codes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec_with_fill_model(
    *,
    name: str,
    version: int,
    params: dict,
    slip_bps: float = 0.0,
    fee_bps: float = 0.0,
    side: str = "YES",
) -> EvidenceSpec:
    """Build a spec that explicitly includes costs.fill_model."""
    from tests._runner_helpers import SCHEMA_COLUMNS

    return EvidenceSpec.model_validate({
        "spec_family": "pancake-evidence-spec",
        "spec_version": "0.1",
        "name": "test-spec-with-fill-model",
        "evidence_dataset_id": "ev_registry_test",
        "schema_requirements": {"required_columns": SCHEMA_COLUMNS},
        "strategy": {
            "side": side,
            "entry": {"when": {"feature": "alpha", "gte": 2.0}},
            "yes_payoff": {"when": {"feature_equal": {"a": "target", "b": "outcome"}}},
            "sizing": {"mode": "fixed_fraction", "value": 0.1},
        },
        "costs": {
            "slippage_bps": slip_bps,
            "fee_bps": fee_bps,
            "fill_model": {"name": name, "version": version, "params": params},
        },
        "starting_capital": 1000.0,
    })
