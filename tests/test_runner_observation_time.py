"""observation_time rule: required when unresolved + derived when fully resolved."""

from __future__ import annotations

from pancake_engine import BacktestConfig, WarningCode, run_backtest

from ._runner_helpers import make_dataset, make_spec, row


def test_observation_time_required_when_unresolved() -> None:
    """Dataset with null resolved_outcome_numeric AND no config.observation_time → blocking error."""
    spec = make_spec(side="YES")
    dataset = make_dataset([
        # one resolved, one unresolved
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.5, outcome=1, alpha=3.0, target=1),
        {"mkt": "m/B", "dec_ts": 300, "res_ts": 400, "price": 0.5,
         "outcome": None, "alpha": 3.0, "target": 1},
    ])
    # No observation_time set → engine must refuse (no Date.now() fallback).
    config = BacktestConfig()
    result = run_backtest(spec, dataset, config)
    assert not result.validation.ok
    codes = {e.code for e in result.validation.errors}
    assert "E_OBSERVATION_TIME_REQUIRED" in codes


def test_observation_time_derived_when_fully_resolved() -> None:
    """All rows resolved → derive max(resolution_time); OBSERVATION_TIME_DERIVED info-warning."""
    spec = make_spec(side="YES")
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.5, outcome=1, alpha=3.0, target=1),
        row(mkt="m/B", dec_ts=300, res_ts=500, price=0.5, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig()  # no observation_time
    result = run_backtest(spec, dataset, config)
    assert result.validation.ok
    assert result.meta["observation_time"] == 500
    assert result.meta["observation_time_derived"] is True
    assert any(w.code == WarningCode.OBSERVATION_TIME_DERIVED for w in result.warnings)


def test_observation_time_derived_rerun_identical_hash() -> None:
    """Auto-derive is deterministic — rerun produces byte-equal result_hash."""
    spec = make_spec(side="YES")
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.5, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig()
    r1 = run_backtest(spec, dataset, config)
    r2 = run_backtest(spec, dataset, config)
    assert r1.result_hash == r2.result_hash
    assert r1.result_hash != ""


def test_observation_time_explicit_overrides_derived() -> None:
    """Explicit config.observation_time is used as-is."""
    spec = make_spec(side="YES")
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.5, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=999)
    result = run_backtest(spec, dataset, config)
    assert result.meta["observation_time"] == 999
    assert result.meta["observation_time_derived"] is False
    assert not any(w.code == WarningCode.OBSERVATION_TIME_DERIVED for w in result.warnings)
