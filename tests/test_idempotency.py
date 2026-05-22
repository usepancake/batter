"""Idempotency: same inputs → byte-equal result_hash."""

from __future__ import annotations

from pancake_engine import BacktestConfig, run_backtest

from ._runner_helpers import make_dataset, make_spec, row


def test_idempotency_rerun_byte_equal() -> None:
    spec = make_spec(side="NO", sizing_value=0.1, slip_bps=50, fee_bps=10, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=500, price=0.42, outcome=0, alpha=3.0, target=1),
        row(mkt="m/A", dec_ts=600, res_ts=900, price=0.58, outcome=1, alpha=2.5, target=0),
        row(mkt="m/B", dec_ts=1000, res_ts=2000, price=0.20, outcome=1, alpha=4.0, target=1),
    ])
    config = BacktestConfig(observation_time=3000)

    r1 = run_backtest(spec, dataset, config)
    r2 = run_backtest(spec, dataset, config)
    r3 = run_backtest(spec, dataset, config)

    assert r1.result_hash == r2.result_hash == r3.result_hash
    assert r1.result_hash != ""


def test_idempotency_observation_time_explicit_matches_derived_value() -> None:
    """Explicit observation_time == derived value → same config_hash; result_hash differs
    because the OBSERVATION_TIME_DERIVED warning fires only on the derived path."""
    spec = make_spec(side="YES")
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.5, outcome=1, alpha=3.0, target=1),
    ])
    r_derived = run_backtest(spec, dataset, BacktestConfig())  # derives 200
    r_explicit = run_backtest(spec, dataset, BacktestConfig(observation_time=200))
    # Same observation_time value → same config_hash
    assert r_derived.config_hash == r_explicit.config_hash
    # But result_hash differs because warnings differ (DERIVED warning only on the derived path)
    assert r_derived.result_hash != r_explicit.result_hash


def test_idempotency_different_observation_time_different_hash() -> None:
    """Different explicit observation_time values → different config_hash → different result_hash."""
    spec = make_spec(side="YES")
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.5, outcome=1, alpha=3.0, target=1),
    ])
    r1 = run_backtest(spec, dataset, BacktestConfig(observation_time=200))
    r2 = run_backtest(spec, dataset, BacktestConfig(observation_time=999))
    assert r1.config_hash != r2.config_hash
    assert r1.result_hash != r2.result_hash


def test_idempotency_dict_key_order_independent() -> None:
    """Same dataset content with different dict key insertion order → same hash."""
    spec = make_spec(side="YES")
    rows_v1 = [{"mkt": "m/A", "dec_ts": 100, "res_ts": 200, "price": 0.5,
                "outcome": 1, "alpha": 3.0, "target": 1}]
    rows_v2 = [{"target": 1, "alpha": 3.0, "outcome": 1, "price": 0.5,
                "res_ts": 200, "dec_ts": 100, "mkt": "m/A"}]
    config = BacktestConfig(observation_time=300)
    r1 = run_backtest(spec, make_dataset(rows_v1), config)
    r2 = run_backtest(spec, make_dataset(rows_v2), config)
    assert r1.result_hash == r2.result_hash
