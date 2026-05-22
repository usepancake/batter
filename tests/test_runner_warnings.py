"""Runtime warnings: sizing_clipped, sizing_zero, entry_price_out_of_range, fill_price_out_of_range."""

from __future__ import annotations

from pancake_engine import BacktestConfig, WarningCode, run_backtest

from ._runner_helpers import make_dataset, make_spec, row


def test_sizing_zero_warning_when_no_available_cash() -> None:
    """When available_cash drops to 0 after a prior trade fully consumed it, SIZING_ZERO fires."""
    spec = make_spec(side="YES", sizing_value=1.0, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=500, price=0.5, outcome=1, alpha=3.0, target=1),
        row(mkt="m/B", dec_ts=200, res_ts=600, price=0.5, outcome=1, alpha=3.0, target=1),  # cash=0
    ])
    config = BacktestConfig(observation_time=1000)
    result = run_backtest(spec, dataset, config)
    assert any(w.code == WarningCode.SIZING_ZERO for w in result.warnings)


def test_entry_price_out_of_range_emits_warning() -> None:
    """Row with entry_price = 1.5 (out of [0, 1] declared range) triggers validation error.

    For prices in (0, 1) declared range but the runner-time check at fill,
    we test the runner-side guard separately below.
    """
    spec = make_spec(side="YES")
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=1.5, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=300)
    result = run_backtest(spec, dataset, config)
    # validation should block this (range declared [0, 1])
    assert not result.validation.ok
    codes = {e.code for e in result.validation.errors}
    assert "E_EVIDENCE_RANGE" in codes


def test_entry_price_zero_runner_skip() -> None:
    """If validation passes but entry_price is exactly 0 (edge), runner emits ENTRY_PRICE_OUT_OF_RANGE."""
    # Build a spec WITHOUT range on price so validation passes
    from pancake_engine.types import EvidenceSpec

    raw_schema = [
        {"name": "mkt", "type": "string", "semantic_role": "market_link"},
        {"name": "dec_ts", "type": "int", "semantic_role": "decision_time"},
        {"name": "res_ts", "type": "int", "semantic_role": "resolution_time"},
        # no range
        {"name": "price", "type": "number", "semantic_role": "entry_price"},
        {"name": "outcome", "type": "int", "semantic_role": "resolved_outcome_numeric"},
        {"name": "alpha", "type": "number", "semantic_role": "feature"},
        {"name": "target", "type": "int", "semantic_role": "feature"},
    ]
    spec = EvidenceSpec.model_validate({
        "spec_family": "pancake-evidence-spec",
        "spec_version": "0.1",
        "name": "norange",
        "schema_requirements": {"required_columns": raw_schema},
        "strategy": {
            "side": "YES",
            "entry": {"when": {"feature": "alpha", "gte": 2.0}},
            "yes_payoff": {"when": {"feature_equal": {"a": "target", "b": "outcome"}}},
            "sizing": {"mode": "fixed_fraction", "value": 0.1},
        },
        "costs": {"slippage_bps": 0, "fee_bps": 0},
        "starting_capital": 1000.0,
    })
    # Use same schema for dataset
    from pancake_engine.types import EvidenceDataset
    dataset = EvidenceDataset.model_validate({
        "id": "ds",
        "schema": {"columns": raw_schema},
        "schema_sha256": "0" * 64,
        "storage_mode": "inline",
        "rows_inline": [
            {"mkt": "m/A", "dec_ts": 100, "res_ts": 200, "price": 0.0,
             "outcome": 1, "alpha": 3.0, "target": 1},
        ],
        "rows_sha256": "0" * 64,
        "row_count": 1,
    })
    config = BacktestConfig(observation_time=300)
    result = run_backtest(spec, dataset, config)
    assert result.validation.ok  # validation didn't block (no range declared)
    assert any(w.code == WarningCode.ENTRY_PRICE_OUT_OF_RANGE for w in result.warnings)
    assert result.metrics.standard.num_trades == 0
