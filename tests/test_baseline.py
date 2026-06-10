"""Baseline (buy-and-hold benchmark) — spec v0.2 engine subset.

Convention: NO-FILTER — the baseline takes the strategy's side on EVERY candidate
row with the same sizing and costs, holding to resolution. The strategy differs
from its baseline only by the entry condition, so the baseline isolates the entry
condition's selection value ("does your filter beat buying everything?").
"""

from __future__ import annotations

import pytest

from pancake_engine import BacktestConfig, run_backtest
from pancake_engine.validate import validate_spec

from ._runner_helpers import make_dataset, make_spec, row

DAY = 86_400


def _dataset():
    outcomes = [1, 0, 1, 1, 0, 1]
    alphas = [3.0, 3.5, 2.6, 4.0, 2.8, 3.2]
    return make_dataset([
        row(mkt=f"m/{i}", dec_ts=i * 5 * DAY, res_ts=i * 5 * DAY + 3 * DAY,
            price=0.5, outcome=outcomes[i], alpha=alphas[i], target=1)
        for i in range(6)
    ])


def _spec(*, baseline=None, entry_gte=2.5):
    spec = make_spec(side="YES", sizing_value=0.1, starting_capital=1000.0,
                     entry_when={"feature": "alpha", "gte": entry_gte})
    if baseline is not None:
        spec = spec.model_copy(
            update={"strategy": spec.strategy.model_copy(update={"baseline": baseline})}
        )
    return spec


CFG = BacktestConfig(observation_time=40 * DAY)


def test_baseline_absent_by_default() -> None:
    r = run_backtest(_spec(), _dataset(), CFG)
    assert r.baseline is None
    assert "baseline" in r.to_dict()
    assert r.to_dict()["baseline"] is None


def test_baseline_emitted_and_trades_every_candidate() -> None:
    r = run_backtest(_spec(baseline={"kind": "buy_and_hold"}, entry_gte=3.2), _dataset(), CFG)
    assert r.baseline is not None
    assert r.baseline["kind"] == "buy_and_hold"
    assert r.baseline["convention"] == "no_filter"
    # baseline trades every candidate row; the filtered strategy trades a subset
    assert r.baseline["num_trades"] == 6
    assert r.metrics.standard.num_trades < 6
    for key in ("total_return", "max_drawdown", "ending_capital", "equity_curve"):
        assert key in r.baseline
    assert len(r.baseline["equity_curve"]) >= 2


def test_baseline_equals_strategy_when_entry_passes_everything() -> None:
    # entry alpha >= 0 admits every row → strategy IS the baseline portfolio
    r = run_backtest(_spec(baseline={"kind": "buy_and_hold"}, entry_gte=0.0), _dataset(), CFG)
    s = r.metrics.standard
    assert r.baseline["num_trades"] == s.num_trades == 6
    assert r.baseline["total_return"] == pytest.approx(s.total_return, rel=1e-12)
    assert r.baseline["ending_capital"] == pytest.approx(s.ending_capital, rel=1e-12)


def test_baseline_in_spec_hash_but_block_not_in_result_hash() -> None:
    plain = run_backtest(_spec(), _dataset(), CFG)
    based = run_backtest(_spec(baseline={"kind": "buy_and_hold"}), _dataset(), CFG)
    # requesting a baseline IS part of the receipt identity (spec hash changes)...
    assert plain.compiled_spec_hash != based.compiled_spec_hash
    assert plain.result_hash != based.result_hash
    # ...but the engine math of the strategy itself is untouched
    assert plain.metrics.standard.total_return == based.metrics.standard.total_return
    assert plain.metrics.standard.sharpe == based.metrics.standard.sharpe


def test_baseline_skipped_on_fast_path() -> None:
    r = run_backtest(_spec(baseline={"kind": "buy_and_hold"}), _dataset(), CFG, with_inference=False)
    assert r.baseline is None  # sweep fast path stays free of it


def test_baseline_unknown_kind_blocked() -> None:
    v = validate_spec(_spec(baseline={"kind": "spy_overlay"}))
    assert not v.ok
    assert "E_EVIDENCE_SPEC_INVALID" in {e.code for e in v.errors}
    r = run_backtest(_spec(baseline={"kind": "spy_overlay"}), _dataset(), CFG)
    assert not r.validation.ok
    assert r.result_hash == ""


def test_baseline_deterministic() -> None:
    a = run_backtest(_spec(baseline={"kind": "buy_and_hold"}), _dataset(), CFG)
    b = run_backtest(_spec(baseline={"kind": "buy_and_hold"}), _dataset(), CFG)
    assert a.result_hash == b.result_hash
    assert a.baseline == b.baseline
