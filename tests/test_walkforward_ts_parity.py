"""WF per-fold parity vs real TS ``runEvidenceBacktest``.

TS has no walk-forward. This test slices a fixture into 3 hand-computed test
windows (matching what Engine 0.3's schedule would produce), runs Engine 0.3's
``run_walkforward``, and asserts each fold's metrics match TS within 1e-9.

Aggregate WF metrics have no TS counterpart — Engine-0.3-native, pinned via
``aggregate_result_hash`` in the example fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pancake_engine import (
    BacktestConfig,
    EvidenceDataset,
    EvidenceSpec,
    WalkforwardConfig,
    run_walkforward,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "runner" / "ts_walkforward_expected.json"
EPSILON = 1e-9


def _load_fixture() -> dict:
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"{FIXTURE_PATH.name} missing; regenerate via "
            "`node tests/fixtures/runner/ts_walkforward_oracle.mjs`"
        )
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_walkforward_per_fold_parity() -> None:
    """Engine 0.3 fold metrics match real TS runEvidenceBacktest per fold within 1e-9."""
    fix = _load_fixture()
    raw_spec = fix["raw_spec"]
    rows = fix["rows"]

    spec = EvidenceSpec.model_validate(raw_spec)
    dataset = EvidenceDataset.model_validate({
        "id": "ds_wf_parity",
        "schema": {"columns": raw_spec["schema_requirements"]["required_columns"]},
        "schema_sha256": "0" * 64,
        "storage_mode": "inline",
        "rows_inline": rows,
        "rows_sha256": "0" * 64,
        "row_count": len(rows),
    })

    DAY = 86_400
    wf_config = WalkforwardConfig(
        window_type="expanding",
        test_horizon=30 * DAY,
        step=30 * DAY,
        min_fold_count=3,
    )
    bt_config = BacktestConfig(observation_time=200 * DAY)
    py_result = run_walkforward(spec, dataset, wf_config, bt_config)

    assert py_result.validation.ok, [e.code for e in py_result.validation.errors]
    assert len(py_result.folds) == 3 == len(fix["folds"])

    for py_fold, ts_fold in zip(py_result.folds, fix["folds"]):
        assert py_fold.definition.test_window == tuple(ts_fold["test_window"])
        py_m = py_fold.result.metrics.standard
        ts_m = ts_fold["ts_result"]["metrics"]
        assert py_m.num_trades == ts_m["num_trades"]
        assert abs(py_m.total_return - ts_m["total_return"]) < EPSILON, (
            f"fold {py_fold.definition.index} total_return: "
            f"py={py_m.total_return}, ts={ts_m['total_return']}"
        )
        assert abs(py_m.ending_capital - ts_m["ending_capital"]) < EPSILON
        assert abs(py_m.max_drawdown - ts_m["max_drawdown"]) < EPSILON

        py_trades = sorted(py_fold.result.trades, key=lambda t: t.entry_t)
        ts_trades = sorted(ts_fold["ts_result"]["trades"], key=lambda t: t["entry_t"])
        assert len(py_trades) == len(ts_trades)
        for py_t, ts_t in zip(py_trades, ts_trades):
            assert abs(py_t.pnl - ts_t["pnl"]) < EPSILON
            assert abs(py_t.return_pct - ts_t["return_pct"]) < EPSILON
            assert abs(py_t.shares - ts_t["shares"]) < EPSILON
