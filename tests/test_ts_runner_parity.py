"""TS runner parity / documented divergence.

Engine 0.3 is correctness-first, not TS parity. For 5 curated fixtures,
``tests/fixtures/runner/ts_runner_oracle.mjs`` produces TS-runner output via
the real ``runEvidenceBacktest`` from pancake-production and commits it to
``ts_runner_expected.json``. This test asserts:

- **match** fixtures: Engine 0.3 metrics within ``1e-9`` of TS.
- **documented_divergence** fixtures: Engine 0.3 differs from TS in the specific
  way recorded in ``pancake-engine-0.3-ts-divergences.md`` (D-1 cash leak,
  D-11 fee realized at entry). The exact shape of the divergence is asserted.

The committed ``ts_runner_expected.json`` is the source of truth — the test
is standalone and does not import anything from pancake-production.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pancake_engine import BacktestConfig, run_backtest
from pancake_engine.types import EvidenceDataset, EvidenceSpec

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "runner" / "ts_runner_expected.json"
EPSILON = 1e-9


def _load_fixtures() -> list[dict]:
    if not FIXTURES_PATH.exists():
        pytest.skip(
            f"{FIXTURES_PATH.name} missing; regenerate with "
            "`node tests/fixtures/runner/ts_runner_oracle.mjs`"
        )
    return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))


def _build_dataset(rows: list[dict], schema_cols: list[dict]) -> EvidenceDataset:
    return EvidenceDataset.model_validate({
        "id": "ds_oracle",
        "schema": {"columns": schema_cols},
        "schema_sha256": "0" * 64,
        "storage_mode": "inline",
        "rows_inline": rows,
        "rows_sha256": "0" * 64,
        "row_count": len(rows),
    })


def _run_python(fixture: dict):
    raw_spec = fixture["raw_spec"]
    spec = EvidenceSpec.model_validate(raw_spec)
    schema_cols = raw_spec["schema_requirements"]["required_columns"]
    dataset = _build_dataset(fixture["rows"], schema_cols)
    config = BacktestConfig(observation_time=fixture["observation_now_sec"])
    return run_backtest(spec, dataset, config)


# -----------------------------------------------------------------------------
# Match fixtures: assert Engine 0.3 == TS within epsilon
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", [
    "single_yes_win_clean",
    "no_side_at_0_96",
    "three_sequential_trades",
])
def test_ts_runner_parity_match(fixture_name: str) -> None:
    fixtures = {f["name"]: f for f in _load_fixtures()}
    fixture = fixtures[fixture_name]
    assert fixture["expected"] == "match", f"{fixture_name}: expected=match in oracle"

    py_result = _run_python(fixture)
    assert py_result.validation.ok, [e.code for e in py_result.validation.errors]

    ts_metrics = fixture["ts_result"]["metrics"]
    py_std = py_result.metrics.standard

    # total_return
    assert abs(py_std.total_return - ts_metrics["total_return"]) < EPSILON, (
        f"total_return: ts={ts_metrics['total_return']}, py={py_std.total_return}"
    )
    # ending_capital
    assert abs(py_std.ending_capital - ts_metrics["ending_capital"]) < EPSILON
    # num_trades
    assert py_std.num_trades == ts_metrics["num_trades"]
    # max_drawdown
    assert abs(py_std.max_drawdown - ts_metrics["max_drawdown"]) < EPSILON, (
        f"max_drawdown: ts={ts_metrics['max_drawdown']}, py={py_std.max_drawdown}"
    )
    # win_rate (TS uses 0 for empty trades; Engine 0.3 uses None; only compare when both non-null)
    ts_win_rate = ts_metrics["win_rate"]
    if py_std.win_rate is not None and ts_win_rate is not None and ts_metrics["num_trades"] > 0:
        assert abs(py_std.win_rate - ts_win_rate) < EPSILON
    # Per-trade fields
    py_trades = sorted(py_result.trades, key=lambda t: t.entry_t)
    ts_trades = sorted(fixture["ts_result"]["trades"], key=lambda t: t["entry_t"])
    assert len(py_trades) == len(ts_trades)
    for py_t, ts_t in zip(py_trades, ts_trades):
        assert abs(py_t.cost - ts_t["cost"]) < EPSILON
        assert abs(py_t.proceeds - ts_t["proceeds"]) < EPSILON
        assert abs(py_t.pnl - ts_t["pnl"]) < EPSILON
        assert abs(py_t.return_pct - ts_t["return_pct"]) < EPSILON
        assert abs(py_t.shares - ts_t["shares"]) < EPSILON
        assert abs(py_t.entry_price - ts_t["entry_price"]) < EPSILON
        assert py_t.market_slug == ts_t["market_slug"]


# -----------------------------------------------------------------------------
# Documented divergence: D-1 cash-leak fix
# -----------------------------------------------------------------------------


def test_ts_divergence_d1_cash_leak() -> None:
    """Engine 0.3 prevents future cash from leaking into earlier decisions.

    TS processes A's full lifecycle (decision + resolution at T=1000) before
    starting B's decision at T=200, so B sees post-A-settlement cash. Engine
    0.3 uses event-time ledger: A's $1800 proceeds at T=1000 cannot reach B's
    decision at T=200.

    Expected divergence: Engine 0.3 total_return ≈ 0.99 (A wins + B sized
    against pre-leak $100), TS total_return ≈ 2.61 (B sized against $1900
    that includes A's future settlement).
    """
    fixtures = {f["name"]: f for f in _load_fixtures()}
    fixture = fixtures["cash_leak_overlapping"]
    assert fixture["expected"] == "documented_divergence"

    py_result = _run_python(fixture)
    assert py_result.validation.ok

    ts_total_return = fixture["ts_result"]["metrics"]["total_return"]
    py_total_return = py_result.metrics.standard.total_return

    # The whole point: Engine 0.3 is strictly LESS than TS for this case
    # (TS overstates returns by spending future cash).
    assert py_total_return < ts_total_return, (
        f"D-1 fix: Engine 0.3 total_return ({py_total_return}) should be strictly less than "
        f"TS total_return ({ts_total_return}) — TS leaks A's future $1800 into B's sizing."
    )
    # Specific shape: Engine 0.3 ≈ 0.99 (A wins, B sees pre-leak $100, B wins → $1990)
    assert abs(py_total_return - 0.99) < 1e-6, (
        f"D-1: expected Engine 0.3 total_return ≈ 0.99, got {py_total_return}"
    )
    # TS ≈ 2.61
    assert abs(ts_total_return - 2.61) < 1e-6


# -----------------------------------------------------------------------------
# Documented divergence: D-11 fee realized at entry
# -----------------------------------------------------------------------------


def test_ts_divergence_d11_fee_realized_at_entry() -> None:
    """Engine 0.3 realizes fees at entry (mark_at_cost = shares × entry_fill_price).

    TS samples equity only at resolution time, so the decision-time fee impact
    is invisible until the trade resolves. Engine 0.3 emits a decision-time
    equity sample at ``starting − fee``.

    Expected divergence:
    - Final metrics (total_return, ending_capital): MATCH (fee total absorbed identically).
    - equity_curve: differs at the decision-time event point.
    """
    fixtures = {f["name"]: f for f in _load_fixtures()}
    fixture = fixtures["fee_realized_at_entry"]
    assert fixture["expected"] == "documented_divergence"

    py_result = _run_python(fixture)
    assert py_result.validation.ok

    ts_metrics = fixture["ts_result"]["metrics"]
    py_std = py_result.metrics.standard

    # Final metrics match (lifecycle P&L identical)
    assert abs(py_std.total_return - ts_metrics["total_return"]) < EPSILON
    assert abs(py_std.ending_capital - ts_metrics["ending_capital"]) < EPSILON

    # But equity_curve differs at the decision-time event
    ts_equity = fixture["ts_result"]["equity_curve"]
    # TS has one point at T=200 (resolution); Engine 0.3 has two points: T=100 (decision) and T=200.
    assert len(ts_equity) == 1
    assert len(py_result.equity_curve) == 2
    decision_point = next(p for p in py_result.equity_curve if p.t == 100)
    # notional = 100, fee = 100 × 0.01 = 1.0, equity at decision = 1000 - 1 = 999
    assert abs(decision_point.equity - 999.0) < EPSILON, (
        f"D-11: Engine 0.3 should realize fee at entry → equity at T=100 = $999, "
        f"got {decision_point.equity}"
    )


# -----------------------------------------------------------------------------
# Sanity: fixture file shape
# -----------------------------------------------------------------------------


def test_ts_runner_fixtures_count() -> None:
    fixtures = _load_fixtures()
    assert len(fixtures) == 5, f"expected 5 TS-runner fixtures, got {len(fixtures)}"


def test_ts_runner_fixtures_have_required_fields() -> None:
    fixtures = _load_fixtures()
    for f in fixtures:
        assert "name" in f
        assert "raw_spec" in f
        assert "rows" in f
        assert "expected" in f
        assert f["expected"] in ("match", "documented_divergence")
        assert "ts_result" in f
        assert "metrics" in f["ts_result"]
