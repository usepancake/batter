"""Regression tests for the 2026-06-14 batter audit fixes (F1 + F2).

F1 — malformed condition ASTs should return a clean blocked verdict (result_hash=="",
     verdict not ok, E_EVIDENCE_SPEC_INVALID) rather than raising.

F2 — a denormal entry price (1e-203) passes the 0<price<1 guard but causes
     OverflowError in PM variance. run_backtest must NOT raise; affected metric
     fields degrade to None.
"""

from __future__ import annotations

import math

import pytest

from pancake_engine import BacktestConfig, run_backtest

from ._runner_helpers import make_dataset, make_spec, row

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DAY = 86_400

_BASE_ROWS = [
    row(mkt=f"m/{i}", dec_ts=i * 2 * DAY, res_ts=i * 2 * DAY + DAY,
        price=0.5, outcome=1, alpha=3.0, target=1)
    for i in range(5)
]
_BASE_DS = make_dataset(_BASE_ROWS)
_BASE_CFG = BacktestConfig(observation_time=10 * DAY)


def _run_entry_when(when: dict) -> object:
    spec = make_spec(entry_when=when)
    return run_backtest(spec, _BASE_DS, _BASE_CFG)


def _assert_clean_block(r, *, label: str) -> None:
    """Assert the result is a clean blocked verdict (not a raise, not garbage)."""
    assert r.result_hash == "", f"{label}: expected result_hash=='' (blocked), got {r.result_hash!r}"
    assert not r.validation.ok, f"{label}: expected validation.ok=False"
    codes = {e.code for e in r.validation.errors}
    assert "E_EVIDENCE_SPEC_INVALID" in codes, (
        f"{label}: expected E_EVIDENCE_SPEC_INVALID in error codes; got {codes}"
    )


# ---------------------------------------------------------------------------
# F1(a) — empty all_of → blocked, not raised
# ---------------------------------------------------------------------------

def test_f1a_empty_all_of_returns_blocked_not_raises() -> None:
    """entry.when = {'all_of': []} must block cleanly, not raise ValueError."""
    r = _run_entry_when({"all_of": []})
    _assert_clean_block(r, label="empty_all_of")


# ---------------------------------------------------------------------------
# F1(b) — unknown operator key (typo 'gt' instead of 'gte') → blocked, not raised
# ---------------------------------------------------------------------------

def test_f1b_typo_operator_gt_returns_blocked_not_raises() -> None:
    """entry.when = {'feature':'signal','gt':0.5} (typo) must block cleanly, not raise."""
    r = _run_entry_when({"feature": "alpha", "gt": 2.0})
    _assert_clean_block(r, label="typo_operator_gt")


# ---------------------------------------------------------------------------
# F1(c) — bare feature node with no operator → blocked, not raised
# ---------------------------------------------------------------------------

def test_f1c_bare_feature_no_op_returns_blocked_not_raises() -> None:
    """entry.when = {'feature':'alpha'} (no gte/lte/eq) must block cleanly, not raise."""
    r = _run_entry_when({"feature": "alpha"})
    _assert_clean_block(r, label="bare_feature_no_op")


# ---------------------------------------------------------------------------
# F1 extra — feature_equal missing 'a'/'b' → blocked, not raised
# ---------------------------------------------------------------------------

def test_f1_feature_equal_missing_keys_returns_blocked() -> None:
    """feature_equal with missing 'b' must block cleanly."""
    r = _run_entry_when({"feature_equal": {"a": "alpha"}})
    _assert_clean_block(r, label="feature_equal_missing_b")


# ---------------------------------------------------------------------------
# F1 extra — unknown top-level condition node key → blocked, not raised
# ---------------------------------------------------------------------------

def test_f1_unknown_condition_node_key_returns_blocked() -> None:
    """Unknown top-level condition node key must block cleanly."""
    r = _run_entry_when({"unknown_key": "value"})
    _assert_clean_block(r, label="unknown_node_key")


# ---------------------------------------------------------------------------
# F1 regression — valid specs still pass and produce results
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("when", [
    {"feature": "alpha", "gte": 2.0},
    {"feature": "alpha", "lte": 5.0},
    {"feature": "alpha", "eq": 3.0},
    {"any_of": [{"feature": "alpha", "gte": 2.0}]},
    {"not": {"feature": "alpha", "gte": 10.0}},
    {"feature_equal": {"a": "target", "b": "outcome"}},
    {"all_of": [{"feature": "alpha", "gte": 1.0}, {"feature": "alpha", "lte": 5.0}]},
])
def test_f1_valid_conditions_still_produce_result(when: dict) -> None:
    """Valid condition ASTs must still run cleanly (non-blocked result)."""
    spec = make_spec(entry_when=when)
    r = run_backtest(spec, _BASE_DS, _BASE_CFG)
    # Must not be an error from condition linting (only fails if other validation fails)
    # For these valid conditions the validation should pass
    assert r.validation.ok, (
        f"valid condition {when!r} unexpectedly failed validation: "
        f"{[e.message for e in r.validation.errors]}"
    )
    assert r.result_hash != "", f"valid condition {when!r} unexpectedly blocked"


# ---------------------------------------------------------------------------
# F2 — denormal entry price (1e-203) must not raise OverflowError
# ---------------------------------------------------------------------------

def test_f2_denormal_entry_price_no_raise() -> None:
    """A dataset with one normal (0.5) and one denormal (1e-203) winning row must
    not raise. The result may be blocked or have None for variance-based metrics,
    but run_backtest itself must complete without exception.

    This is the deterministic repro from the harness (inv_overflow_hardening).
    """
    # entry_when: signal >= 0.0 → enters on all rows
    spec = make_spec(
        entry_when={"feature": "alpha", "gte": 0.0},
        slip_bps=0.0,
        fee_bps=0.0,
        starting_capital=1000.0,
    )
    rows = [
        row(mkt="m/0", dec_ts=0,       res_ts=DAY,     price=0.5,    outcome=1, alpha=1.0),
        row(mkt="m/1", dec_ts=2 * DAY, res_ts=3 * DAY, price=1e-203, outcome=1, alpha=1.0),
    ]
    ds = make_dataset(rows)
    cfg = BacktestConfig(observation_time=4 * DAY)

    # Must NOT raise any exception (in particular no OverflowError from pm.py:191)
    r = run_backtest(spec, ds, cfg)

    # If not blocked, all floats in the result's standard + PM metrics must be finite or None
    if r.result_hash != "":
        _assert_metrics_finite_or_none(r, label="F2 denormal price")


def test_f2_result_metrics_finite_or_none() -> None:
    """After F2 fix, std_return_pct and sharpe_trade_level may be None but must
    not be non-finite. Other metric fields must remain finite.
    """
    spec = make_spec(
        entry_when={"feature": "alpha", "gte": 0.0},
        slip_bps=0.0,
        fee_bps=0.0,
        starting_capital=1000.0,
    )
    rows = [
        row(mkt="m/0", dec_ts=0,       res_ts=DAY,     price=0.5,    outcome=1, alpha=1.0),
        row(mkt="m/1", dec_ts=2 * DAY, res_ts=3 * DAY, price=1e-203, outcome=1, alpha=1.0),
    ]
    ds = make_dataset(rows)
    cfg = BacktestConfig(observation_time=4 * DAY)

    r = run_backtest(spec, ds, cfg)
    if r.result_hash == "":
        # Blocked — just confirm no raise happened (already covered above)
        return

    _assert_metrics_finite_or_none(r, label="F2 result metrics")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _assert_metrics_finite_or_none(r: object, *, label: str) -> None:
    """Check that all float fields in standard + PM metrics are finite or None."""
    std = r.metrics.standard  # type: ignore[union-attr]
    pm = r.metrics.pm          # type: ignore[union-attr]

    float_fields: list[tuple[str, object]] = [
        # standard
        ("standard.total_return", std.total_return),
        ("standard.cagr", std.cagr),
        ("standard.sharpe", std.sharpe),
        ("standard.sortino", std.sortino),
        ("standard.max_drawdown", std.max_drawdown),
        ("standard.win_rate", std.win_rate),
        ("standard.starting_capital", std.starting_capital),
        ("standard.ending_capital", std.ending_capital),
        # pm
        ("pm.win_rate_ci95_low", pm.win_rate_ci95_low),
        ("pm.win_rate_ci95_high", pm.win_rate_ci95_high),
        ("pm.mean_return_pct", pm.mean_return_pct),
        ("pm.std_return_pct", pm.std_return_pct),
        ("pm.sharpe_trade_level", pm.sharpe_trade_level),
        ("pm.brier_crowd", pm.brier_crowd),
    ]
    for field, val in float_fields:
        if val is None:
            continue
        assert math.isfinite(float(val)), (
            f"{label}: {field} must be None or finite, got {val!r}"
        )


# ---------------------------------------------------------------------------
# F1(d) — empty {} when-nodes on the MANDATORY entry / yes_payoff blocks.
# make_spec()'s `entry_when or {default}` cannot express an empty when, so we
# build the spec directly. These previously raised in compile_condition
# (residual gap after the first F1 pass); must now block cleanly.
# ---------------------------------------------------------------------------

from ._runner_helpers import EvidenceSpec, SCHEMA_COLUMNS  # noqa: E402

_VALID_ENTRY = {"feature": "alpha", "gte": 2.0}
_VALID_YP = {"feature_equal": {"a": "target", "b": "outcome"}}


def _spec_with_whens(entry_when: dict, yes_payoff_when: dict) -> EvidenceSpec:
    """Build a spec with EXACT when-nodes (bypasses make_spec's `or default`)."""
    return EvidenceSpec.model_validate({
        "spec_family": "pancake-evidence-spec",
        "spec_version": "0.1",
        "name": "test-spec",
        "evidence_dataset_id": "ev_runner_test",
        "schema_requirements": {"required_columns": SCHEMA_COLUMNS},
        "strategy": {
            "side": "YES",
            "entry": {"when": entry_when},
            "yes_payoff": {"when": yes_payoff_when},
            "sizing": {"mode": "fixed_fraction", "value": 0.1},
        },
        "costs": {"slippage_bps": 0.0, "fee_bps": 0.0},
        "starting_capital": 1000.0,
    })


def test_f1d_empty_entry_when_returns_blocked_not_raises() -> None:
    """entry.when = {} must block cleanly, not raise (residual F1 gap)."""
    r = run_backtest(_spec_with_whens({}, _VALID_YP), _BASE_DS, _BASE_CFG)
    _assert_clean_block(r, label="empty_entry_when")


def test_f1d_empty_yes_payoff_when_returns_blocked_not_raises() -> None:
    """yes_payoff.when = {} must block cleanly, not raise (residual F1 gap)."""
    r = run_backtest(_spec_with_whens(_VALID_ENTRY, {}), _BASE_DS, _BASE_CFG)
    _assert_clean_block(r, label="empty_yes_payoff_when")
