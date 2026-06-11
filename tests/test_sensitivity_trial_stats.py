"""0.10.2: trial_stats (multiple-testing honesty) on SensitivityResult.

Tests cover:
- bit-identity pin for the expected_max_sharpe extract (refactor must not break DSR)
- base-is-best consistency: dsr_best == deflated_sharpe when base cell is argmax
- null semantics for each field
- determinism (3×)
- to_dict includes trial_stats
- examples hash oracle: result_hash unchanged after the additive field
"""

from __future__ import annotations

import struct

import pytest

from pancake_engine import run_sensitivity_analysis
from pancake_engine.metrics.psr import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)

from ._runner_helpers import make_dataset, make_spec, row

# ── fixtures ──────────────────────────────────────────────────────────────────

# Sub-daily dataset (same as core sensitivity tests): Sharpe degenerate,
# good for structural / null-path tests.
_DAY = 86_400


def _sub_daily_spec_dataset():
    spec = make_spec(side="YES", sizing_value=0.1, entry_when={"feature": "alpha", "gte": 2.5})
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=500, price=0.40, outcome=1, alpha=3.0, target=1),
        row(mkt="m/B", dec_ts=600, res_ts=1000, price=0.55, outcome=0, alpha=3.5, target=0),
        row(mkt="m/C", dec_ts=1100, res_ts=1500, price=0.30, outcome=1, alpha=2.6, target=1),
        row(mkt="m/D", dec_ts=1600, res_ts=2000, price=0.62, outcome=0, alpha=4.0, target=0),
        row(mkt="m/E", dec_ts=2100, res_ts=2500, price=0.25, outcome=1, alpha=2.8, target=1),
        row(mkt="m/F", dec_ts=2600, res_ts=3000, price=0.70, outcome=0, alpha=3.2, target=0),
    ])
    return spec, dataset


def _multi_day_spec_dataset():
    """Multi-day data so daily_returns span multiple UTC days → Sharpe defined."""
    outcomes = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1]
    alphas = [3.0, 3.5, 2.6, 4.0, 2.8, 3.2, 3.1, 2.9, 3.7, 2.7]
    spec = make_spec(side="YES", sizing_value=0.1, entry_when={"feature": "alpha", "gte": 2.5})
    dataset = make_dataset([
        row(mkt=f"m/{i}", dec_ts=i * 5 * _DAY, res_ts=i * 5 * _DAY + 3 * _DAY,
            price=0.5, outcome=outcomes[i], alpha=alphas[i], target=1)
        for i in range(10)
    ])
    return spec, dataset


# ── 1. Bit-identity pin for the expected_max_sharpe refactor ─────────────────

# Pre-computed reference (generated before the refactor):
#   deflated_sharpe_ratio(RETURNS, TRIAL_SHARPES) → 0x3feb47643830db65
_RETURNS = [0.01, -0.02, 0.03, 0.005, -0.01, 0.02, 0.015, -0.008, 0.012, 0.0, 0.04, -0.015]
_TRIAL_SHARPES = [0.02, 0.05, -0.01, 0.08, 0.03, 0.06, 0.00, 0.04, 0.07, -0.02]
_DSR_HEX_PIN = "0x3feb47643830db65"


def _float_hex(v: float) -> str:
    return hex(struct.unpack("Q", struct.pack("d", v))[0])


def test_dsr_bit_identity_after_extract():
    """Refactoring expected_max_sharpe out of deflated_sharpe_ratio must be byte-identical."""
    result = deflated_sharpe_ratio(_RETURNS, _TRIAL_SHARPES)
    assert result is not None
    assert _float_hex(result) == _DSR_HEX_PIN, (
        f"bit-identity broken: got {_float_hex(result)}, expected {_DSR_HEX_PIN}"
    )


# ── 2. expected_max_sharpe null semantics ─────────────────────────────────────

def test_expected_max_sharpe_none_fewer_than_2_trials():
    assert expected_max_sharpe([]) is None
    assert expected_max_sharpe([0.05]) is None


def test_expected_max_sharpe_none_zero_variance():
    assert expected_max_sharpe([0.05, 0.05, 0.05]) is None


def test_expected_max_sharpe_returns_float_for_valid_input():
    em = expected_max_sharpe(_TRIAL_SHARPES)
    assert em is not None
    assert isinstance(em, float)
    assert em > 0  # with dispersion and N>2 the expected max is positive here


def test_expected_max_sharpe_annualized_scales_down():
    # Passing annualized Sharpes should give a smaller expected_max than per-period
    # (divides by sqrt(252) before computing variance → smaller std, smaller result).
    em_pp = expected_max_sharpe(_TRIAL_SHARPES, sharpes_annualized=False)
    em_ann = expected_max_sharpe(_TRIAL_SHARPES, sharpes_annualized=True)
    assert em_pp is not None and em_ann is not None
    assert em_ann < em_pp


# ── 3. trial_stats structure & always-present ─────────────────────────────────

def test_trial_stats_always_present():
    spec, dataset = _sub_daily_spec_dataset()
    res = run_sensitivity_analysis(spec, dataset, n_mc=50, mc_seed=0)
    assert res.trial_stats is not None
    assert isinstance(res.trial_stats, dict)
    assert set(res.trial_stats.keys()) == {"n_trials", "sharpe_best", "dsr_best", "expected_max_sharpe"}


def test_trial_stats_in_to_dict():
    spec, dataset = _sub_daily_spec_dataset()
    res = run_sensitivity_analysis(spec, dataset, n_mc=50, mc_seed=0)
    d = res.to_dict()
    assert "trial_stats" in d
    ts = d["trial_stats"]
    assert set(ts.keys()) == {"n_trials", "sharpe_best", "dsr_best", "expected_max_sharpe"}


def test_trial_stats_n_trials_matches_defined_sharpe_cells():
    spec, dataset = _sub_daily_spec_dataset()
    res = run_sensitivity_analysis(spec, dataset, n_mc=50, mc_seed=0)
    defined = sum(1 for row_vals in res.sharpe_grid for s in row_vals if s is not None)
    assert res.trial_stats["n_trials"] == defined


# ── 4. Null semantics for sharpe_best ────────────────────────────────────────

def test_trial_stats_sharpe_best_null_when_no_defined_cells():
    # All-None sharpe grid (entry above every alpha) → sharpe_best is None.
    spec = make_spec(side="YES", sizing_value=0.1, entry_when={"feature": "alpha", "gte": 99.0})
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=500, price=0.40, outcome=1, alpha=3.0, target=1),
        row(mkt="m/B", dec_ts=600, res_ts=1000, price=0.55, outcome=0, alpha=3.5, target=0),
    ])
    res = run_sensitivity_analysis(spec, dataset, n_mc=10, mc_seed=0)
    # Most/all cells will fire no trades → Sharpe is None for those cells.
    # At gte=99, all cells have None sharpe.
    flat = [s for r in res.sharpe_grid for s in r if s is not None]
    if not flat:
        assert res.trial_stats["sharpe_best"] is None
        assert res.trial_stats["dsr_best"] is None
        # expected_max_sharpe also None when n_trials < 2
        assert res.trial_stats["expected_max_sharpe"] is None


# ── 5. expected_max_sharpe null when n_trials < 2 ────────────────────────────

def test_trial_stats_expected_max_null_when_fewer_than_2_cells():
    # Force a single defined cell by choosing explicit narrow grids.
    # The base cell has alpha=3.5 and gte=3.5 fires exactly one trade.
    # Use a tiny grid so only the base fires.
    spec = make_spec(side="YES", sizing_value=0.1, entry_when={"feature": "alpha", "gte": 3.5})
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=500, price=0.40, outcome=1, alpha=3.5, target=1),
    ])
    # Restrict entry thresholds so only the very-high end fires 0 trades and
    # base fires 1 trade, giving at most 1 defined Sharpe cell across the grid.
    res = run_sensitivity_analysis(
        spec, dataset,
        entry_thresholds=[3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0],
        sizing_fractions=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        n_mc=10, mc_seed=0,
    )
    flat = [s for r in res.sharpe_grid for s in r if s is not None]
    if len(flat) < 2:
        assert res.trial_stats["expected_max_sharpe"] is None


# ── 6. dsr_best null when expected_max is None ───────────────────────────────

def test_trial_stats_dsr_best_null_when_expected_max_null():
    """dsr_best must be None whenever expected_max_sharpe is None."""
    spec, dataset = _sub_daily_spec_dataset()
    res = run_sensitivity_analysis(spec, dataset, n_mc=50, mc_seed=0)
    ts = res.trial_stats
    if ts["expected_max_sharpe"] is None:
        assert ts["dsr_best"] is None


# ── 7. base-is-best consistency: dsr_best == deflated_sharpe ─────────────────

def test_dsr_best_equals_deflated_sharpe_when_base_is_best():
    """When the base cell is the global argmax, dsr_best must equal deflated_sharpe bit-for-bit.

    Strategy: use explicit grids centered tightly on the base so the base cell
    fires the most trades and has the best Sharpe across the multi-day dataset.
    We assert equality when the condition holds, skip otherwise.
    """
    spec, dataset = _multi_day_spec_dataset()
    # Very tight grid around base (gte=2.5, sizing=0.1) so nearby cells have
    # nearly identical parameters and the base often wins.
    entries = [2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7]
    sizings = [0.09, 0.095, 0.1, 0.105, 0.11, 0.115, 0.12]
    res = run_sensitivity_analysis(
        spec, dataset,
        entry_thresholds=entries,
        sizing_fractions=sizings,
        n_mc=200, mc_seed=0,
    )
    ts = res.trial_stats
    base_sharpe = res.sharpe_grid[res.base_entry_idx][res.base_sizing_idx]
    flat = [s for r in res.sharpe_grid for s in r if s is not None]
    if not flat or base_sharpe is None:
        pytest.skip("base cell has no defined Sharpe")
    if max(flat) != base_sharpe:
        pytest.skip("base cell is not the argmax in this run — consistency condition not triggered")
    # The condition holds: dsr_best must equal deflated_sharpe exactly.
    assert ts["dsr_best"] is not None or res.deflated_sharpe is None, (
        "dsr_best unexpectedly None when deflated_sharpe is not None"
    )
    if res.deflated_sharpe is not None and ts["dsr_best"] is not None:
        assert ts["dsr_best"] == res.deflated_sharpe, (
            f"dsr_best {ts['dsr_best']!r} != deflated_sharpe {res.deflated_sharpe!r}"
        )


# ── 8. dsr_best in [0, 1] when defined ───────────────────────────────────────

def test_dsr_best_in_unit_interval_when_defined():
    spec, dataset = _multi_day_spec_dataset()
    res = run_sensitivity_analysis(spec, dataset, n_mc=200, mc_seed=0)
    ts = res.trial_stats
    if ts["dsr_best"] is not None:
        assert 0.0 <= ts["dsr_best"] <= 1.0


def test_sharpe_best_is_max_of_flat_sharpes():
    spec, dataset = _multi_day_spec_dataset()
    res = run_sensitivity_analysis(spec, dataset, n_mc=200, mc_seed=0)
    flat = [s for r in res.sharpe_grid for s in r if s is not None]
    if flat:
        assert res.trial_stats["sharpe_best"] == max(flat)


# ── 9. Determinism (3×) ───────────────────────────────────────────────────────

def test_trial_stats_deterministic():
    spec, dataset = _multi_day_spec_dataset()
    results = [
        run_sensitivity_analysis(spec, dataset, n_mc=200, mc_seed=42).trial_stats
        for _ in range(3)
    ]
    assert results[0] == results[1] == results[2]


# ── 10. examples hash oracle ─────────────────────────────────────────────────

def test_examples_hash_oracle_toy():
    """result_hash of the toy example must be byte-identical to the pinned value.

    This asserts that the sensitivity additive field did NOT touch run_backtest /
    compute_result_hash payload in any way. Mirrors the logic in examples/toy/run.py
    exactly (including BacktestConfig) so the hash comparison is valid.
    """
    import json as _json
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    expected = _json.loads((project_root / "examples" / "toy" / "expected_result.json").read_text())

    from pancake_engine import BacktestConfig, load_dataset, load_spec, run_backtest

    DAY = 86_400
    spec = load_spec(project_root / "examples" / "toy" / "spec.json")
    dataset = load_dataset(project_root / "examples" / "toy" / "dataset.json")
    result = run_backtest(spec, dataset, BacktestConfig(observation_time=50 * DAY))
    assert result.result_hash == expected["result_hash"], (
        f"result_hash changed: got {result.result_hash}, expected {expected['result_hash']}"
    )
