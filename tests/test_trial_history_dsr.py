"""Ledger Seam — TrialHistory + BacktestResult.deflated block tests.

Coverage:
  - TrialHistory validation (empty, non-finite, empty source)
  - Wiring: block present with plausible dsr; absent when trial_history=None
  - Wiring: absent on fast path (with_inference=False)
  - dsr DECREASES as n_trials grows (same invariant as test_dsr_decreases_with_more_trials)
  - scale handling: annualized=True vs annualized=False give consistent results
  - own_sharpe_included flag is always True in the block
  - n_trials == len(supplied) + 1  (own Sharpe appended)
  - determinism: same inputs → identical block
  - result_hash UNCHANGED by supplying trial_history (critical contract test)
"""

from __future__ import annotations

import math
import sys

import pytest

sys.path.insert(0, "tests")
from _runner_helpers import make_dataset, make_spec, row  # noqa: E402

from pancake_engine import BacktestConfig, run_backtest
from pancake_engine.trials import TrialHistory

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_DAY = 86_400  # seconds


def _spec_and_dataset():
    """Dataset with day-spanning timestamps so daily_returns and Sharpe are defined."""
    spec = make_spec(side="YES", sizing_value=0.1, slip_bps=50, fee_bps=10, starting_capital=1000.0)
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=1 * _DAY, res_ts=10 * _DAY, price=0.5, outcome=1, alpha=3, target=1),
        row(mkt="m/B", dec_ts=15 * _DAY, res_ts=25 * _DAY, price=0.6, outcome=0, alpha=3, target=1),
        row(mkt="m/C", dec_ts=30 * _DAY, res_ts=40 * _DAY, price=0.4, outcome=1, alpha=3, target=1),
        row(mkt="m/D", dec_ts=45 * _DAY, res_ts=55 * _DAY, price=0.55, outcome=1, alpha=3, target=1),
        row(mkt="m/E", dec_ts=60 * _DAY, res_ts=70 * _DAY, price=0.45, outcome=0, alpha=3, target=1),
    ])
    cfg = BacktestConfig(observation_time=80 * _DAY)
    return spec, dataset, cfg


# ---------------------------------------------------------------------------
# TrialHistory validation
# ---------------------------------------------------------------------------

class TestTrialHistoryValidation:
    def test_empty_tuple_raises(self):
        with pytest.raises(ValueError, match="E_TRIAL_HISTORY_EMPTY"):
            TrialHistory(trial_sharpes=(), annualized=True, source="test")

    def test_non_finite_nan_raises(self):
        with pytest.raises(ValueError, match="E_TRIAL_HISTORY_NON_FINITE"):
            TrialHistory(trial_sharpes=(1.0, float("nan")), annualized=True, source="test")

    def test_non_finite_inf_raises(self):
        with pytest.raises(ValueError, match="E_TRIAL_HISTORY_NON_FINITE"):
            TrialHistory(trial_sharpes=(1.0, float("inf")), annualized=True, source="test")

    def test_empty_source_raises(self):
        with pytest.raises(ValueError, match="E_TRIAL_HISTORY_EMPTY_SOURCE"):
            TrialHistory(trial_sharpes=(1.0,), annualized=True, source="")

    def test_whitespace_only_source_raises(self):
        with pytest.raises(ValueError, match="E_TRIAL_HISTORY_EMPTY_SOURCE"):
            TrialHistory(trial_sharpes=(1.0,), annualized=True, source="   ")

    def test_single_valid_entry_ok(self):
        th = TrialHistory(trial_sharpes=(1.5,), annualized=True, source="platform-ledger:session=abc")
        assert len(th.trial_sharpes) == 1

    def test_frozen(self):
        th = TrialHistory(trial_sharpes=(1.0,), annualized=False, source="test")
        with pytest.raises(Exception):
            th.trial_sharpes = (2.0,)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Wiring: deflated block present / absent
# ---------------------------------------------------------------------------

class TestDeflatedBlockWiring:
    def test_block_present_when_trial_history_supplied(self):
        spec, dataset, cfg = _spec_and_dataset()
        th = TrialHistory(trial_sharpes=(1.0, 0.8, 1.2), annualized=True, source="test")
        r = run_backtest(spec, dataset, cfg, trial_history=th)
        assert r.deflated is not None

    def test_block_absent_when_no_trial_history(self):
        spec, dataset, cfg = _spec_and_dataset()
        r = run_backtest(spec, dataset, cfg)
        assert r.deflated is None

    def test_block_absent_on_fast_path(self):
        spec, dataset, cfg = _spec_and_dataset()
        th = TrialHistory(trial_sharpes=(1.0, 0.8), annualized=True, source="test")
        r = run_backtest(spec, dataset, cfg, with_inference=False, trial_history=th)
        assert r.deflated is None

    def test_block_fields(self):
        spec, dataset, cfg = _spec_and_dataset()
        th = TrialHistory(trial_sharpes=(1.0, 0.8, 1.2), annualized=True, source="s:session=x")
        r = run_backtest(spec, dataset, cfg, trial_history=th)
        blk = r.deflated
        assert blk is not None
        assert "dsr" in blk
        assert "n_trials" in blk
        assert "source" in blk
        assert "own_sharpe_included" in blk

    def test_own_sharpe_included_always_true(self):
        spec, dataset, cfg = _spec_and_dataset()
        th = TrialHistory(trial_sharpes=(1.0,), annualized=True, source="test")
        r = run_backtest(spec, dataset, cfg, trial_history=th)
        assert r.deflated is not None
        assert r.deflated["own_sharpe_included"] is True

    def test_n_trials_equals_supplied_plus_one(self):
        spec, dataset, cfg = _spec_and_dataset()
        supplied = (1.0, 0.8, 1.2)
        th = TrialHistory(trial_sharpes=supplied, annualized=True, source="test")
        r = run_backtest(spec, dataset, cfg, trial_history=th)
        assert r.deflated is not None
        assert r.deflated["n_trials"] == len(supplied) + 1

    def test_source_propagated(self):
        spec, dataset, cfg = _spec_and_dataset()
        src = "platform-ledger:search_session=abc123"
        th = TrialHistory(trial_sharpes=(1.0, 0.8), annualized=True, source=src)
        r = run_backtest(spec, dataset, cfg, trial_history=th)
        assert r.deflated is not None
        assert r.deflated["source"] == src

    def test_dsr_is_float_or_none(self):
        spec, dataset, cfg = _spec_and_dataset()
        th = TrialHistory(trial_sharpes=(1.0, 0.8, 1.2), annualized=True, source="test")
        r = run_backtest(spec, dataset, cfg, trial_history=th)
        blk = r.deflated
        assert blk is not None
        assert blk["dsr"] is None or isinstance(blk["dsr"], float)

    def test_dsr_is_in_range_when_defined(self):
        spec, dataset, cfg = _spec_and_dataset()
        th = TrialHistory(trial_sharpes=(1.0, 0.8, 1.2), annualized=True, source="test")
        r = run_backtest(spec, dataset, cfg, trial_history=th)
        blk = r.deflated
        assert blk is not None
        if blk["dsr"] is not None:
            assert 0.0 <= blk["dsr"] <= 1.0

    def test_to_dict_includes_deflated(self):
        spec, dataset, cfg = _spec_and_dataset()
        th = TrialHistory(trial_sharpes=(1.0, 0.8), annualized=True, source="test")
        r = run_backtest(spec, dataset, cfg, trial_history=th)
        d = r.to_dict()
        assert "deflated" in d
        assert d["deflated"] is not None

    def test_to_dict_deflated_none_when_absent(self):
        spec, dataset, cfg = _spec_and_dataset()
        r = run_backtest(spec, dataset, cfg)
        assert r.to_dict()["deflated"] is None


# ---------------------------------------------------------------------------
# DSR decreases as n_trials grows
# ---------------------------------------------------------------------------

class TestDSRDecreaseWithTrials:
    def test_dsr_decreases_as_n_trials_grows(self):
        """More trials → higher multiple-testing bar → lower DSR."""
        spec, dataset, cfg = _spec_and_dataset()
        # Start with 2 trials (min for DSR to be defined), grow to 50.
        prev_dsr = 1.1  # sentinel above [0,1]
        for n in (2, 5, 10, 25, 50):
            sharpes = tuple(1.0 + 0.1 * i for i in range(n))
            th = TrialHistory(trial_sharpes=sharpes, annualized=True, source="test")
            r = run_backtest(spec, dataset, cfg, trial_history=th)
            blk = r.deflated
            assert blk is not None
            if blk["dsr"] is not None:
                assert blk["dsr"] <= prev_dsr + 1e-9, (
                    f"DSR should not increase with more trials: n={n}, dsr={blk['dsr']:.6f}, prev={prev_dsr:.6f}"
                )
                prev_dsr = blk["dsr"]


# ---------------------------------------------------------------------------
# Scale handling: annualized=True vs annualized=False
# ---------------------------------------------------------------------------

class TestScaleHandling:
    def test_annualized_true_and_false_consistent(self):
        """annualized=True with annualized Sharpes should produce same DSR as
        annualized=False with per-period Sharpes (divided by sqrt(252))."""
        import math as _math
        spec, dataset, cfg = _spec_and_dataset()
        ann_sharpes = (1.2, 0.9, 1.5)
        pp_sharpes = tuple(s / _math.sqrt(252) for s in ann_sharpes)

        th_ann = TrialHistory(trial_sharpes=ann_sharpes, annualized=True, source="test")
        th_pp = TrialHistory(trial_sharpes=pp_sharpes, annualized=False, source="test")

        r_ann = run_backtest(spec, dataset, cfg, trial_history=th_ann)
        r_pp = run_backtest(spec, dataset, cfg, trial_history=th_pp)

        blk_ann = r_ann.deflated
        blk_pp = r_pp.deflated
        assert blk_ann is not None and blk_pp is not None

        if blk_ann["dsr"] is not None and blk_pp["dsr"] is not None:
            assert blk_ann["dsr"] == pytest.approx(blk_pp["dsr"], abs=1e-9)

    def test_annualized_false_accepted(self):
        spec, dataset, cfg = _spec_and_dataset()
        th = TrialHistory(trial_sharpes=(0.05, 0.04, 0.06), annualized=False, source="test")
        r = run_backtest(spec, dataset, cfg, trial_history=th)
        assert r.deflated is not None


# ---------------------------------------------------------------------------
# result_hash UNCHANGED by trial_history (CRITICAL contract)
# ---------------------------------------------------------------------------

class TestResultHashContract:
    def test_result_hash_identical_with_and_without_trial_history(self):
        spec, dataset, cfg = _spec_and_dataset()
        th = TrialHistory(trial_sharpes=(1.0, 0.8, 1.2, 0.5), annualized=True, source="test")

        r_without = run_backtest(spec, dataset, cfg)
        r_with = run_backtest(spec, dataset, cfg, trial_history=th)

        assert r_without.result_hash != ""
        assert r_with.result_hash != ""
        assert r_without.result_hash == r_with.result_hash, (
            "result_hash MUST be identical with/without trial_history — "
            "it is an execution argument, not part of the run's spec/config."
        )

    def test_result_hash_not_empty(self):
        spec, dataset, cfg = _spec_and_dataset()
        th = TrialHistory(trial_sharpes=(1.0,), annualized=True, source="test")
        r = run_backtest(spec, dataset, cfg, trial_history=th)
        assert r.result_hash != ""


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_inputs_produce_identical_block(self):
        spec, dataset, cfg = _spec_and_dataset()
        th = TrialHistory(trial_sharpes=(1.0, 0.8, 1.2), annualized=True, source="test")
        r1 = run_backtest(spec, dataset, cfg, trial_history=th)
        r2 = run_backtest(spec, dataset, cfg, trial_history=th)
        assert r1.deflated == r2.deflated

    def test_different_trial_sharpes_produce_different_dsr(self):
        spec, dataset, cfg = _spec_and_dataset()
        th_few = TrialHistory(trial_sharpes=(1.0, 0.8), annualized=True, source="test")
        th_many = TrialHistory(trial_sharpes=(1.0, 0.8, 1.2, 0.5, 0.9, 1.1, 0.7), annualized=True, source="test")
        r_few = run_backtest(spec, dataset, cfg, trial_history=th_few)
        r_many = run_backtest(spec, dataset, cfg, trial_history=th_many)
        # Both should have defined DSR, and they should differ.
        blk_few = r_few.deflated
        blk_many = r_many.deflated
        assert blk_few is not None and blk_many is not None
        if blk_few["dsr"] is not None and blk_many["dsr"] is not None:
            assert blk_few["dsr"] != blk_many["dsr"]
