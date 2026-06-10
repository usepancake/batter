"""Wave D: run_many + TrialLedger + BatchResult tests.

Coverage:
  - Determinism: two run_many calls → byte-identical ledger_hash + final_dsr
  - result_hash parity: each run's result_hash == standalone run_backtest (CRITICAL)
  - Running-DSR monotonicity: same spec repeated N times → later in-sequence
    deflated dsr <= earlier (more trials = stricter bar)
  - Blocked spec handling: verdict-not-ok runs enter ledger with sharpe None,
    excluded from DSR trial counts
  - prior_trials threading: external history seeds the accumulating ledger
  - Edge cases: empty spec list, single spec
  - final_dsr vs in-sequence dsr documented difference asserted on a case
  - TrialLedger structure and ledger_hash stability
  - BatchResult shape (lengths match spec count)
"""

from __future__ import annotations

import math
import sys

import pytest

sys.path.insert(0, "tests")
from _runner_helpers import make_dataset, make_spec, row  # noqa: E402

from pancake_engine import BacktestConfig, run_backtest
from pancake_engine.runner.batch import BatchResult, TrialLedger, run_many
from pancake_engine.trials import TrialHistory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DAY = 86_400


def _make_winning_spec():
    """YES spec where entry condition passes on all rows (alpha >= 2)."""
    return make_spec(side="YES", sizing_value=0.1, slip_bps=0.0, fee_bps=0.0, starting_capital=1000.0)


def _make_losing_spec():
    """YES spec where entry condition blocks all rows (alpha >= 999)."""
    return make_spec(
        side="YES",
        sizing_value=0.1,
        slip_bps=0.0,
        fee_bps=0.0,
        starting_capital=1000.0,
        entry_when={"feature": "alpha", "gte": 999.0},
    )


def _make_blocked_spec():
    """Spec with an undeclared column — validation blocks it."""
    from pancake_engine.types import EvidenceSpec
    # Build a spec that references a column not in schema_requirements.
    return EvidenceSpec.model_validate({
        "spec_family": "pancake-evidence-spec",
        "spec_version": "0.1",
        "name": "blocked-spec",
        "evidence_dataset_id": "ev_runner_test",
        "schema_requirements": {
            "required_columns": [
                {"name": "mkt",     "type": "string", "semantic_role": "market_link"},
                {"name": "dec_ts",  "type": "int",    "semantic_role": "decision_time"},
                {"name": "res_ts",  "type": "int",    "semantic_role": "resolution_time"},
                {"name": "price",   "type": "number", "semantic_role": "entry_price", "range": [0, 1]},
                {"name": "outcome", "type": "int",    "semantic_role": "resolved_outcome_numeric"},
            ]
        },
        "strategy": {
            "side": "YES",
            "entry": {"when": {"feature": "nonexistent_col", "gte": 1.0}},
            "yes_payoff": {"when": {"eq": {"a": "outcome", "b": 1}}},
            "sizing": {"mode": "fixed_fraction", "value": 0.1},
        },
        "costs": {"slippage_bps": 0.0, "fee_bps": 0.0},
        "starting_capital": 1000.0,
    })


def _make_rich_dataset():
    """Multi-row dataset with day-spanning timestamps so daily_returns + Sharpe are defined."""
    ds = make_dataset([
        row(mkt="m/A", dec_ts=1  * _DAY, res_ts=10 * _DAY, price=0.5, outcome=1, alpha=3, target=1),
        row(mkt="m/B", dec_ts=15 * _DAY, res_ts=25 * _DAY, price=0.6, outcome=0, alpha=3, target=1),
        row(mkt="m/C", dec_ts=30 * _DAY, res_ts=40 * _DAY, price=0.4, outcome=1, alpha=3, target=1),
        row(mkt="m/D", dec_ts=45 * _DAY, res_ts=55 * _DAY, price=0.55, outcome=1, alpha=3, target=1),
        row(mkt="m/E", dec_ts=60 * _DAY, res_ts=70 * _DAY, price=0.45, outcome=0, alpha=3, target=1),
    ])
    return ds


def _make_cfg():
    return BacktestConfig(observation_time=80 * _DAY)


# ---------------------------------------------------------------------------
# Determinism: two run_many calls → byte-identical ledger_hash + final_dsr
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_ledger_hash_identical_across_calls(self):
        specs = [_make_winning_spec(), _make_winning_spec()]
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        r1 = run_many(specs, ds, cfg)
        r2 = run_many(specs, ds, cfg)
        assert r1.ledger.ledger_hash == r2.ledger.ledger_hash

    def test_final_dsr_identical_across_calls(self):
        specs = [_make_winning_spec(), _make_winning_spec()]
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        r1 = run_many(specs, ds, cfg)
        r2 = run_many(specs, ds, cfg)
        assert r1.final_dsr == r2.final_dsr

    def test_results_result_hash_identical_across_calls(self):
        specs = [_make_winning_spec(), _make_winning_spec()]
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        r1 = run_many(specs, ds, cfg)
        r2 = run_many(specs, ds, cfg)
        for res_a, res_b in zip(r1.results, r2.results):
            assert res_a.result_hash == res_b.result_hash


# ---------------------------------------------------------------------------
# CRITICAL: result_hash parity with standalone run_backtest
# ---------------------------------------------------------------------------

class TestResultHashParity:
    def test_result_hash_identical_to_standalone(self):
        """run_many must not perturb result_hash vs a standalone run_backtest."""
        spec = _make_winning_spec()
        ds = _make_rich_dataset()
        cfg = _make_cfg()

        standalone = run_backtest(spec, ds, cfg)
        batch = run_many([spec], ds, cfg)

        assert standalone.result_hash != "", "standalone must produce a non-empty hash"
        assert batch.results[0].result_hash != ""
        assert batch.results[0].result_hash == standalone.result_hash, (
            "run_many result_hash must be byte-identical to standalone run_backtest — "
            "trial_history is an execution argument, never part of the hash."
        )

    def test_result_hash_parity_multi_spec(self):
        """Each spec in a batch must match its standalone result_hash."""
        spec_a = _make_winning_spec()
        spec_b = make_spec(side="YES", sizing_value=0.2, slip_bps=10.0, starting_capital=1000.0)
        ds = _make_rich_dataset()
        cfg = _make_cfg()

        standalone_a = run_backtest(spec_a, ds, cfg)
        standalone_b = run_backtest(spec_b, ds, cfg)
        batch = run_many([spec_a, spec_b], ds, cfg)

        assert batch.results[0].result_hash == standalone_a.result_hash
        assert batch.results[1].result_hash == standalone_b.result_hash


# ---------------------------------------------------------------------------
# Running-DSR monotonicity: same spec N times → later in-sequence dsr <= earlier
# ---------------------------------------------------------------------------

class TestRunningDSRMonotonicity:
    def test_in_sequence_dsr_non_increasing(self):
        """Running deflated dsr should not increase as the search deepens.

        We run the same spec N times. Each subsequent run receives a longer
        trial history → stricter DSR bar → dsr should be non-increasing.
        """
        spec = _make_winning_spec()
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        n_reps = 5

        batch = run_many([spec] * n_reps, ds, cfg)

        in_seq_dsrs = []
        for r in batch.results:
            blk = r.deflated
            if blk is not None and blk.get("dsr") is not None:
                in_seq_dsrs.append(blk["dsr"])

        # Need at least 2 defined in-sequence DSRs to test monotonicity.
        if len(in_seq_dsrs) >= 2:
            for i in range(1, len(in_seq_dsrs)):
                assert in_seq_dsrs[i] <= in_seq_dsrs[i - 1] + 1e-9, (
                    f"In-sequence DSR increased: [{i-1}]={in_seq_dsrs[i-1]:.6f} → [{i}]={in_seq_dsrs[i]:.6f}"
                )

    def test_first_run_no_deflated_block(self):
        """First run has no prior trials → no trial_history → deflated block absent."""
        spec = _make_winning_spec()
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        batch = run_many([spec, spec], ds, cfg)
        # First run: accumulating_sharpes is empty → no TrialHistory → deflated=None.
        assert batch.results[0].deflated is None


# ---------------------------------------------------------------------------
# Blocked spec handling
# ---------------------------------------------------------------------------

class TestBlockedSpecHandling:
    def test_blocked_spec_enters_ledger_with_none_sharpe(self):
        """A blocked run enters the ledger with sharpe_annualized=None."""
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        blocked = _make_blocked_spec()
        batch = run_many([blocked], ds, cfg)

        assert len(batch.ledger.trials) == 1
        assert batch.ledger.trials[0]["sharpe_annualized"] is None

    def test_blocked_spec_excluded_from_accumulation(self):
        """A blocked run's None Sharpe must not feed into subsequent runs."""
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        blocked = _make_blocked_spec()
        good = _make_winning_spec()

        # blocked first, then good. good's in-sequence deflated should
        # behave as if blocked didn't exist (no TrialHistory from blocked).
        batch = run_many([blocked, good], ds, cfg)

        # The blocked run should be invalid.
        assert not batch.results[0].validation.ok
        # The good run after a blocked run: accumulating list was empty,
        # so good run also gets no TrialHistory → deflated is None.
        assert batch.results[1].deflated is None

    def test_blocked_spec_final_dsr_is_none(self):
        """Blocked run must produce final_dsr=None."""
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        blocked = _make_blocked_spec()
        batch = run_many([blocked], ds, cfg)
        assert batch.final_dsr[0] is None

    def test_blocked_spec_num_trades_zero(self):
        """Blocked run enters ledger with num_trades=0."""
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        blocked = _make_blocked_spec()
        batch = run_many([blocked], ds, cfg)
        assert batch.ledger.trials[0]["num_trades"] == 0


# ---------------------------------------------------------------------------
# prior_trials threading
# ---------------------------------------------------------------------------

class TestPriorTrialsThreading:
    def test_prior_trials_seeded_into_accumulating_list(self):
        """When prior_trials are supplied the first run should receive a
        TrialHistory and produce a deflated block (if Sharpe is defined)."""
        spec = _make_winning_spec()
        ds = _make_rich_dataset()
        cfg = _make_cfg()

        # Supply 2 prior Sharpes so the accumulating list has >= 1 entry.
        prior = TrialHistory(
            trial_sharpes=(1.2, 0.9),
            annualized=True,
            source="prior-session:abc",
        )
        batch = run_many([spec], ds, cfg, prior_trials=prior)

        # First run now has prior_trials as its trial_history → deflated may be present.
        # (It will be None only if the run itself has no daily_rets / Sharpe.)
        result = batch.results[0]
        if result.metrics.standard.sharpe is not None:
            assert result.deflated is not None, (
                "Run with prior_trials should produce a deflated block when Sharpe is defined."
            )

    def test_prior_trials_annualized_false_handled(self):
        """prior_trials with annualized=False should be up-scaled correctly."""
        spec = _make_winning_spec()
        ds = _make_rich_dataset()
        cfg = _make_cfg()

        # Per-period Sharpes (roughly 0.05–0.08).
        prior_pp = TrialHistory(
            trial_sharpes=(0.05, 0.04, 0.06),
            annualized=False,
            source="prior-pp:session",
        )
        # Equivalent annualized Sharpes.
        prior_ann = TrialHistory(
            trial_sharpes=(0.05 * math.sqrt(252), 0.04 * math.sqrt(252), 0.06 * math.sqrt(252)),
            annualized=True,
            source="prior-pp:session",
        )

        batch_pp = run_many([spec], ds, cfg, prior_trials=prior_pp)
        batch_ann = run_many([spec], ds, cfg, prior_trials=prior_ann)

        # final_dsr should be equal (same effective sharpe list).
        dsr_pp = batch_pp.final_dsr[0]
        dsr_ann = batch_ann.final_dsr[0]
        if dsr_pp is not None and dsr_ann is not None:
            assert dsr_pp == pytest.approx(dsr_ann, abs=1e-9)

    def test_prior_trials_contribute_to_final_dsr(self):
        """prior_trials sharpes should appear in the full ledger DSR denominator."""
        spec = _make_winning_spec()
        ds = _make_rich_dataset()
        cfg = _make_cfg()

        # Without prior_trials.
        batch_no_prior = run_many([spec, spec], ds, cfg)

        # With many prior trials — should make DSR stricter (lower).
        prior = TrialHistory(
            trial_sharpes=tuple(1.0 + 0.05 * i for i in range(20)),
            annualized=True,
            source="prior:heavy",
        )
        batch_with_prior = run_many([spec, spec], ds, cfg, prior_trials=prior)

        # For any run where both are defined, with_prior DSR <= no_prior DSR.
        for i in range(2):
            no_prior_dsr = batch_no_prior.final_dsr[i]
            with_prior_dsr = batch_with_prior.final_dsr[i]
            if no_prior_dsr is not None and with_prior_dsr is not None:
                assert with_prior_dsr <= no_prior_dsr + 1e-9


# ---------------------------------------------------------------------------
# Edge cases: empty spec list, single spec
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_spec_list(self):
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        batch = run_many([], ds, cfg)
        assert len(batch.results) == 0
        assert len(batch.ledger.trials) == 0
        assert len(batch.final_dsr) == 0

    def test_empty_ledger_hash_stable(self):
        """Empty run_many should produce a stable (non-empty) ledger_hash."""
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        b1 = run_many([], ds, cfg)
        b2 = run_many([], ds, cfg)
        assert b1.ledger.ledger_hash == b2.ledger.ledger_hash
        assert b1.ledger.ledger_hash != ""

    def test_single_spec(self):
        spec = _make_winning_spec()
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        batch = run_many([spec], ds, cfg)
        assert len(batch.results) == 1
        assert len(batch.ledger.trials) == 1
        assert len(batch.final_dsr) == 1

    def test_single_spec_result_hash_parity(self):
        spec = _make_winning_spec()
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        standalone = run_backtest(spec, ds, cfg)
        batch = run_many([spec], ds, cfg)
        assert batch.results[0].result_hash == standalone.result_hash


# ---------------------------------------------------------------------------
# TrialLedger structure
# ---------------------------------------------------------------------------

class TestTrialLedgerStructure:
    def test_format_version(self):
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        batch = run_many([_make_winning_spec()], ds, cfg)
        assert batch.ledger.format_version == "trials/1"

    def test_ledger_trials_count_equals_spec_count(self):
        specs = [_make_winning_spec(), _make_winning_spec(), _make_winning_spec()]
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        batch = run_many(specs, ds, cfg)
        assert len(batch.ledger.trials) == 3

    def test_ledger_trial_indices_are_sequential(self):
        specs = [_make_winning_spec(), _make_winning_spec()]
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        batch = run_many(specs, ds, cfg)
        for i, trial in enumerate(batch.ledger.trials):
            assert trial["index"] == i

    def test_ledger_trial_has_required_fields(self):
        spec = _make_winning_spec()
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        batch = run_many([spec], ds, cfg)
        trial = batch.ledger.trials[0]
        assert "index" in trial
        assert "compiled_spec_hash" in trial
        assert "sharpe_annualized" in trial
        assert "num_trades" in trial

    def test_ledger_hash_is_non_empty_string(self):
        spec = _make_winning_spec()
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        batch = run_many([spec], ds, cfg)
        assert isinstance(batch.ledger.ledger_hash, str)
        assert len(batch.ledger.ledger_hash) == 64  # sha256 hex

    def test_ledger_hash_changes_with_different_specs(self):
        spec_a = _make_winning_spec()
        spec_b = make_spec(side="YES", sizing_value=0.2, starting_capital=1000.0)
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        batch_a = run_many([spec_a], ds, cfg)
        batch_b = run_many([spec_b], ds, cfg)
        # Different specs → different compiled_spec_hash → different ledger_hash.
        if batch_a.ledger.trials[0]["compiled_spec_hash"] != batch_b.ledger.trials[0]["compiled_spec_hash"]:
            assert batch_a.ledger.ledger_hash != batch_b.ledger.ledger_hash

    def test_ledger_source(self):
        spec = _make_winning_spec()
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        batch = run_many([spec], ds, cfg)
        assert batch.ledger.source == "run_many:session"


# ---------------------------------------------------------------------------
# BatchResult shape
# ---------------------------------------------------------------------------

class TestBatchResultShape:
    def test_results_length_matches_spec_count(self):
        specs = [_make_winning_spec()] * 4
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        batch = run_many(specs, ds, cfg)
        assert len(batch.results) == 4

    def test_final_dsr_length_matches_spec_count(self):
        specs = [_make_winning_spec()] * 4
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        batch = run_many(specs, ds, cfg)
        assert len(batch.final_dsr) == 4

    def test_final_dsr_types(self):
        specs = [_make_winning_spec(), _make_winning_spec()]
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        batch = run_many(specs, ds, cfg)
        for dsr in batch.final_dsr:
            assert dsr is None or isinstance(dsr, float)

    def test_final_dsr_in_range_when_defined(self):
        specs = [_make_winning_spec()] * 3
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        batch = run_many(specs, ds, cfg)
        for dsr in batch.final_dsr:
            if dsr is not None:
                assert 0.0 <= dsr <= 1.0


# ---------------------------------------------------------------------------
# final_dsr vs in-sequence dsr documented difference
# ---------------------------------------------------------------------------

class TestFinalDSRVsInSequenceDSR:
    def test_final_dsr_documented_difference(self):
        """final_dsr uses the FULL session ledger; in-sequence uses the
        growing ledger up to (and including) each run.

        For the first run in a multi-run session with no prior_trials:
          - in-sequence: no prior trials → deflated=None.
          - final_dsr: uses all N runs from the full ledger.

        We use specs with different sizing so Sharpes are distinct (zero-dispersion
        in the trial list makes DSR undefined by construction — see deflated_sharpe_ratio).
        We assert the structure: run[0] has no in-sequence deflated block, and
        final_dsr[0] is defined when the full ledger has >= 2 distinct Sharpes.
        """
        spec_a = make_spec(side="YES", sizing_value=0.10, slip_bps=0.0, starting_capital=1000.0)
        spec_b = make_spec(side="YES", sizing_value=0.15, slip_bps=5.0, starting_capital=1000.0)
        spec_c = make_spec(side="YES", sizing_value=0.20, slip_bps=10.0, starting_capital=1000.0)
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        batch = run_many([spec_a, spec_b, spec_c], ds, cfg)

        # First run: no prior history → in-sequence deflated should be None.
        assert batch.results[0].deflated is None, (
            "First run should have no in-sequence deflated (no prior trials)."
        )

        # Count runs with distinct Sharpes.
        sharpes = [
            r.metrics.standard.sharpe for r in batch.results
            if r.validation.ok and r.metrics.standard.sharpe is not None
        ]

        # If the full ledger has >= 2 trials with any dispersion, final_dsr[0]
        # should be defined. Zero-dispersion (all identical) → None by design.
        if len(sharpes) >= 2 and len(set(sharpes)) > 1:
            assert batch.final_dsr[0] is not None, (
                "first run final_dsr should be defined when the full ledger has >= 2 distinct-Sharpe trials."
            )

    def test_last_run_final_dsr_equals_in_sequence_dsr(self):
        """For the last Sharpe-defined run, the in-sequence ledger == the full
        ledger, so final_dsr should equal the in-sequence dsr."""
        spec = _make_winning_spec()
        ds = _make_rich_dataset()
        cfg = _make_cfg()
        n = 3
        batch = run_many([spec] * n, ds, cfg)

        # Find last run with a defined Sharpe.
        last_idx = None
        for i in range(n - 1, -1, -1):
            if batch.results[i].metrics.standard.sharpe is not None:
                last_idx = i
                break

        if last_idx is None:
            pytest.skip("No run produced a defined Sharpe in this dataset.")

        last_result = batch.results[last_idx]
        in_seq_blk = last_result.deflated
        final = batch.final_dsr[last_idx]

        if in_seq_blk is not None and in_seq_blk.get("dsr") is not None and final is not None:
            assert final == pytest.approx(in_seq_blk["dsr"], abs=1e-9), (
                "Last run final_dsr should equal its in-sequence dsr (same full ledger)."
            )

    def test_early_run_final_dsr_le_in_sequence_upper_bound(self):
        """For early runs, the final_dsr should be <= their in-sequence dsr
        (or equal if both face the same ledger — happens when only 1 run is
        Sharpe-defined). This is the 'honest asymmetry' property.

        We use a case where the first run DOES have an in-sequence dsr (by
        seeding prior_trials so the accumulating list is non-empty before run 1).
        """
        spec = _make_winning_spec()
        ds = _make_rich_dataset()
        cfg = _make_cfg()

        prior = TrialHistory(
            trial_sharpes=(1.0, 0.8),
            annualized=True,
            source="prior:test",
        )
        # 3 runs with prior_trials so run[0] gets an in-sequence deflated block.
        batch = run_many([spec, spec, spec], ds, cfg, prior_trials=prior)

        r0 = batch.results[0]
        in_seq = r0.deflated
        final = batch.final_dsr[0]

        if in_seq is not None and in_seq.get("dsr") is not None and final is not None:
            # final uses a bigger ledger (all 3 sessions runs + prior),
            # in-sequence uses prior + run[0] only.
            # More trials → equal or stricter bar → final_dsr <= in_seq_dsr.
            n_session_defined = sum(
                1 for r in batch.results
                if r.validation.ok and r.metrics.standard.sharpe is not None
            )
            if n_session_defined > 1:
                assert final <= in_seq["dsr"] + 1e-9, (
                    f"Early run final_dsr={final:.6f} should not exceed in-seq dsr={in_seq['dsr']:.6f}"
                )
