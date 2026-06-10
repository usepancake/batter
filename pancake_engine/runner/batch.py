"""run_many — deterministic multi-spec orchestration and TrialLedger.

Wave D of batter 0.10.0. This module orchestrates multiple ``run_backtest``
calls in input order (deterministic; no parallelism — correctness first) while
threading an accumulating trial history so the Deflated Sharpe Ratio is
computed against the real search depth at each step.

Convention — the accumulating ledger:
  Run i receives a TrialHistory that contains:
    (a) all annualized Sharpes from prior_trials (if supplied), plus
    (b) the annualized Sharpes of completed runs 0 … i-1.

  The ledger GROWS as the search deepens, so later runs face a stricter DSR
  bar than earlier ones. This asymmetry is honest and deliberate: a researcher
  who tests 20 specs should not get the same significance credit as one who
  tested 2, and the growing ledger encodes exactly that cost.

  A run whose verdict is not OK (spec/dataset validation blocked it) enters
  the TrialLedger with sharpe_annualized=None and is EXCLUDED from the trial
  sharpe list passed to subsequent runs and from the final DSR computation.
  Blocked runs do not represent a genuine sample — including their (undefined)
  Sharpe in the DSR denominator would be numerically meaningless.

Convention — final_dsr vs in-sequence deflated:
  BatchResult.final_dsr[i] is each run's DSR computed AFTER the full session
  closes: the trial list is all N completed runs (Sharpe-defined) plus
  prior_trials. This is the number the platform reports to the user as the
  session's summary significance, because it reflects the complete multiple-
  testing cost of the session. It will be equal to or lower than the in-
  sequence deflated.dsr for early runs (they faced a smaller ledger in-flight)
  and identical for the last Sharpe-defined run (its in-sequence ledger already
  equals the full ledger).

TrialLedger — frozen session record:
  format_version "trials/1" pins the schema for the platform persistence layer.
  ledger_hash is sha256_canonical over the canonical form of the record
  (excluding ledger_hash itself), so it is byte-identical for identical inputs
  regardless of insertion order (which is fixed — input order) or platform.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..hash import sha256_canonical
from ..metrics.psr import deflated_sharpe_ratio
from ..result import BacktestResult
from ..trials import TrialHistory
from ..types import EvidenceDataset, EvidenceSpec
from ..config import BacktestConfig
from .engine import run_backtest

__all__ = ["run_many", "TrialLedger", "BatchResult"]

_FORMAT_VERSION = "trials/1"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialLedger:
    """Frozen session record produced by ``run_many``.

    Fields:
        format_version: Always ``"trials/1"`` — pins the schema for persistence.
        trials: One entry per run in input order.  Each entry carries:
            - index (int): position in the run_many call (0-based)
            - compiled_spec_hash (str): the run's compiled_spec_hash; empty
              string when the run was blocked by validation.
            - sharpe_annualized (float | None): the run's annualized Sharpe;
              None when blocked OR when no trades / undefined Sharpe.
            - num_trades (int): number of closed trades; 0 when blocked.
        source: Provenance tag, e.g. ``"run_many:session=abc123"`` (non-empty).
        ledger_hash: SHA-256 over the canonical form of
            {format_version, trials, source} — deterministic across platforms.

    Blocked runs (verdict not ok) are included in trials for auditability but
    are excluded from DSR computation (sharpe_annualized=None marks them).
    """

    format_version: str
    trials: tuple[dict, ...]
    source: str
    ledger_hash: str

    @staticmethod
    def build(
        results: list[BacktestResult],
        source: str,
    ) -> "TrialLedger":
        """Construct a TrialLedger from a completed list of BacktestResults."""
        trials_list = []
        for i, r in enumerate(results):
            sharpe_ann: Optional[float] = None
            if r.validation.ok and r.metrics.standard.sharpe is not None:
                sharpe_ann = r.metrics.standard.sharpe  # already annualized (×sqrt(252))
            trials_list.append({
                "index": i,
                "compiled_spec_hash": r.compiled_spec_hash or "",
                "sharpe_annualized": sharpe_ann,
                "num_trades": r.metrics.standard.num_trades,
            })

        canonical_body = {
            "format_version": _FORMAT_VERSION,
            "trials": trials_list,
            "source": source,
        }
        ledger_hash = sha256_canonical(canonical_body)

        return TrialLedger(
            format_version=_FORMAT_VERSION,
            trials=tuple(trials_list),
            source=source,
            ledger_hash=ledger_hash,
        )


@dataclass(frozen=True)
class BatchResult:
    """Output of ``run_many``.

    Fields:
        results: BacktestResult for each spec, in input order.  Length equals
            ``len(specs)``.
        ledger: The frozen TrialLedger for this session.
        final_dsr: Per-run DSR computed against the COMPLETE session ledger
            (all N Sharpe-defined runs + prior_trials).  None where:
            - the run was blocked (no valid return series),
            - the run's Sharpe is None,
            - fewer than 2 total trials in the final ledger (DSR undefined),
            - the full-ledger trial-Sharpe list has zero dispersion.
            Length equals ``len(specs)``.
    """

    results: tuple[BacktestResult, ...]
    ledger: TrialLedger
    final_dsr: tuple[Optional[float], ...]


# ---------------------------------------------------------------------------
# run_many
# ---------------------------------------------------------------------------


def run_many(
    specs: list[EvidenceSpec],
    dataset: EvidenceDataset,
    config: Optional[BacktestConfig] = None,
    *,
    with_inference: bool = True,
    prior_trials: Optional[TrialHistory] = None,
) -> BatchResult:
    """Run each spec via ``run_backtest`` in input order, threading an
    accumulating trial history for running DSR.

    Determinism contract:
        - Specs are processed in input order; no parallelism.
        - Same inputs → byte-identical ``ledger.ledger_hash`` and
          ``final_dsr`` on every call, every machine.
        - Each run's ``result_hash`` is byte-identical to a standalone
          ``run_backtest`` call for the same spec (the accumulating
          trial_history is an execution argument, never part of the hash).

    Accumulating ledger threading:
        Run i receives a TrialHistory built from:
          (a) prior_trials.trial_sharpes (if prior_trials supplied), plus
          (b) annualized Sharpes of Sharpe-defined completed runs 0 … i-1.
        When the combined list has fewer than 1 entry (TrialHistory minimum),
        no trial_history is passed to run i (the deflated block stays None).
        Blocked runs are excluded from the accumulating list.

    Args:
        specs: Ordered list of EvidenceSpec objects.
        dataset: Shared EvidenceDataset for all runs.
        config: Optional BacktestConfig (shared; default BacktestConfig()).
        with_inference: Passed to each run_backtest call.
        prior_trials: Optional externally-supplied history from a previous
            session or platform ledger.  Its Sharpes seed the accumulating
            list before any run in this session.

    Returns:
        BatchResult with results, ledger, and final_dsr.
    """
    if config is None:
        config = BacktestConfig()

    # Seed the accumulating sharpe list from prior_trials (annualized scale).
    # We always use annualized=True internally (metrics.standard.sharpe is
    # annualized ×sqrt(252)); prior_trials may supply per-period Sharpes,
    # so we up-scale them to annualized before adding to the ledger.
    accumulating_sharpes: list[float] = []
    if prior_trials is not None:
        if prior_trials.annualized:
            accumulating_sharpes = list(prior_trials.trial_sharpes)
        else:
            accumulating_sharpes = [
                s * math.sqrt(252) for s in prior_trials.trial_sharpes
            ]

    results: list[BacktestResult] = []

    for spec in specs:
        # Build the TrialHistory for this run from the accumulating ledger.
        # TrialHistory requires >= 1 entry; supply None when ledger is empty.
        if accumulating_sharpes:
            th = TrialHistory(
                trial_sharpes=tuple(accumulating_sharpes),
                annualized=True,
                source=prior_trials.source if prior_trials is not None else "run_many:accumulating",
            )
        else:
            th = None

        result = run_backtest(
            spec,
            dataset,
            config,
            with_inference=with_inference,
            trial_history=th,
        )
        results.append(result)

        # Accumulate: add this run's Sharpe if defined (blocked and None excluded).
        if result.validation.ok and result.metrics.standard.sharpe is not None:
            accumulating_sharpes.append(result.metrics.standard.sharpe)

    # Build the TrialLedger from all completed results.
    ledger = TrialLedger.build(results, source="run_many:session")

    # Compute final_dsr: each run's DSR against the COMPLETE session ledger.
    # Full ledger = all Sharpe-defined runs in this session + prior_trials.
    all_session_sharpes: list[float] = [
        entry["sharpe_annualized"]
        for entry in ledger.trials
        if entry["sharpe_annualized"] is not None
    ]

    # Combine with prior_trials (already annualized) for the final denominator.
    prior_sharpes_ann: list[float] = []
    if prior_trials is not None:
        if prior_trials.annualized:
            prior_sharpes_ann = list(prior_trials.trial_sharpes)
        else:
            prior_sharpes_ann = [s * math.sqrt(252) for s in prior_trials.trial_sharpes]

    full_ledger_sharpes = prior_sharpes_ann + all_session_sharpes

    final_dsr_list: list[Optional[float]] = []
    for result in results:
        if not result.validation.ok or result.metrics.standard.sharpe is None:
            final_dsr_list.append(None)
            continue

        # Need the run's daily returns for DSR computation.
        daily_rets = result.metrics.standard  # not directly available on result

        # daily_rets live in the BacktestResult but aren't stored — we need to
        # recompute them. However, BacktestResult.equity_curve IS stored; we
        # use the same helper as run_backtest to reconstruct daily_rets.
        from ..metrics import daily_returns_carry_forward
        daily_rets_list = list(daily_returns_carry_forward(result.equity_curve))

        if not daily_rets_list:
            final_dsr_list.append(None)
            continue

        # Build the full trial sharpe list for this run's final DSR:
        # all full_ledger_sharpes — DSR convention: the run's own Sharpe is
        # INCLUDED in the trial count (it is itself a trial). Since
        # full_ledger_sharpes already contains this run's Sharpe (it was
        # added to all_session_sharpes above), we pass it directly.
        # This mirrors how run_backtest appends own_sharpe to trial_sharpes.
        # IMPORTANT: full_ledger_sharpes are annualized; pass sharpes_annualized=True.
        if len(full_ledger_sharpes) < 2:
            final_dsr_list.append(None)
            continue

        dsr = deflated_sharpe_ratio(
            daily_rets_list,
            full_ledger_sharpes,
            sharpes_annualized=True,
        )
        final_dsr_list.append(dsr)

    return BatchResult(
        results=tuple(results),
        ledger=ledger,
        final_dsr=tuple(final_dsr_list),
    )
