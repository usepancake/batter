"""Probability of Backtest Overfitting (PBO) via Combinatorial Purged Cross-Validation.

Method: Bailey, Borwein, López de Prado & Zhu 2016 — "The Probability of
Backtest Overfitting" (Journal of Computational Finance 20(2)).

## Algorithm

Input: a panel of per-period returns for N strategy configs over T periods.

In batter's setting the natural panel is the sensitivity sweep's configs
(entry × sizing grid) each evaluated on the SAME dataset.  ``run_backtest``
with ``with_inference=False`` per config produces each config's equity curve;
``daily_returns_carry_forward`` over that curve is the per-config return series.

### CPCV splits

Split the T daily observations into S contiguous groups (``n_groups=8``,
default; must be even and >= 4). For each of the C(S, S/2) combinations (the
full combinatorial enumeration capped at ``_MAX_COMBINATIONS``):

    train = the selected S/2 groups (temporally concatenated in original order)
    test  = the complement S/2 groups (temporally concatenated in original order)

In each split:
1. Rank configs by train-Sharpe. Take the train-winner.
2. Compute the train-winner's rank among all configs on the test set.
3. Compute the relative rank percentile:
       ω_c = 1 − rank_oos(winner) / (N + 1)    [Bailey et al. §3, eq. (4), inverted]
   where rank_oos is the 1-based rank (1 = best, N = worst). Bailey's paper
   ranks 1 = WORST; batter ranks 1 = BEST, so the inversion preserves the
   semantics ω > 0.5 ↔ above-median OOS (see the comment at the computation).
4. Compute the logit:
       λ_c = ln(ω_c / (1 − ω_c))

PBO = fraction of splits where λ_c < 0, i.e. the train-winner's OOS relative
rank is BELOW median (ω < 0.5 → below-median OOS performance).

### Ranking convention (ties)

- Train Sharpe: higher is better; ties broken by config index (lower index wins).
- OOS Sharpe: higher is better. For ω_c we use the rank-based formula with the
  train-winner's OOS Sharpe, where rank 1 = highest (best) OOS Sharpe.
  Configs with None OOS Sharpe are ranked LAST (worst). Ties broken by config
  index (lower index → better rank, i.e. smaller rank number).

### Sharpe convention

Per-period (non-annualised) Sharpe = mean / std (ddof=1), consistent with
``metrics.psr.psr_sharpe_hat``.  This is the same quantity the PSR is built on.
Annualisation is NOT applied — both IS and OOS Sharpes are on the same per-period
scale, so the ranking is invariant to the sqrt(252) factor.

### Purging

With contiguous group splits over daily returns there is no leakage in this
setting: each config's returns are point-in-time equity changes with no label
spanning groups.  ``purge_bars`` (default 0) omits that many daily observations
at each group boundary.  For PM daily returns the default (0) is sound: the
equity curve already reflects resolution at the observation timestamp, so
adjacent groups share no open positions.  The parameter is exposed for future
bar-series use.

### Degenerate conditions

A degenerate result (``degenerate=True``) is emitted — never a crash — when:

- Fewer than 2 configs have a defined Sharpe on the training split of ANY
  split (ranking is undefined with < 2 comparable configs).
- Any group has fewer than 2 daily return observations (Sharpe undefined per
  the PSR convention).

A degenerate result carries ``pbo=float('nan')`` and ``degenerate_reason`` is
a descriptive string.

### Combination-count cap

``_MAX_COMBINATIONS = 70`` (= C(8,4)).  If the natural C(S, S/2) exceeds the
cap, only the first ``_MAX_COMBINATIONS`` combinations (in the fixed
lexicographic order of ``itertools.combinations``) are used.  This is a DoS
guard matching the ``_MAX_RESAMPLES`` pattern in bootstrap.py.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Optional

from .__version__ import ENGINE, ENGINE_VERSION
from .config import BacktestConfig
from .metrics.psr import psr_sharpe_hat
from .metrics.series import daily_returns_carry_forward
from .runner.engine import run_backtest
from .sensitivity import _find_gte, _set_gte
from .types import EvidenceDataset, EvidenceSpec

__all__ = ["PBOResult", "run_pbo_analysis", "_MAX_COMBINATIONS"]

# DoS guard: cap total number of CPCV splits (matches _MAX_RESAMPLES pattern).
# C(8,4) = 70; this is also the default for n_groups=8.
_MAX_COMBINATIONS: int = 70


@dataclass(frozen=True)
class PBOResult:
    """Result of a PBO/CPCV analysis.

    Fields
    ------
    engine, engine_version
        Provenance — same convention as SensitivityResult.
    pbo
        Probability of Backtest Overfitting ∈ [0, 1] = fraction of splits
        where the IS-winner performs BELOW MEDIAN OOS.  ``float('nan')`` when
        ``degenerate=True``.
    n_splits
        Number of CPCV splits evaluated.
    n_configs
        Total number of (entry_threshold × sizing_fraction) configurations.
    logit_distribution
        λ_c = ln(ω_c / (1 − ω_c)) per split.  λ < 0 ↔ below-median OOS.
    oos_rank_distribution
        ω_c = 1 − rank_oos(winner) / (N + 1) per split, each ∈ (0, 1); rank 1 =
        best OOS, so 0.875 means best-of-7 — NOT near-worst.
    degenerate
        True when the analysis cannot be meaningfully completed (see module
        docstring for conditions).
    degenerate_reason
        Human-readable explanation when ``degenerate=True``; None otherwise.
    """

    engine: str
    engine_version: str
    pbo: float  # nan when degenerate
    n_splits: int
    n_configs: int
    logit_distribution: list[float]
    oos_rank_distribution: list[float]
    degenerate: bool
    degenerate_reason: Optional[str]

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "engine_version": self.engine_version,
            "pbo": self.pbo,
            "n_splits": self.n_splits,
            "n_configs": self.n_configs,
            "logit_distribution": self.logit_distribution,
            "oos_rank_distribution": self.oos_rank_distribution,
            "degenerate": self.degenerate,
            "degenerate_reason": self.degenerate_reason,
        }


def _degenerate(reason: str, n_configs: int = 0) -> PBOResult:
    return PBOResult(
        engine=ENGINE,
        engine_version=ENGINE_VERSION,
        pbo=float("nan"),
        n_splits=0,
        n_configs=n_configs,
        logit_distribution=[],
        oos_rank_distribution=[],
        degenerate=True,
        degenerate_reason=reason,
    )


def _rank_configs(
    sharpes: list[Optional[float]],
) -> list[int]:
    """Return 1-based ranks (1 = best = highest Sharpe).

    Configs with None Sharpe are ranked last.  Ties: lower config index gets
    better rank (smaller rank number).

    The rank list is parallel to ``sharpes`` (same length).
    """
    n = len(sharpes)
    # Build (sharpe_or_-inf, config_idx) pairs, sort descending, assign ranks.
    order = sorted(
        range(n),
        key=lambda i: (
            # None sorts last: use -inf as proxy; then tie-break by index (lower=better)
            sharpes[i] if sharpes[i] is not None else float("-inf"),
            -i,  # lower index → larger secondary key → earlier in descending sort
        ),
        reverse=True,
    )
    ranks = [0] * n
    for rank_one_based, cfg_idx in enumerate(order, start=1):
        ranks[cfg_idx] = rank_one_based
    return ranks


def _sharpe_from_returns(returns: list[float]) -> Optional[float]:
    """Per-period Sharpe (mean/std, ddof=1).  None when n<2 or std=0."""
    return psr_sharpe_hat(returns)


def run_pbo_analysis(
    spec: EvidenceSpec,
    dataset: EvidenceDataset,
    *,
    entry_thresholds: Optional[list[float]] = None,
    sizing_fractions: Optional[list[float]] = None,
    n_groups: int = 8,
    purge_bars: int = 0,
    config: Optional[BacktestConfig] = None,
) -> PBOResult:
    """Run PBO/CPCV over the entry × sizing configuration panel.

    Parameters
    ----------
    spec
        The base spec.  Entry and sizing levers are swept identically to
        ``run_sensitivity_analysis`` (reuses ``_find_gte`` / ``_set_gte``).
    dataset
        Full dataset (all periods).
    entry_thresholds
        Explicit list of entry ``gte`` thresholds.  Default: the same 7-point
        centred grid as the sensitivity sweep (derived from spec).  If the spec
        has no ``gte`` lever, defaults to ``[base]`` (single config, likely
        degenerate).
    sizing_fractions
        Explicit list of sizing fractions.  Default: ``[spec.strategy.sizing.value]``
        (single sizing, focus is on entry variation; use explicit list to sweep
        both axes).
    n_groups
        Number of contiguous time groups S.  Must be even and >= 4.  Default 8.
    purge_bars
        Number of daily observations to drop at each group boundary (purge
        leakage gap).  Default 0 (sound for PM daily returns; see module docs).
    config
        Optional ``BacktestConfig`` forwarded to ``run_backtest``.

    Returns
    -------
    PBOResult
        Fields detailed in the class docstring.  Never raises — degenerate
        inputs produce a ``PBOResult`` with ``degenerate=True``.
    """
    # 1. Validate n_groups.
    if n_groups < 4:
        raise ValueError(
            f"E_PBO_INVALID_N_GROUPS: n_groups must be >= 4 (got {n_groups})"
        )
    if n_groups % 2 != 0:
        raise ValueError(
            f"E_PBO_INVALID_N_GROUPS: n_groups must be even (got {n_groups})"
        )

    # 2. Derive the configuration panel (mirrors sensitivity.py lever derivation).
    entry = spec.strategy.entry if isinstance(spec.strategy.entry, dict) else {}
    entry_when = entry.get("when", {}) if isinstance(entry.get("when"), dict) else {}
    base_entry = _find_gte(entry_when)

    if entry_thresholds is None:
        if base_entry is None:
            # No gte lever: single config (the spec as-is).
            entry_thresholds = [0.0]
            # We'll run the spec unchanged for this single entry; it may be degenerate.
        else:
            from .sensitivity import _centered_grid, GRID_N
            entry_thresholds, _ = _centered_grid(
                base_entry, max(0.01, abs(base_entry) * 0.1), 1e-6, float("inf"), GRID_N
            )

    if sizing_fractions is None:
        sizing_fractions = [float(spec.strategy.sizing.value)]

    n_configs = len(entry_thresholds) * len(sizing_fractions)

    if n_configs < 2:
        return _degenerate(
            "E_PBO_INSUFFICIENT_CONFIGS: PBO requires >= 2 configs with defined Sharpe; "
            f"got {n_configs} config(s) total.",
            n_configs=n_configs,
        )

    # 3. Build the full panel of daily-return series, one per config.
    #    Config ordering: (entry_0, sizing_0), (entry_0, sizing_1), ..., (entry_E, sizing_S)
    #    — row-major, same as sensitivity sweep's sharpe_grid ordering.
    config_returns: list[list[float]] = []
    for e in entry_thresholds:
        if base_entry is not None:
            new_entry_cond = {**entry, "when": _set_gte(entry_when, e)}
        else:
            new_entry_cond = entry  # no gte lever: identical for all e
        for s in sizing_fractions:
            new_sizing = spec.strategy.sizing.model_copy(update={"value": s})
            new_strategy = spec.strategy.model_copy(
                update={"entry": new_entry_cond, "sizing": new_sizing}
            )
            res = run_backtest(
                spec.model_copy(update={"strategy": new_strategy}),
                dataset,
                config,
                with_inference=False,
            )
            daily = daily_returns_carry_forward(res.equity_curve)
            config_returns.append(daily)

    # 4. Build contiguous time groups from the UNION of observation days.
    #    All configs share the same dataset/calendar → the day-index space is the
    #    same. Use the longest per-config daily return series as the T axis.
    T = max((len(r) for r in config_returns), default=0)

    if T == 0:
        return _degenerate(
            "E_PBO_NO_RETURNS: all configs produced zero daily returns.",
            n_configs=n_configs,
        )

    # Each group gets floor(T / n_groups) days; the last group absorbs remainder.
    group_size = T // n_groups
    if group_size < 2:
        return _degenerate(
            f"E_PBO_GROUP_TOO_SHORT: T={T} periods split into {n_groups} groups "
            f"yields group_size={group_size} < 2; each group needs >= 2 daily "
            "observations for a defined Sharpe.",
            n_configs=n_configs,
        )

    # Group boundaries: group g covers days [g * group_size, (g+1) * group_size)
    # except the last group which extends to T.
    def _group_slice(g: int) -> tuple[int, int]:
        start = g * group_size
        end = (g + 1) * group_size if g < n_groups - 1 else T
        return start, end

    # 5. Enumerate CPCV splits.
    half = n_groups // 2
    all_combos = list(itertools.combinations(range(n_groups), half))
    if len(all_combos) > _MAX_COMBINATIONS:
        all_combos = all_combos[:_MAX_COMBINATIONS]  # deterministic truncation

    # 6. Per-split: compute IS/OOS Sharpes, rank, logit.
    logit_dist: list[float] = []
    oos_rank_dist: list[float] = []

    for train_groups in all_combos:
        test_groups = tuple(g for g in range(n_groups) if g not in set(train_groups))

        # Collect the day indices for train and test, applying purge_bars.
        def _day_indices(groups: tuple[int, ...]) -> list[int]:
            """Day indices for the given groups, with boundary purging."""
            indices: list[int] = []
            for g in sorted(groups):
                start, end = _group_slice(g)
                # Apply purge gap at the leading boundary (except the very first group
                # in the full series), and trailing boundary (except the very last).
                actual_start = start + purge_bars if start > 0 else start
                actual_end = end - purge_bars if end < T else end
                if actual_end > actual_start:
                    indices.extend(range(actual_start, actual_end))
            return indices

        train_days = _day_indices(train_groups)
        test_days = _day_indices(test_groups)

        if len(train_days) < 2 or len(test_days) < 2:
            # Degenerate split — not enough observations.
            continue

        # Per-config IS and OOS Sharpes.
        is_sharpes: list[Optional[float]] = []
        oos_sharpes: list[Optional[float]] = []
        for cr in config_returns:
            is_rets = [cr[d] for d in train_days if d < len(cr)]
            oos_rets = [cr[d] for d in test_days if d < len(cr)]
            is_sharpes.append(_sharpe_from_returns(is_rets))
            oos_sharpes.append(_sharpe_from_returns(oos_rets))

        # Need >= 2 defined IS Sharpes to rank meaningfully.
        n_defined_is = sum(1 for s in is_sharpes if s is not None)
        if n_defined_is < 2:
            continue

        # IS ranks (1 = best Sharpe).
        is_ranks = _rank_configs(is_sharpes)
        # IS winner = config with rank 1.
        winner_idx = is_ranks.index(1)

        # OOS ranks (1 = best Sharpe).
        oos_ranks = _rank_configs(oos_sharpes)
        winner_oos_rank = oos_ranks[winner_idx]

        # ω_c = relative rank percentile [Bailey et al. §3, eq. (4)].
        # rank 1 = BEST; ω = rank / (N+1); ω close to 0 = winner is best OOS.
        # But the paper's ω is the PERCENTILE so that ω > 0.5 = above median.
        # In Bailey et al. rank 1 = BEST OOS → ω = rank/(N+1) is SMALL for the
        # best.  However the logit λ < 0 should flag BELOW-median OOS performance.
        # To reconcile: ω_c = (N + 1 - rank_oos) / (N + 1) = inverted percentile,
        # so ω_c > 0.5 means the IS-winner ranks above the median OOS,
        # and ω_c < 0.5 means below-median OOS → λ < 0 → counts toward PBO.
        #
        # Equivalently: rank_percentile = winner_oos_rank / (n_configs + 1)
        # where rank_percentile close to 0 = OOS-best → good; close to 1 = OOS-worst.
        # Then ω_c = 1 - rank_percentile, so ω_c ∈ (0,1) and λ = ln(ω/(1-ω)).
        # λ > 0 ↔ ω > 0.5 ↔ IS-winner ranks in top half OOS (overfit = NOT present).
        # λ < 0 ↔ ω < 0.5 ↔ IS-winner ranks in bottom half OOS (overfit = present).
        # PBO = fraction(λ < 0).
        #
        # In the paper's notation (eq.4): ω_̄c = rank_oos(c*) / (N+1), BUT with
        # rank_oos counted from WORST (rank 1 = worst, rank N = best), so a BELOW-
        # median outcome has ω_c < 0.5.  Our _rank_configs uses rank 1 = BEST,
        # so we invert: ω_c = 1 - winner_oos_rank / (n_configs + 1).
        omega = 1.0 - winner_oos_rank / (n_configs + 1)

        # Guard: omega must be strictly in (0, 1) for logit to be finite.
        # With the formula above: omega ∈ (0, n_configs/(n_configs+1)).
        # It can't be exactly 0 (winner_oos_rank <= n_configs) or >= 1 (rank >= 1).
        # Extra clamp for floating point safety.
        eps = 1e-12
        omega = max(eps, min(1.0 - eps, omega))

        lam = math.log(omega / (1.0 - omega))
        logit_dist.append(lam)
        oos_rank_dist.append(omega)

    if not logit_dist:
        return _degenerate(
            "E_PBO_NO_VALID_SPLITS: all CPCV splits were degenerate (insufficient "
            "observations or fewer than 2 configs with defined IS Sharpe).",
            n_configs=n_configs,
        )

    n_splits = len(logit_dist)
    pbo = sum(1.0 for lam in logit_dist if lam < 0.0) / n_splits

    return PBOResult(
        engine=ENGINE,
        engine_version=ENGINE_VERSION,
        pbo=pbo,
        n_splits=n_splits,
        n_configs=n_configs,
        logit_distribution=logit_dist,
        oos_rank_distribution=oos_rank_dist,
        degenerate=False,
        degenerate_reason=None,
    )
