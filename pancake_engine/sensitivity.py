"""ADR-0046 robustness: parameter sweep (entry × sizing) + Monte-Carlo drawdown.

Pure, additive module. Reuses ``run_backtest`` as the per-cell evaluator and the
ADR-0004 metric primitives — no new math. Sensitivity output is NOT a receipt:
it carries no ``result_hash`` and never enters the ``/run`` contract.

Axes (ADR-0046 §2, amended): evidence strategies hold to resolution, so there is
no exit-price lever. The two real levers are the **entry threshold**
(``strategy.entry.when.gte``) and the **sizing fraction** (``strategy.sizing.value``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .__version__ import ENGINE, ENGINE_VERSION
from .config import BacktestConfig
from .metrics.fdr import fdr_control
from .metrics.psr import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from .metrics.series import daily_returns_carry_forward
from .runner.engine import run_backtest
from .types import EvidenceDataset, EvidenceSpec

GRID_N = 7


@dataclass(frozen=True)
class SensitivityResult:
    engine: str
    engine_version: str
    # Sharpe per cell: sharpe_grid[entry_idx][sizing_idx]; None where undefined.
    sharpe_grid: list[list[Optional[float]]]
    # 0.9: total_return per cell (additive, NOT a receipt → no result_hash impact).
    # Same shape as sharpe_grid; None where Sharpe is also None.
    total_return_grid: list[list[Optional[float]]]
    entry_thresholds: list[float]
    sizing_fractions: list[float]
    base_entry_idx: int
    base_sizing_idx: int
    # Monte-Carlo running-drawdown fan: one point per resample step (t = trades
    # elapsed, 0..base_num_trades), each carrying the p5/p25/p50/p75/p95 of the
    # worst-drawdown-so-far across the reshuffles (fraction; 0.10 = 10%).
    mc_drawdown_points: list[dict[str, float]]
    mc_n: int
    mc_seed: int
    base_num_trades: int
    # 0.8 (additive; sensitivity output is NOT a receipt → no result_hash impact):
    # multiple-testing-aware credibility over the sweep. deflated_sharpe is the DSR of
    # the base cell against the expected MAX Sharpe across all swept configs; the fdr_*
    # fields are Benjamini-Yekutieli FDR control over the per-cell one-sided p-values
    # (p = 1 - PSR), answering "how many of the swept configs survive multiple testing".
    deflated_sharpe: Optional[float] = None
    fdr_method: str = "by"
    fdr_n_tested: int = 0
    fdr_n_significant: int = 0
    fdr_min_raw_p: Optional[float] = None
    fdr_min_adjusted_p: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "engine_version": self.engine_version,
            "sharpe_grid": self.sharpe_grid,
            "total_return_grid": self.total_return_grid,
            "entry_thresholds": self.entry_thresholds,
            "sizing_fractions": self.sizing_fractions,
            "base_entry_idx": self.base_entry_idx,
            "base_sizing_idx": self.base_sizing_idx,
            "mc_drawdown_points": self.mc_drawdown_points,
            "mc_n": self.mc_n,
            "mc_seed": self.mc_seed,
            "base_num_trades": self.base_num_trades,
            "deflated_sharpe": self.deflated_sharpe,
            "fdr_method": self.fdr_method,
            "fdr_n_tested": self.fdr_n_tested,
            "fdr_n_significant": self.fdr_n_significant,
            "fdr_min_raw_p": self.fdr_min_raw_p,
            "fdr_min_adjusted_p": self.fdr_min_adjusted_p,
        }


def _find_gte(cond: dict[str, Any]) -> Optional[float]:
    """First ``gte`` value in an entry-condition AST, or None."""
    if isinstance(cond.get("gte"), (int, float)):
        return float(cond["gte"])
    for key in ("all_of", "any_of"):
        for sub in cond.get(key) or []:
            v = _find_gte(sub)
            if v is not None:
                return v
    if isinstance(cond.get("not"), dict):
        return _find_gte(cond["not"])
    return None


def _set_gte(cond: dict[str, Any], new: float) -> dict[str, Any]:
    """Deep-copy a condition AST with the first ``gte`` replaced by ``new``."""
    out = copy.deepcopy(cond)

    def walk(node: dict[str, Any]) -> bool:
        if isinstance(node.get("gte"), (int, float)):
            node["gte"] = new
            return True
        for key in ("all_of", "any_of"):
            for sub in node.get(key) or []:
                if walk(sub):
                    return True
        if isinstance(node.get("not"), dict):
            return walk(node["not"])
        return False

    walk(out)
    return out


def _centered_grid(
    base: float, step: float, lo: float, hi: float, n: int = GRID_N
) -> tuple[list[float], int]:
    """``n`` points centered on ``base`` spaced by ``step``, clamped to [lo, hi].
    Returns (grid, index-nearest-base)."""
    half = n // 2
    grid = [round(min(hi, max(lo, base + (i - half) * step)), 4) for i in range(n)]
    base_idx = min(range(n), key=lambda i: abs(grid[i] - base))
    return grid, base_idx


def run_sensitivity_analysis(
    spec: EvidenceSpec,
    dataset: EvidenceDataset,
    *,
    entry_thresholds: Optional[list[float]] = None,
    sizing_fractions: Optional[list[float]] = None,
    n_mc: int = 1000,
    mc_seed: int = 0,
    config: Optional[BacktestConfig] = None,
) -> SensitivityResult:
    """Run the entry×sizing Sharpe sweep + Monte-Carlo drawdown for a spec.

    Each grid cell is a full ``run_backtest`` against the same dataset/costs with
    only the entry threshold + sizing fraction varied. The MC drawdown reshuffles
    the base cell's realised trade order ``n_mc`` times (deterministic ``mc_seed``).
    """
    if n_mc < 1:
        raise ValueError("E_SENSITIVITY_INVALID_N_MC: n_mc must be >= 1")
    entry = spec.strategy.entry if isinstance(spec.strategy.entry, dict) else {}
    entry_when = entry.get("when", {}) if isinstance(entry.get("when"), dict) else {}
    base_entry = _find_gte(entry_when)
    if base_entry is None:
        raise ValueError(
            "E_SENSITIVITY_NO_ENTRY_THRESHOLD: entry condition has no `gte` lever to sweep"
        )
    base_sizing = float(spec.strategy.sizing.value)

    if entry_thresholds is None:
        # The entry feature is domain-agnostic (a price in [0,1], an alpha score,
        # etc.) — step relative to the base, clamp only to a small positive lower
        # bound. Cells past the feature's real range simply fire no trades → None.
        entry_thresholds, base_entry_idx = _centered_grid(
            base_entry, max(0.01, abs(base_entry) * 0.1), 1e-6, float("inf")
        )
    else:
        base_entry_idx = min(
            range(len(entry_thresholds)), key=lambda i: abs(entry_thresholds[i] - base_entry)
        )
    if sizing_fractions is None:
        sizing_fractions, base_sizing_idx = _centered_grid(
            base_sizing, max(0.01, base_sizing / 3.0), 0.01, 1.0
        )
    else:
        base_sizing_idx = min(
            range(len(sizing_fractions)), key=lambda i: abs(sizing_fractions[i] - base_sizing)
        )

    sharpe_grid: list[list[Optional[float]]] = []
    total_return_grid: list[list[Optional[float]]] = []
    cell_p_values: list[float] = []  # one-sided p = 1 - PSR per defined cell (for FDR)
    base_result = None
    for ei, e in enumerate(entry_thresholds):
        new_entry = {**entry, "when": _set_gte(entry_when, e)}
        sharpe_row: list[Optional[float]] = []
        total_return_row: list[Optional[float]] = []
        for si, s in enumerate(sizing_fractions):
            new_sizing = spec.strategy.sizing.model_copy(update={"value": s})
            new_strategy = spec.strategy.model_copy(
                update={"entry": new_entry, "sizing": new_sizing}
            )
            # with_inference=False: the sweep only reads .sharpe; skipping the
            # per-cell bootstrap CIs + permutation test is the ~50× speedup that
            # keeps the whole sweep inside the request budget (ADR-0046). PSR below
            # is O(n) moments only (no resampling), so it preserves that speedup.
            res = run_backtest(
                spec.model_copy(update={"strategy": new_strategy}),
                dataset,
                config,
                with_inference=False,
            )
            cell_sharpe = res.metrics.standard.sharpe
            sharpe_row.append(cell_sharpe)
            # 0.9: collect total_return for the heatmap surface.
            # Use None only when no trades fired (zero-trade cell → total_return=0 is meaningless).
            # total_return is always defined even when Sharpe is None (degenerate daily returns).
            total_return_row.append(
                res.metrics.standard.total_return
                if res.metrics.standard.num_trades > 0
                else None
            )
            cell_psr = probabilistic_sharpe_ratio(daily_returns_carry_forward(res.equity_curve))
            if cell_psr is not None:
                cell_p_values.append(1.0 - cell_psr)
            if ei == base_entry_idx and si == base_sizing_idx:
                base_result = res
        sharpe_grid.append(sharpe_row)
        total_return_grid.append(total_return_row)

    if base_result is None:  # base indices fell outside the grid (clamped) — run it directly
        # Only the base cell's trades/equity feed the MC fan; its CIs are unused.
        base_result = run_backtest(spec, dataset, config, with_inference=False)

    # Monte-Carlo running-drawdown fan. Per reshuffle, walk the equity in the
    # shuffled trade order and record the worst-drawdown-so-far at each step
    # (same definition as metrics.standard._max_drawdown). Percentile across
    # reshuffles per step → the fan the frontend mc_drawdown block renders.
    starting = base_result.metrics.standard.starting_capital
    pnls = [t.pnl for t in base_result.trades]
    n_steps = len(pnls) + 1  # step 0 = starting capital, no drawdown
    mc_drawdown_points: list[dict[str, float]] = []
    if pnls:
        rng = np.random.default_rng(mc_seed)
        dd = np.empty((n_mc, n_steps), dtype=float)
        for r in range(n_mc):
            eq = starting
            peak = eq
            worst = 0.0
            dd[r, 0] = 0.0
            for i, idx in enumerate(rng.permutation(len(pnls))):
                eq += pnls[idx]
                if eq > peak:
                    peak = eq
                step_dd = (peak - eq) / peak if peak > 0 else 0.0
                if step_dd > worst:
                    worst = step_dd
                dd[r, i + 1] = worst
        pct = np.percentile(dd, [5, 25, 50, 75, 95], axis=0)  # (5, n_steps)
        mc_drawdown_points = [
            {
                "t": float(s),
                "p5": float(pct[0, s]),
                "p25": float(pct[1, s]),
                "p50": float(pct[2, s]),
                "p75": float(pct[3, s]),
                "p95": float(pct[4, s]),
            }
            for s in range(n_steps)
        ]
    else:
        # Zero-trade base cell: still emit the single t=0 starting point so the
        # documented len == base_num_trades + 1 invariant holds (n_steps == 1).
        mc_drawdown_points = [
            {"t": 0.0, "p5": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0}
        ]

    # 0.8: deflated Sharpe (base cell vs the sweep's expected-max) + BHY FDR over the
    # per-cell p-values. Additive — sensitivity carries no result_hash.
    flat_sharpes = [s for grid_row in sharpe_grid for s in grid_row if s is not None]
    base_daily = daily_returns_carry_forward(base_result.equity_curve)
    deflated = (
        deflated_sharpe_ratio(base_daily, flat_sharpes, sharpes_annualized=True)
        if len(flat_sharpes) >= 2
        else None
    )
    fdr = fdr_control(cell_p_values, alpha=0.05, method="by") if cell_p_values else None

    return SensitivityResult(
        engine=ENGINE,
        engine_version=ENGINE_VERSION,
        sharpe_grid=sharpe_grid,
        total_return_grid=total_return_grid,
        entry_thresholds=entry_thresholds,
        sizing_fractions=sizing_fractions,
        base_entry_idx=base_entry_idx,
        base_sizing_idx=base_sizing_idx,
        mc_drawdown_points=mc_drawdown_points,
        mc_n=n_mc,
        mc_seed=mc_seed,
        base_num_trades=base_result.metrics.standard.num_trades,
        deflated_sharpe=deflated,
        fdr_method="by",
        fdr_n_tested=(fdr.n if fdr else 0),
        fdr_n_significant=(fdr.n_significant if fdr else 0),
        fdr_min_raw_p=(fdr.min_raw_p if fdr else None),
        fdr_min_adjusted_p=(fdr.min_adjusted_p if fdr else None),
    )
