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
from .metrics.standard import _max_drawdown
from .result import EquityPoint
from .runner.engine import run_backtest
from .types import EvidenceDataset, EvidenceSpec

GRID_N = 7


@dataclass(frozen=True)
class SensitivityResult:
    engine: str
    engine_version: str
    # Sharpe per cell: sharpe_grid[entry_idx][sizing_idx]; None where undefined.
    sharpe_grid: list[list[Optional[float]]]
    entry_thresholds: list[float]
    sizing_fractions: list[float]
    base_entry_idx: int
    base_sizing_idx: int
    # Monte-Carlo worst-drawdown percentiles (fraction; 0.10 = 10%).
    mc_drawdown_p5: float
    mc_drawdown_p25: float
    mc_drawdown_p50: float
    mc_drawdown_p75: float
    mc_drawdown_p95: float
    mc_n: int
    mc_seed: int
    base_num_trades: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "engine_version": self.engine_version,
            "sharpe_grid": self.sharpe_grid,
            "entry_thresholds": self.entry_thresholds,
            "sizing_fractions": self.sizing_fractions,
            "base_entry_idx": self.base_entry_idx,
            "base_sizing_idx": self.base_sizing_idx,
            "mc_drawdown": {
                "p5": self.mc_drawdown_p5,
                "p25": self.mc_drawdown_p25,
                "p50": self.mc_drawdown_p50,
                "p75": self.mc_drawdown_p75,
                "p95": self.mc_drawdown_p95,
            },
            "mc_n": self.mc_n,
            "mc_seed": self.mc_seed,
            "base_num_trades": self.base_num_trades,
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
    base_result = None
    for ei, e in enumerate(entry_thresholds):
        new_entry = {**entry, "when": _set_gte(entry_when, e)}
        row: list[Optional[float]] = []
        for si, s in enumerate(sizing_fractions):
            new_sizing = spec.strategy.sizing.model_copy(update={"value": s})
            new_strategy = spec.strategy.model_copy(
                update={"entry": new_entry, "sizing": new_sizing}
            )
            res = run_backtest(spec.model_copy(update={"strategy": new_strategy}), dataset, config)
            row.append(res.metrics.standard.sharpe)
            if ei == base_entry_idx and si == base_sizing_idx:
                base_result = res
        sharpe_grid.append(row)

    if base_result is None:  # base indices fell outside the grid (clamped) — run it directly
        base_result = run_backtest(spec, dataset, config)

    starting = base_result.metrics.standard.starting_capital
    pnls = [t.pnl for t in base_result.trades]
    rng = np.random.default_rng(mc_seed)
    draws: list[float] = []
    for _ in range(n_mc if pnls else 0):
        eq = starting
        curve = [EquityPoint(t=0, equity=eq)]
        for i, idx in enumerate(rng.permutation(len(pnls))):
            eq += pnls[idx]
            curve.append(EquityPoint(t=i + 1, equity=eq))
        draws.append(_max_drawdown(curve))
    p5, p25, p50, p75, p95 = (
        float(x) for x in np.percentile(np.array(draws or [0.0]), [5, 25, 50, 75, 95])
    )

    return SensitivityResult(
        engine=ENGINE,
        engine_version=ENGINE_VERSION,
        sharpe_grid=sharpe_grid,
        entry_thresholds=entry_thresholds,
        sizing_fractions=sizing_fractions,
        base_entry_idx=base_entry_idx,
        base_sizing_idx=base_sizing_idx,
        mc_drawdown_p5=p5,
        mc_drawdown_p25=p25,
        mc_drawdown_p50=p50,
        mc_drawdown_p75=p75,
        mc_drawdown_p95=p95,
        mc_n=n_mc,
        mc_seed=mc_seed,
        base_num_trades=base_result.metrics.standard.num_trades,
    )
