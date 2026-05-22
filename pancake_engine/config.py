"""Backtest configuration for Pancake Engine 0.3.

The config is part of ``result_hash`` via ``config_hash`` — every field
present in a ``BacktestConfig`` is canonicalized and hashed, so reruns with
the same dataset + spec + config produce the same ``result_hash``.

PR-1 ships a single value for every config knob; the schema is locked here
so the hash shape does not churn when later PRs add modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

__all__ = ["BacktestConfig", "WalkforwardConfig"]


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a single ``run_backtest`` call.

    All fields default to PR-1's only-supported values. Later PRs may add
    new options to each field, but the field set itself is stable.
    """

    observation_time: Optional[int] = None
    """Unix seconds. If ``None``, the engine derives from dataset
    (see architecture §observation_time rule)."""

    engine_mode: Literal["event_time_v1"] = "event_time_v1"
    sizing_basis: Literal["available_cash"] = "available_cash"
    mark_policy: Literal["mark_at_cost"] = "mark_at_cost"
    slippage_model: Literal["multiplicative_bps"] = "multiplicative_bps"

    def canonical_dict(self) -> dict[str, Any]:
        """Return the config in a stable, hashable shape.

        Keys are sorted by ``canonicalize`` at hash time, but we still pin
        the shape explicitly so the schema is auditable. Fields with
        default values are always included to avoid hash-shape churn.
        """
        return {
            "engine_mode": self.engine_mode,
            "mark_policy": self.mark_policy,
            "observation_time": self.observation_time,
            "sizing_basis": self.sizing_basis,
            "slippage_model": self.slippage_model,
        }


# -----------------------------------------------------------------------------
# Walk-forward config (PR-2)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkforwardConfig:
    """Configuration for a single ``run_walkforward`` call.

    PR-2 ships frozen-spec evaluation only. The ``train_horizon`` is metadata
    for documentation — the engine never reads train rows in 0.3.
    """

    window_type: Literal["expanding", "rolling", "anchored"]
    """Fold-generation mode."""

    test_horizon: Union[int, str]
    """Test window width. Integer = unix seconds. String = pandas-style anchor
    (e.g., ``MS`` = 1 month start, ``3MS`` = 3 months, ``QS`` = 1 quarter,
    ``2QS`` = 6 months). PR-2 supports ``MS`` and ``QS`` only."""

    step: Union[int, str]
    """Step between fold test_window starts. Same units as ``test_horizon``."""

    train_horizon: Union[int, str, None] = None
    """Required for ``rolling`` mode; ignored otherwise. Same units as
    ``test_horizon``. Metadata only — engine does not read train rows."""

    min_test_rows: int = 20
    """Below this, the fold's BacktestResult will fire ``LOW_TRADES_IN_FOLD``
    (handled by per-fold credibility check at trade-count level)."""

    min_fold_count: int = 3
    """Minimum number of folds the schedule must produce. Below 2 errors
    (``E_WALKFORWARD_INSUFFICIENT_FOLDS``). At exactly 2 an
    ``OVERRIDE_MIN_FOLD_COUNT`` info-warning is emitted."""

    resolution_policy: Literal[
        "allow_overhang", "skip_overhang", "truncate_at_window_end"
    ] = "allow_overhang"
    """Behavior for trades whose ``resolution_time`` exceeds the fold's
    ``test_window`` end. ``truncate_at_window_end`` raises in PR-2 (requires
    ``mark_at_last_observed_price`` which is PR-3+)."""

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "min_fold_count": self.min_fold_count,
            "min_test_rows": self.min_test_rows,
            "resolution_policy": self.resolution_policy,
            "step": self.step,
            "test_horizon": self.test_horizon,
            "train_horizon": self.train_horizon,
            "window_type": self.window_type,
        }
