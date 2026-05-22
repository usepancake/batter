"""Backtest configuration for Pancake Engine 0.3.

The config is part of ``result_hash`` via ``config_hash`` — every field
present in a ``BacktestConfig`` is canonicalized and hashed, so reruns with
the same dataset + spec + config produce the same ``result_hash``.

PR-1 ships a single value for every config knob; the schema is locked here
so the hash shape does not churn when later PRs add modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

__all__ = ["BacktestConfig"]


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
