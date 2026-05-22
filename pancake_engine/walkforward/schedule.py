"""Fold schedule generation: expanding / rolling / anchored.

The schedule is a deterministic function of the dataset's decision_time span
and the ``WalkforwardConfig``. Identical inputs → identical schedule across
runs / OSes / Python versions.
"""

from __future__ import annotations

from typing import Any

from ..config import WalkforwardConfig
from ..types import EvidenceDataset
from ..validate.dataset import RoleLookup
from .result import FoldDefinition
from .window import advance_anchor, next_anchor_boundary, parse_anchor

__all__ = ["build_fold_schedule"]


def build_fold_schedule(
    dataset: EvidenceDataset,
    wf_config: WalkforwardConfig,
    role_lookup: RoleLookup,
) -> list[FoldDefinition]:
    """Build the fold schedule for a dataset under a WF config."""
    rows = dataset.rows_inline or []
    if not rows:
        return []
    decision_time_col = role_lookup["decision_time"]
    dec_times = sorted(int(r[decision_time_col]) for r in rows)
    dataset_start = dec_times[0]
    dataset_end = dec_times[-1] + 1  # half-open

    test_n, test_unit = parse_anchor(wf_config.test_horizon)
    step_n, step_unit = parse_anchor(wf_config.step)
    if test_unit != step_unit and not (
        test_unit in ("MS", "QS") and step_unit in ("MS", "QS")
    ):
        raise ValueError(
            f"E_WALKFORWARD_CONFIG: test_horizon ({test_unit}) and step ({step_unit}) "
            "must use compatible units"
        )

    if wf_config.window_type == "anchored":
        if test_unit == "sec":
            raise ValueError(
                "E_WALKFORWARD_CONFIG: 'anchored' window_type requires MS or QS units, not seconds"
            )

    if wf_config.window_type == "rolling":
        if wf_config.train_horizon is None:
            raise ValueError(
                "E_WALKFORWARD_CONFIG: 'rolling' window_type requires train_horizon to be set"
            )
        train_n, train_unit = parse_anchor(wf_config.train_horizon)
    else:
        train_n, train_unit = None, None

    # First test_window start
    if wf_config.window_type == "anchored":
        # Snap to next anchor boundary at or after dataset_start
        test_start = next_anchor_boundary(dataset_start, test_unit)
    elif wf_config.window_type == "rolling":
        # Test starts after one train_horizon span
        test_start = advance_anchor(dataset_start, train_n, train_unit)
    else:  # expanding
        test_start = dataset_start

    folds: list[FoldDefinition] = []
    idx = 0
    while True:
        test_end = _advance(test_start, test_n, test_unit)
        if test_start >= dataset_end:
            break
        # Train window:
        if wf_config.window_type == "rolling":
            train_start = advance_anchor(test_start, -train_n if train_unit == "sec" else 0,
                                         "sec" if train_unit == "sec" else train_unit)
            # For MS/QS, can't easily go backward — recompute:
            if train_unit != "sec":
                train_start = _advance(test_start, -train_n, train_unit)
            train_end = test_start
        else:
            # expanding / anchored: train is [dataset_start, test_start)
            train_start = dataset_start
            train_end = test_start

        folds.append(FoldDefinition(
            index=idx,
            train_window=(int(train_start), int(train_end)),
            test_window=(int(test_start), int(test_end)),
        ))
        idx += 1
        # Advance
        new_start = _advance(test_start, step_n, step_unit)
        if new_start <= test_start:
            # Defensive — should never happen given parse_anchor guards
            break
        test_start = new_start

    return folds


def _advance(ts: int, n: int, unit: str) -> int:
    """Advance ``ts`` by ``n`` units of ``unit``. Negative ``n`` for backward."""
    if unit == "sec":
        return ts + n
    # MS / QS — use month math
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    months = n if unit == "MS" else n * 3
    new_year = dt.year + (dt.month - 1 + months) // 12
    if (dt.month - 1 + months) % 12 < 0:
        new_year -= 1
        new_month = (dt.month - 1 + months) % 12 + 12 + 1
    else:
        new_month = (dt.month - 1 + months) % 12 + 1
    return int(datetime(new_year, new_month, 1, tzinfo=timezone.utc).timestamp())
