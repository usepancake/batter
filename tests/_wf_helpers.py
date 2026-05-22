"""Shared fixture builders for PR-2 walk-forward tests."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ._runner_helpers import SCHEMA_COLUMNS, make_dataset, make_spec, row


def utc_ts(year: int, month: int, day: int = 1) -> int:
    """Return UTC unix seconds for ``year-month-day`` midnight."""
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())


def make_wf_dataset(
    decision_resolution_pairs: list[tuple[int, int, dict[str, Any]]],
    *,
    provenance: dict[str, Any] | None = None,
):
    """Build a dataset from a list of ``(decision_t, resolution_t, extra)`` tuples.

    ``extra`` is merged into each row (default: alpha=3.0, target=1, outcome=1, price=0.5).
    """
    rows = []
    for i, (d, r, extra) in enumerate(decision_resolution_pairs):
        base = {
            "mkt": f"m/T{i}",
            "dec_ts": d,
            "res_ts": r,
            "price": 0.5,
            "outcome": 1,
            "alpha": 3.0,
            "target": 1,
        }
        base.update(extra)
        rows.append(base)
    ds = make_dataset(rows)
    if provenance is not None:
        # Pydantic extra="allow" passes it through
        ds = ds.model_copy(update={"provenance": provenance})  # type: ignore[arg-type]
    return ds
