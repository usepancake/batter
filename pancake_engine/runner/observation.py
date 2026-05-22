"""``observation_time`` resolution rule (architecture §observation_time).

```
if config.observation_time is set:
    use it.
elif every row has non-null resolved_outcome_numeric:
    derive observation_time = max(row.resolution_time)
else:
    raise E_OBSERVATION_TIME_REQUIRED
```

**No `time.time()` fallback.** Deliberate divergence from TS L123 (which
falls back to ``Date.now()``).
"""

from __future__ import annotations

from typing import Any, Optional

from ..config import BacktestConfig
from ..types import EvidenceDataset

__all__ = ["resolve_observation_time", "ObservationTimeError"]


class ObservationTimeError(ValueError):
    """Raised when ``observation_time`` cannot be derived and was not provided."""


def resolve_observation_time(
    dataset: EvidenceDataset,
    config: BacktestConfig,
    *,
    resolved_outcome_col: str,
    resolution_time_col: str,
) -> tuple[int, bool]:
    """Return ``(observation_time, derived)``.

    ``derived=True`` means the engine should emit an ``OBSERVATION_TIME_DERIVED``
    info-warning. ``derived=False`` means the caller provided an explicit value.

    Raises :class:`ObservationTimeError` if any row is unresolved AND
    ``config.observation_time`` is not set.
    """
    if config.observation_time is not None:
        return int(config.observation_time), False

    rows = dataset.rows_inline or []
    for i, row in enumerate(rows):
        outcome = row.get(resolved_outcome_col)
        if outcome is None:
            raise ObservationTimeError(
                "E_OBSERVATION_TIME_REQUIRED: dataset contains unresolved rows "
                f"(row {i} has null {resolved_outcome_col!r}); pass "
                "config.observation_time explicitly."
            )

    res_times = [int(row[resolution_time_col]) for row in rows]
    if not res_times:
        raise ObservationTimeError(
            "E_OBSERVATION_TIME_REQUIRED: dataset has zero rows; cannot derive observation_time."
        )
    return max(res_times), True
