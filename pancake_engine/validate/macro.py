"""validate_reference_dataset — MacroSignalContract validation (Wave 4).

Validates a list of reference-series rows against MacroSignalContract.
The engine never runs macro datasets directly; this validator is called at
dataset-registration time by the platform before rows are joined into
evidence datasets as feature columns.

Platform flow:
    1. Ingest FRED (or other) series → rows validated here.
    2. Platform left-joins rows into evidence rows on
       ``observation_time ≤ decision_time`` per series.
    3. PM EvidenceSpec declares the joined column as a declared feature;
       the engine treats it like any other feature column — no
       macro-awareness required in the run path.

Required columns per row:
    - series_id (string): identifies the time series (e.g. "UNRATE").
    - observation_time (int, epoch sec): must be monotone non-decreasing
      within a series; no duplicate (series_id, observation_time) pairs.
    - value (number, finite): the observed scalar; any real number is valid
      (unemployment rates, spreads, real interest rates, price indices, …).

Error codes:
    Reused:
      E_EVIDENCE_ROWS_MISSING    — zero rows
      E_EVIDENCE_SCHEMA_MISMATCH — required column absent from a row
      E_EVIDENCE_TYPE            — column value has wrong type
      E_EVIDENCE_MONOTONICITY    — duplicate (series_id, observation_time)
      E_EVIDENCE_RANGE           — value is NaN or Inf (finiteness violated)
    New:
      E_REFERENCE_OBSERVATION_TIME_ORDER — observation_time decreases within
                                           a series (non-monotone non-decreasing)
"""

from __future__ import annotations

import math
from typing import Any

from .verdict import ValidationVerdict

__all__ = ["validate_reference_dataset"]

# Required column names and their expected types (mirrors MacroSignalContract).
_REQUIRED: tuple[tuple[str, str], ...] = (
    ("series_id", "string"),
    ("observation_time", "int"),
    ("value", "number"),
)


def validate_reference_dataset(rows: list[dict[str, Any]]) -> ValidationVerdict:
    """Validate a list of macro reference-series rows.

    Returns a :class:`ValidationVerdict`.  Accumulates all errors — does
    NOT fail fast.  Callers should check ``verdict.ok`` before using the data.
    """
    v = ValidationVerdict()

    if not rows:
        v.add_error(
            "E_EVIDENCE_ROWS_MISSING",
            "reference dataset has zero rows; cannot register an empty series",
        )
        return v

    # Per-series state for monotonicity + dedup checks.
    # last_obs_time: series_id → last seen observation_time (for order check)
    # seen_pairs: series_id → set of observation_times seen (for dedup)
    last_obs_time: dict[str, int] = {}
    seen_pairs: dict[str, set[int]] = {}

    for i, row in enumerate(rows):
        row_ok = True  # tracks whether this row's type/missing checks passed

        # --- 1. Missing column check ---
        for col_name, _ in _REQUIRED:
            if col_name not in row or row[col_name] is None:
                v.add_error(
                    "E_EVIDENCE_SCHEMA_MISMATCH",
                    f"row {i}: required column {col_name!r} is missing or null",
                    row_index=i, column=col_name,
                )
                row_ok = False

        if not row_ok:
            # Cannot do further checks on a row with missing columns.
            continue

        series_id = row["series_id"]
        observation_time = row["observation_time"]
        value = row["value"]

        # --- 2. Type checks ---
        if not isinstance(series_id, str):
            v.add_error(
                "E_EVIDENCE_TYPE",
                f"row {i}: 'series_id' must be a string; got {type(series_id).__name__!r}",
                row_index=i, column="series_id",
            )
            row_ok = False

        if isinstance(observation_time, bool) or not isinstance(observation_time, int):
            v.add_error(
                "E_EVIDENCE_TYPE",
                f"row {i}: 'observation_time' must be an int (epoch sec); "
                f"got {type(observation_time).__name__!r}",
                row_index=i, column="observation_time",
            )
            row_ok = False

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            v.add_error(
                "E_EVIDENCE_TYPE",
                f"row {i}: 'value' must be a number (int or float); "
                f"got {type(value).__name__!r}",
                row_index=i, column="value",
            )
            row_ok = False

        if not row_ok:
            continue

        # --- 3. Finiteness check on value ---
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            v.add_error(
                "E_EVIDENCE_RANGE",
                f"row {i}: 'value' must be finite; got {value!r}",
                row_index=i, column="value",
            )
            # Still continue to monotonicity / dedup checks on the timestamps.

        # --- 4. Monotone non-decreasing check (per series) ---
        series_key = str(series_id)
        if series_key in last_obs_time:
            prev = last_obs_time[series_key]
            if observation_time < prev:
                v.add_error(
                    "E_REFERENCE_OBSERVATION_TIME_ORDER",
                    f"row {i}: observation_time {observation_time} is less than "
                    f"previous observation_time {prev} for series_id={series_id!r} — "
                    "rows must be in non-decreasing order within a series",
                    row_index=i, series_id=series_id,
                    observation_time=observation_time, previous=prev,
                )

        last_obs_time[series_key] = observation_time

        # --- 5. Dedup check: no duplicate (series_id, observation_time) ---
        seen = seen_pairs.setdefault(series_key, set())
        if observation_time in seen:
            v.add_error(
                "E_EVIDENCE_MONOTONICITY",
                f"row {i}: duplicate (series_id={series_id!r}, "
                f"observation_time={observation_time}) — "
                "each (series, timestamp) must be unique",
                row_index=i, series_id=series_id,
                observation_time=observation_time,
            )
        seen.add(observation_time)

    return v
