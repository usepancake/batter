"""validate_sports_dataset — SportsEventContract validation (Wave E).

Validates a list of sports-event rows against SportsEventContract.
Mirror of validate_reference_dataset (macro.py) for the sports domain.

Required columns per row:
    - market_link (string)        — event identifier
    - decision_time (int)         — epoch-sec entry decision timestamp
    - resolution_time (int)       — epoch-sec event resolution timestamp
    - entry_price (number, (0,1)) — traded-side probability
    - resolved_outcome_numeric (int 0|1) — binary outcome
    - event_id (string)           — unique event identifier
    - league (string)             — league / competition name

Error codes (reused from existing vocabulary):
    E_EVIDENCE_ROWS_MISSING    — zero rows
    E_EVIDENCE_SCHEMA_MISMATCH — required column absent from a row
    E_EVIDENCE_TYPE            — column value has wrong type
    E_EVIDENCE_RANGE           — entry_price out of (0, 1) or outcome not 0/1
"""

from __future__ import annotations

import math
from typing import Any

from .verdict import ValidationVerdict

__all__ = ["validate_sports_dataset"]

# (col_name, type_tag)
_REQUIRED: tuple[tuple[str, str], ...] = (
    ("market_link",              "string"),
    ("decision_time",            "int"),
    ("resolution_time",          "int"),
    ("entry_price",              "number"),
    ("resolved_outcome_numeric", "int"),
    ("event_id",                 "string"),
    ("league",                   "string"),
)


def validate_sports_dataset(rows: list[dict[str, Any]]) -> ValidationVerdict:
    """Validate a list of sports-event rows against SportsEventContract.

    Returns a :class:`ValidationVerdict`.  Accumulates all errors — does
    NOT fail fast.  Callers should check ``verdict.ok`` before using the data.
    """
    v = ValidationVerdict()

    if not rows:
        v.add_error(
            "E_EVIDENCE_ROWS_MISSING",
            "sports dataset has zero rows; cannot register an empty dataset",
        )
        return v

    for i, row in enumerate(rows):
        row_ok = True

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
            continue

        market_link = row["market_link"]
        decision_time = row["decision_time"]
        resolution_time = row["resolution_time"]
        entry_price = row["entry_price"]
        outcome = row["resolved_outcome_numeric"]
        event_id = row["event_id"]
        league = row["league"]

        # --- 2. Type checks ---
        if not isinstance(market_link, str):
            v.add_error(
                "E_EVIDENCE_TYPE",
                f"row {i}: 'market_link' must be a string; got {type(market_link).__name__!r}",
                row_index=i, column="market_link",
            )
            row_ok = False

        if isinstance(decision_time, bool) or not isinstance(decision_time, int):
            v.add_error(
                "E_EVIDENCE_TYPE",
                f"row {i}: 'decision_time' must be an int (epoch sec); "
                f"got {type(decision_time).__name__!r}",
                row_index=i, column="decision_time",
            )
            row_ok = False

        if isinstance(resolution_time, bool) or not isinstance(resolution_time, int):
            v.add_error(
                "E_EVIDENCE_TYPE",
                f"row {i}: 'resolution_time' must be an int (epoch sec); "
                f"got {type(resolution_time).__name__!r}",
                row_index=i, column="resolution_time",
            )
            row_ok = False

        if isinstance(entry_price, bool) or not isinstance(entry_price, (int, float)):
            v.add_error(
                "E_EVIDENCE_TYPE",
                f"row {i}: 'entry_price' must be a number; "
                f"got {type(entry_price).__name__!r}",
                row_index=i, column="entry_price",
            )
            row_ok = False

        if isinstance(outcome, bool) or not isinstance(outcome, int):
            v.add_error(
                "E_EVIDENCE_TYPE",
                f"row {i}: 'resolved_outcome_numeric' must be an int (0 or 1); "
                f"got {type(outcome).__name__!r}",
                row_index=i, column="resolved_outcome_numeric",
            )
            row_ok = False

        if not isinstance(event_id, str):
            v.add_error(
                "E_EVIDENCE_TYPE",
                f"row {i}: 'event_id' must be a string; got {type(event_id).__name__!r}",
                row_index=i, column="event_id",
            )
            row_ok = False

        if not isinstance(league, str):
            v.add_error(
                "E_EVIDENCE_TYPE",
                f"row {i}: 'league' must be a string; got {type(league).__name__!r}",
                row_index=i, column="league",
            )
            row_ok = False

        if not row_ok:
            continue

        # --- 3. Range checks ---
        # entry_price must be in (0, 1) — exclusive bounds (probability for the side)
        if isinstance(entry_price, float) and not math.isfinite(entry_price):
            v.add_error(
                "E_EVIDENCE_RANGE",
                f"row {i}: 'entry_price' must be finite; got {entry_price!r}",
                row_index=i, column="entry_price",
            )
        elif not (0 < entry_price < 1):
            v.add_error(
                "E_EVIDENCE_RANGE",
                f"row {i}: 'entry_price'={entry_price!r} must be in (0, 1) exclusive",
                row_index=i, column="entry_price",
            )

        # resolved_outcome_numeric must be 0 or 1
        if outcome not in (0, 1):
            v.add_error(
                "E_EVIDENCE_RANGE",
                f"row {i}: 'resolved_outcome_numeric'={outcome!r} must be 0 or 1",
                row_index=i, column="resolved_outcome_numeric",
            )

    return v
