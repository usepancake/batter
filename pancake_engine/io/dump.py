"""Canonical JSON result writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from ..canonical import canonical_string
from ..result import BacktestResult

__all__ = ["dump_result", "result_to_canonical_json"]

PathLike = Union[str, Path]


def result_to_canonical_json(result: BacktestResult) -> str:
    """Return the canonical JSON string for a ``BacktestResult``."""
    return canonical_string(result.to_dict())


def dump_result(result: BacktestResult, path: PathLike, *, indent: int | None = None) -> None:
    """Write a ``BacktestResult`` to ``path`` as JSON.

    ``indent=None`` writes canonical JSON (sorted keys, no whitespace).
    ``indent=2`` writes pretty JSON for human inspection.
    """
    if indent is None:
        text = result_to_canonical_json(result)
    else:
        text = json.dumps(result.to_dict(), indent=indent, sort_keys=True, default=_default)
    Path(path).write_text(text, encoding="utf-8")


def _default(o: Any) -> Any:
    # pydantic / dataclass-asdict already flattens, but defensive for misc objects
    if hasattr(o, "to_dict"):
        return o.to_dict()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
