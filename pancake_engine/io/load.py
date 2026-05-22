"""Strict JSON load with duplicate-key detection at parse time.

Python's default ``json.loads`` silently drops duplicate JSON object keys
(last value wins), and the resulting ``dict`` carries no record of the
duplicate. Detection therefore requires an ``object_pairs_hook`` at parse
time — installed here as :func:`_detect_duplicate_keys`.

Direct use of ``json.loads`` without this hook is unsafe for canonical
hashing, because two JSON files that differ only in a duplicate key would
hash identically.

Engine 0.3 is correctness-first, not TS parity. Known TS divergences are
documented in pancake-production/docs/research/pancake-engine-0.3-ts-divergences.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from ..types import EvidenceDataset, EvidenceSpec

__all__ = ["load_json", "load_dataset", "load_spec", "parse_json"]

PathLike = Union[str, Path]


def _detect_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``object_pairs_hook`` that raises on duplicate keys at parse time."""
    seen: set[str] = set()
    for k, _ in pairs:
        if k in seen:
            raise ValueError(f"E_DUPLICATE_KEY: duplicate key '{k}' in JSON object")
        seen.add(k)
    return dict(pairs)


def parse_json(text: str) -> Any:
    """Parse a JSON string with duplicate-key detection."""
    return json.loads(text, object_pairs_hook=_detect_duplicate_keys)


def load_json(path: PathLike) -> Any:
    """Load JSON from ``path`` with duplicate-key detection at parse time."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return parse_json(text)


def load_dataset(path: PathLike) -> EvidenceDataset:
    """Load an ``EvidenceDataset`` JSON file."""
    raw = load_json(path)
    return EvidenceDataset.model_validate(raw)


def load_spec(path: PathLike) -> EvidenceSpec:
    """Load an ``EvidenceSpec`` JSON file."""
    raw = load_json(path)
    return EvidenceSpec.model_validate(raw)
