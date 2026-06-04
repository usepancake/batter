"""SHA-256 over canonical bytes for cross-runtime stable hashing.

Engine 0.3 is correctness-first, not TS parity. Known TS divergences are
documented in docs/math-audit-0.4.md.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .canonical import canonicalize

__all__ = ["sha256_canonical"]


def sha256_canonical(obj: Any) -> str:
    """Return the lowercase hex SHA-256 digest of ``canonicalize(obj)``."""
    return hashlib.sha256(canonicalize(obj)).hexdigest()
