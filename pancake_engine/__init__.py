"""Pancake Engine 0.3 — deterministic Python research engine over EvidenceDataset.

Engine 0.3 is correctness-first, not TS parity. Known TS divergences are documented
in pancake-production/docs/research/pancake-engine-0.3-ts-divergences.md.

PR-0 ships the canonicalization substrate and hash interface only.
Runner, metrics, and walk-forward land in later PRs.
"""

from .__version__ import ENGINE, ENGINE_MODE, ENGINE_VERSION, __version__
from .canonical import canonical_string, canonicalize
from .hash import sha256_canonical
from .io.load import load_dataset, load_json, load_spec
from .types import EvidenceDataset, EvidenceSpec

__all__ = [
    "ENGINE",
    "ENGINE_MODE",
    "ENGINE_VERSION",
    "__version__",
    "canonical_string",
    "canonicalize",
    "sha256_canonical",
    "load_dataset",
    "load_json",
    "load_spec",
    "EvidenceDataset",
    "EvidenceSpec",
]
