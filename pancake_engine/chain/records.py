"""ChainRecord — the frozen, hash-linked record type for the chain.

Hash payload structure (format_version "chain/1"):
{
    "format_version": "chain/1",
    "engine": ENGINE,
    "engine_version": ENGINE_VERSION,
    "seq": int,
    "t": int,
    "kind": str,
    "payload": dict,
    "prev_hash": str,
}

record_hash = sha256_canonical(above dict).

Genesis (seq 0, kind "deploy") MUST carry:
    payload.compiled_spec_hash  — compiled spec identity
    payload.result_hash         — backtest result identity (provenance pin)
    payload.dataset_id          — dataset identity
    payload.starting_cash       — initial cash balance (float, finite, > 0); used by
                                  verify_chain as the anchor for exact P&L roll-forward.
                                  Producers computing total_cash on settlement records
                                  MUST use math.fsum over the same cash_delta sequence
                                  that verify_chain walks — use ChainBuilder.running_cash()
                                  to obtain the current authoritative value without
                                  hand-computing the sum.

prev_hash of genesis is "".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..__version__ import ENGINE, ENGINE_VERSION
from ..hash import sha256_canonical

__all__ = [
    "ChainRecord",
    "CHAIN_FORMAT_VERSION",
    "_compute_record_hash",
]

CHAIN_FORMAT_VERSION = "chain/1"

# Required keys for the genesis (deploy) record payload.
GENESIS_REQUIRED_KEYS = frozenset({"compiled_spec_hash", "result_hash", "dataset_id", "starting_cash"})


@dataclass(frozen=True)
class ChainRecord:
    """A single immutable, hash-linked chain record.

    Fields:
        seq:         0-based dense sequence number.
        t:           Unix seconds (monotone nondecreasing across the chain).
        kind:        Record kind — one of deploy | tick | order_state | fill |
                     settlement | guard | reconciliation.
        payload:     Kind-specific data dict (immutable copy stored as dict).
        prev_hash:   record_hash of the preceding record ("" for genesis).
        record_hash: sha256_canonical of the hash payload.
    """

    seq: int
    t: int
    kind: str
    payload: dict[str, Any]
    prev_hash: str
    record_hash: str


def _compute_record_hash(
    *,
    seq: int,
    t: int,
    kind: str,
    payload: dict[str, Any],
    prev_hash: str,
) -> str:
    """Compute record_hash for a chain record.

    The hash payload includes engine identity so records are not portable
    across engine versions without an explicit migration.
    """
    hash_payload: dict[str, Any] = {
        "format_version": CHAIN_FORMAT_VERSION,
        "engine": ENGINE,
        "engine_version": ENGINE_VERSION,
        "seq": seq,
        "t": t,
        "kind": kind,
        "payload": payload,
        "prev_hash": prev_hash,
    }
    return sha256_canonical(hash_payload)
