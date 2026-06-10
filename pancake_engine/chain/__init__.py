"""pancake_engine.chain — Wave C: hash-linked deployment record chain.

Public surface:
  ChainBuilder   — append records to a chain, enforcing all invariants.
  ChainRecord    — frozen record dataclass.
  verify_chain   — re-verify a list of ChainRecord objects.
  ChainVerdict   — result of verify_chain.
  ChainTransitionError — typed error for illegal order-state transitions.
"""

from .builder import ChainBuilder
from .errors import ChainTransitionError
from .records import ChainRecord, CHAIN_FORMAT_VERSION
from .verify import ChainVerdict, verify_chain

__all__ = [
    "ChainBuilder",
    "ChainRecord",
    "ChainVerdict",
    "ChainTransitionError",
    "CHAIN_FORMAT_VERSION",
    "verify_chain",
]
