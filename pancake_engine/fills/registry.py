"""Fill-model registry: versioned, named fill implementations.

Design doc: docs/design-0.9.0-contracts-and-fills.md §2.

The registry is engine-side (no user code). Each fill model is a ``FillModel``
implementation keyed by ``(name, version)``.  Unknown name/version →
``ValueError`` with code ``E_EVIDENCE_SPEC_INVALID`` (no silent fallback;
the 0.6.0 always-true lesson applies here too).

Wave 2 ships: ``static_bps@1`` — the exact inline math from
``runner/engine.py::_process_decision`` reproduced bit-for-bit.

Waves 3/4: ``book_replay@1``, ``next_bar_open@1`` (not in scope here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["EntryFill", "FillModel", "resolve"]

BPS_DIVISOR = 10_000


@dataclass(frozen=True)
class EntryFill:
    """Result of ``FillModel.apply_entry()``.

    ``fill_price``  — post-slippage price paid per share.
    ``fee``         — total fee deducted from the notional.
    ``shares``      — shares acquired: ``(notional - fee) / fill_price``.
    """

    fill_price: float
    fee: float
    shares: float


@runtime_checkable
class FillModel(Protocol):
    """Minimal interface the engine needs at decision time.

    The engine calls ``apply_entry`` once per filled decision; the model
    receives the raw quote, the sizing notional, and the cost parameters
    from the spec.  It returns an ``EntryFill`` carrying the three values
    the engine needs to open a ``Position``.

    All implementations MUST be deterministic (IEEE-exact / spec-correctly-
    rounded decimal only).  No I/O, no randomness.
    """

    def apply_entry(
        self,
        *,
        quote: float,
        notional: float,
        slippage_bps: float,
        fee_bps: float,
    ) -> EntryFill:
        """Compute the post-fill price, fee, and share count for an entry.

        Args:
            quote:         pre-slip quote price (entry_price from dataset, already
                           in (0, 1)).
            notional:      cash allocated (= sizing.notional from compute_sizing).
            slippage_bps:  bps from costs.slippage_bps.
            fee_bps:       bps from costs.fee_bps.

        Returns:
            EntryFill with (fill_price, fee, shares).
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


class _StaticBpsV1:
    """static_bps@1 — today's inline math from _process_decision, bit-for-bit.

    fill_price = quote * (1 + slippage_bps / 10_000)
    fee        = notional * (fee_bps / 10_000)
    shares     = (notional - fee) / fill_price

    No params accepted (static_bps@1 has no tunable parameters; a non-empty
    params dict in the spec is rejected at validate_spec).
    """

    def apply_entry(
        self,
        *,
        quote: float,
        notional: float,
        slippage_bps: float,
        fee_bps: float,
    ) -> EntryFill:
        fill_price = quote * (1.0 + slippage_bps / BPS_DIVISOR)
        fee = notional * (fee_bps / BPS_DIVISOR)
        shares = (notional - fee) / fill_price
        return EntryFill(fill_price=fill_price, fee=fee, shares=shares)


# ---------------------------------------------------------------------------
# Registry dict: (name, version) → singleton instance
# ---------------------------------------------------------------------------

_STATIC_BPS_V1 = _StaticBpsV1()

_REGISTRY: dict[tuple[str, int], FillModel] = {
    ("static_bps", 1): _STATIC_BPS_V1,
}


def resolve(name: str, version: int) -> FillModel:
    """Look up a fill model by name + version.

    Returns the registered ``FillModel`` instance.

    Raises:
        ValueError: with message starting ``E_EVIDENCE_SPEC_INVALID`` when the
                    (name, version) pair is not registered (no silent fallback).
    """
    model = _REGISTRY.get((name, version))
    if model is None:
        known = sorted(f"{n}@{v}" for n, v in _REGISTRY)
        raise ValueError(
            f"E_EVIDENCE_SPEC_INVALID: unknown fill model {name!r}@{version}; "
            f"registered models: {known}"
        )
    return model


def default_model() -> FillModel:
    """The implicit default when fill_model is absent from the spec.

    Always static_bps@1.  Called by compile_spec when costs.fill_model is None.
    """
    return _STATIC_BPS_V1
