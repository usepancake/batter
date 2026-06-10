"""Fill-model registry: versioned, named fill implementations.

Design doc: docs/design-0.9.0-contracts-and-fills.md §2.

The registry is engine-side (no user code). Each fill model is a ``FillModel``
implementation keyed by ``(name, version)``.  Unknown name/version →
``ValueError`` with code ``E_EVIDENCE_SPEC_INVALID`` (no silent fallback;
the 0.6.0 always-true lesson applies here too).

Wave 2 ships: ``static_bps@1`` — the exact inline math from
``runner/engine.py::_process_decision`` reproduced bit-for-bit.

Wave 3 ships: ``next_bar_open@1`` — ADR-0043 locked fill for the
``crypto_ohlcv`` domain.  Side-aware: long buys at ``open*(1+slip)``;
short sells at ``open*(1-slip)`` with asymmetric share accounting.

Wave 4: ``book_replay@1`` (not in scope here).
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
    ``shares``      — shares acquired.

    For a PM / static_bps long entry: shares = (notional - fee) / fill_price.
    For a crypto short entry: shares = notional / fill_price (fee deducted from
    cash separately — see next_bar_open@1 for the full accounting).
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

    ``side`` — optional; "long" (default) or "short".  PM callers omit it;
    crypto callers pass it explicitly.  Static_bps@1 ignores side (PM fills
    are always long-equivalent); next_bar_open@1 requires it.
    """

    def apply_entry(
        self,
        *,
        quote: float,
        notional: float,
        slippage_bps: float,
        fee_bps: float,
        side: str = "long",
    ) -> EntryFill:
        """Compute the post-fill price, fee, and share count for an entry.

        Args:
            quote:         pre-slip quote price.
            notional:      cash allocated (= sizing.notional from compute_sizing).
            slippage_bps:  bps from costs.slippage_bps.
            fee_bps:       bps from costs.fee_bps.
            side:          "long" or "short" (default "long").

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
    Side is ignored (PM fills are always long-equivalent).
    """

    def apply_entry(
        self,
        *,
        quote: float,
        notional: float,
        slippage_bps: float,
        fee_bps: float,
        side: str = "long",
    ) -> EntryFill:
        fill_price = quote * (1.0 + slippage_bps / BPS_DIVISOR)
        fee = notional * (fee_bps / BPS_DIVISOR)
        shares = (notional - fee) / fill_price
        return EntryFill(fill_price=fill_price, fee=fee, shares=shares)


class _NextBarOpenV1:
    """next_bar_open@1 — ADR-0043 locked fill for the crypto_ohlcv domain.

    Fills at the NEXT BAR'S OPEN (the caller passes that open as ``quote``).
    Slippage is multiplicative and side-aware:

        long  → fill_price = quote * (1 + slip)   (buyer pays more)
                fee        = notional * fee_rate
                shares     = (notional - fee) / fill_price

        short → fill_price = quote * (1 - slip)   (seller receives less)
                fee        = notional * fee_rate
                shares     = notional / fill_price  (cash accounting: cash
                             increases by notional-fee; shares = notional/fill)

    This reproduces the run_crypto_ohlcv entry math (run.py lines 134–149)
    bit-for-bit so the registry entry and the runner stay byte-identical.
    No params accepted.
    """

    def apply_entry(
        self,
        *,
        quote: float,
        notional: float,
        slippage_bps: float,
        fee_bps: float,
        side: str = "long",
    ) -> EntryFill:
        slip = slippage_bps / BPS_DIVISOR
        fee_rate = fee_bps / BPS_DIVISOR
        fee = notional * fee_rate
        if side == "long":
            fill_price = quote * (1.0 + slip)
            shares = (notional - fee) / fill_price
        else:  # short
            fill_price = quote * (1.0 - slip)
            # short: cash += notional - fee; shares = notional / fill_price
            # (negative position is the caller's responsibility)
            shares = notional / fill_price if fill_price > 0.0 else 0.0
        return EntryFill(fill_price=fill_price, fee=fee, shares=shares)


# ---------------------------------------------------------------------------
# Registry dict: (name, version) → singleton instance
# ---------------------------------------------------------------------------

_STATIC_BPS_V1 = _StaticBpsV1()
_NEXT_BAR_OPEN_V1 = _NextBarOpenV1()

_REGISTRY: dict[tuple[str, int], FillModel] = {
    ("static_bps", 1): _STATIC_BPS_V1,
    ("next_bar_open", 1): _NEXT_BAR_OPEN_V1,
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
