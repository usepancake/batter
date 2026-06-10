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

Wave 4 (0.9.x Wave A): ``book_replay@1`` — PM depth-aware fill.  Walks the
captured L2 ask levels at decision_time to compute a VWAP fill price.
``book_slices`` (list of snapshot dicts) must be supplied via ``apply_entry``
when this model is active; the engine passes them from the book_dataset.
``ttr_fill_adjustment`` param is declared but reserved — specs that request it
are rejected with ``E_EVIDENCE_SPEC_INVALID`` (declared-but-unimplemented
params must hard-fail, never silently no-op).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = ["EntryFill", "FillModel", "FillBlocked", "resolve"]

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
    the engine needs to open a ``Position``, OR a ``FillBlocked`` when the
    model deterministically cannot fill (e.g. book_replay@1 on a missing
    slice / insufficient depth).  ``FillBlocked`` is part of the public
    Protocol — callers MUST handle it; silently treating a blocked fill as
    filled is the always-true bug class.

    All implementations MUST be deterministic (IEEE-exact / spec-correctly-
    rounded decimal only).  No I/O, no randomness.

    ``side`` — optional; "long" (default) or "short".  PM callers omit it;
    crypto callers pass it explicitly.  Static_bps@1 ignores side (PM fills
    are always long-equivalent); next_bar_open@1 requires it.

    ``market_link`` — optional; the market identifier for the current row.
    ``decision_time`` — optional; unix seconds for the current decision.
    ``book_slices`` — optional; list of book snapshot dicts from the
    book_dataset (shape per ADR-0041 column names).  Required by
    book_replay@1; ignored by all other models.
    """

    def apply_entry(
        self,
        *,
        quote: float,
        notional: float,
        slippage_bps: float,
        fee_bps: float,
        side: str = "long",
        market_link: str | None = None,
        decision_time: int | None = None,
        book_slices: list[dict[str, Any]] | None = None,
    ) -> EntryFill | FillBlocked:
        """Compute the post-fill price, fee, and share count for an entry.

        Args:
            quote:         pre-slip quote price.
            notional:      cash allocated (= sizing.notional from compute_sizing).
            slippage_bps:  bps from costs.slippage_bps.
            fee_bps:       bps from costs.fee_bps.
            side:          "long" or "short" (default "long").
            market_link:   market identifier (used by book_replay@1).
            decision_time: unix epoch seconds (used by book_replay@1).
            book_slices:   L2 snapshot rows from book_dataset (book_replay@1 only).

        Returns:
            EntryFill with (fill_price, fee, shares), or FillBlocked when the
            model deterministically cannot fill.
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
    Extra kwargs (market_link, decision_time, book_slices) are ignored.
    """

    def apply_entry(
        self,
        *,
        quote: float,
        notional: float,
        slippage_bps: float,
        fee_bps: float,
        side: str = "long",
        market_link: str | None = None,
        decision_time: int | None = None,
        book_slices: list[dict[str, Any]] | None = None,
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
    Extra kwargs (market_link, decision_time, book_slices) are ignored.
    """

    def apply_entry(
        self,
        *,
        quote: float,
        notional: float,
        slippage_bps: float,
        fee_bps: float,
        side: str = "long",
        market_link: str | None = None,
        decision_time: int | None = None,
        book_slices: list[dict[str, Any]] | None = None,
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


# Sentinel object returned by _BookReplayV1.apply_entry when the fill cannot
# proceed due to missing slice or insufficient depth.  The engine checks for
# this and emits the appropriate warning + skips the row.
@dataclass(frozen=True)
class FillBlocked:
    """Indicates book_replay@1 could not fill; carries the block reason."""

    reason: str  # "BOOK_SLICE_MISSING" | "BOOK_DEPTH_INSUFFICIENT"
    context: dict[str, Any]


class _BookReplayV1:
    """book_replay@1 — PM depth-aware fill (0.9.x Wave A).

    For a buy of N notional at decision_time: find the latest snapshot for
    that market_link with snapshot_time <= decision_time (strictly no future
    book).  Walk ask levels ascending by (price, size), cumulatively consume
    size until notional is filled; volume-weighted average price = fill_price.
    Fee is applied on notional (same as static_bps@1).

    Blocking conditions (no silent fallback — 0.6.0 lesson):
    - No snapshot with snapshot_time <= decision_time → BOOK_SLICE_MISSING
    - Ask depth < notional to fill → BOOK_DEPTH_INSUFFICIENT

    TTR param: ``ttr_fill_adjustment`` declared but reserved.  Specs requesting
    it are rejected at validate_spec with E_EVIDENCE_SPEC_INVALID
    ("reserved for a future version") before apply_entry is ever called.

    Returns an ``EntryFill`` on success or a ``FillBlocked`` sentinel on
    block.  The engine caller checks type and routes accordingly.

    Determinism guarantees:
    - Levels sorted by (price, size) explicitly — never dict order.
    - All arithmetic: IEEE-64 (+, -, *, / and math.fsum only).
    - book_slices filtered and sorted before any arithmetic.
    """

    def apply_entry(
        self,
        *,
        quote: float,
        notional: float,
        slippage_bps: float,
        fee_bps: float,
        side: str = "long",
        market_link: str | None = None,
        decision_time: int | None = None,
        book_slices: list[dict[str, Any]] | None = None,
    ) -> EntryFill | FillBlocked:
        """Walk ask levels to compute VWAP fill, or return a FillBlocked."""
        # 1. Find the latest snapshot with snapshot_time <= decision_time.
        #    Columns per ADR-0041: market_link, snapshot_time, side, level_price,
        #    level_size.  We only consume ASK levels (bid-side irrelevant for buys).
        if book_slices is None:
            book_slices = []

        # Filter: correct market, ask side, snapshot_time strictly <= decision_time.
        dt = decision_time if decision_time is not None else 0
        relevant: list[dict[str, Any]] = [
            s for s in book_slices
            if s.get("market_link") == market_link
            and str(s.get("side", "")).upper() == "ASK"
            and int(s["snapshot_time"]) <= dt
        ]

        if not relevant:
            return FillBlocked(
                reason="BOOK_SLICE_MISSING",
                context={"market_link": market_link, "decision_time": dt},
            )

        # 2. Select the latest snapshot (max snapshot_time).
        latest_ts = max(int(s["snapshot_time"]) for s in relevant)
        levels_at_ts = [s for s in relevant if int(s["snapshot_time"]) == latest_ts]

        # 3. Sort ask levels ascending by (price, size) — deterministic, never dict order.
        levels_at_ts.sort(key=lambda s: (float(s["level_price"]), float(s["level_size"])))

        # 4. Walk levels, cumulatively consume size until notional is filled.
        #    shares = ∑(size_consumed_i); fill_price = fsum(price_i * size_i) / total_shares
        remaining = notional  # cash to spend
        weighted_prices: list[float] = []
        total_shares: float = 0.0

        for lvl in levels_at_ts:
            if remaining <= 0.0:
                break
            price = float(lvl["level_price"])
            available_size = float(lvl["level_size"])  # size in shares
            # cash needed to buy all available_size at this level
            cash_for_level = price * available_size
            if cash_for_level <= remaining:
                # Consume whole level
                weighted_prices.append(price * available_size)
                total_shares += available_size
                remaining -= cash_for_level
            else:
                # Partial level: buy as many shares as remaining cash allows
                shares_here = remaining / price
                weighted_prices.append(price * shares_here)
                total_shares += shares_here
                remaining = 0.0

        if remaining > 0.0:
            # Insufficient depth to fill the full notional
            return FillBlocked(
                reason="BOOK_DEPTH_INSUFFICIENT",
                context={
                    "market_link": market_link,
                    "decision_time": dt,
                    "snapshot_time": latest_ts,
                    "notional": notional,
                    "unfilled_notional": remaining,
                },
            )

        # 5. VWAP fill price = total_cost / total_shares (IEEE-exact via fsum).
        total_cost = math.fsum(weighted_prices)
        fill_price = total_cost / total_shares if total_shares > 0.0 else 0.0

        # 6. Fee on notional (same as static_bps@1).
        fee = notional * (fee_bps / BPS_DIVISOR)
        # shares already computed from the level walk; fee reduces investable but
        # the actual shares are already determined by the depth walk.
        # Convention: shares = total_shares from the walk (depth determines the
        # actual share count); cost = notional (pre-fee cash allocated).
        return EntryFill(fill_price=fill_price, fee=fee, shares=total_shares)


# ---------------------------------------------------------------------------
# Registry dict: (name, version) → singleton instance
# ---------------------------------------------------------------------------

_STATIC_BPS_V1 = _StaticBpsV1()
_NEXT_BAR_OPEN_V1 = _NextBarOpenV1()
_BOOK_REPLAY_V1 = _BookReplayV1()

_REGISTRY: dict[tuple[str, int], FillModel] = {
    ("static_bps", 1): _STATIC_BPS_V1,
    ("next_bar_open", 1): _NEXT_BAR_OPEN_V1,
    ("book_replay", 1): _BOOK_REPLAY_V1,
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
