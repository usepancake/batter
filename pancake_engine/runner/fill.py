"""SimFillRouter — the rule-145 bar-close fill, Python-side (ADR-0031 / ADR-0035).

The per-tick analogue of the backtest's ``_process_decision`` fill math. Given a
bar close and the account state, it sizes the order (``fixed_fraction`` ×
available_cash), applies multiplicative slippage and a notional fee, and returns
the fill. The only difference from the backtest fill is the *price source*: the
quote comes from the snapshot ``bar.close`` (rule 145), side-adjusted for NO
(``1 - yes_close``), instead of a dataset ``entry_price`` column.

ADR-0031 keeps this math Python-side (no dual-math drift). The same ``(0, 1)``
price guards and ``fixed_fraction`` sizing as the backtest apply.

0.9.x Wave A — SimFillRouter unification:
``SimFillRouter`` now resolves the fill model from the registry (default:
``static_bps@1``) so the paper and backtest lanes share one implementation.
The public ``fill()`` call-surface and its return types are byte-identical to
pre-unification; existing ``tick()`` tests prove it.  The router is constructed
with an optional ``fill_model_name`` / ``fill_model_version`` pair; omitting
both selects ``static_bps@1``, matching the pre-0.9.x default.

Note: ``book_replay@1`` is NOT supported through the SimFillRouter (the paper
lane does not carry L2 book snapshots at tick time — that is a 0.10.0 concern).
Constructing a router with ``fill_model_name="book_replay"`` raises
``ValueError`` so the caller fails loudly rather than silently degrading to
the wrong model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..fills.registry import FillBlocked
from ..fills.registry import default_model as _default_fill_model
from ..fills.registry import resolve as _resolve_fill_model
from .sizing import compute_sizing

__all__ = ["Fill", "FillRejection", "SimFillRouter", "BPS_DIVISOR"]

BPS_DIVISOR = 10_000


@dataclass(frozen=True)
class Fill:
    """A simulated fill for one open."""

    fill_price: float
    """Post-slip price paid per share of the side traded."""

    quote_price: float
    """Pre-slip side-adjusted quote (= ``bar.close`` for YES, ``1 - close`` for NO)."""

    shares: float
    cost: float
    """Cash deducted at open = notional (includes fee)."""

    fee: float


@dataclass(frozen=True)
class FillRejection:
    """No fill produced. ``reason`` is a stable code; ``detail`` carries context."""

    reason: str
    detail: dict[str, Any] = field(default_factory=dict)


class SimFillRouter:
    """Deterministic single-bar fill router (rule 145).

    Stateless across calls except for its cost parameters; a fresh router is
    cheap. Sequential calls within a tick each size off the *current* available
    cash, matching the backtest's sequential-decision behavior.

    0.9.x Wave A unification: the fill math is now delegated to the registry
    fill model (default ``static_bps@1``) so paper and backtest share one
    implementation.  The ``quote`` passed to the model is the bar-close
    side-adjusted price; the model applies slippage from there.

    ``book_replay@1`` is not routable through SimFillRouter (no L2 data at
    tick time); constructing with that model name raises ``ValueError``.
    """

    def __init__(
        self,
        *,
        slippage_bps: float,
        fee_bps: float,
        fill_model_name: str | None = None,
        fill_model_version: int | None = None,
    ) -> None:
        self.slippage_bps = slippage_bps
        self.fee_bps = fee_bps
        # Resolve the fill model once at construction time.
        if fill_model_name is not None and fill_model_version is not None:
            if fill_model_name == "book_replay":
                raise ValueError(
                    "SimFillRouter does not support book_replay@1: "
                    "the paper lane does not carry L2 book snapshots at tick time."
                )
            self._fill_model = _resolve_fill_model(fill_model_name, fill_model_version)
        else:
            self._fill_model = _default_fill_model()  # static_bps@1

    def fill(
        self,
        *,
        side: str,
        yes_close: float,
        available_cash: float,
        sizing_value: float,
    ) -> Fill | FillRejection:
        """Simulate a fill at ``bar.close`` for ``side``.

        Returns a :class:`Fill` on success or a :class:`FillRejection` when the
        quote/fill price is outside ``(0, 1)`` or sizing rounds to zero.
        """
        # Side-adjusted quote: a NO share costs 1 - yes_close.
        quote = yes_close if side == "YES" else 1.0 - yes_close
        if not (0.0 < quote < 1.0):
            return FillRejection(
                "quote_out_of_range",
                {"side": side, "yes_close": yes_close, "quote": quote},
            )

        sizing = compute_sizing(available_cash, sizing_value)
        if sizing.notional <= 0:
            return FillRejection(
                "sizing_zero",
                {"available_cash": available_cash, "sizing_value": sizing_value},
            )

        # Delegate to registry model (static_bps@1 by default).
        # The model returns an EntryFill, or FillBlocked for depth-aware
        # models. The paper router only resolves static_bps/next_bar_open
        # today (no book feed in paper), so a blocked fill here is a
        # deterministic rejection, surfaced — never silently treated as filled.
        entry_fill = self._fill_model.apply_entry(
            quote=quote,
            notional=sizing.notional,
            slippage_bps=self.slippage_bps,
            fee_bps=self.fee_bps,
        )
        if isinstance(entry_fill, FillBlocked):
            return FillRejection(
                "fill_blocked",
                {"quote": quote, "reason": entry_fill.reason, "context": entry_fill.context},
            )

        if not (0.0 < entry_fill.fill_price < 1.0):
            return FillRejection(
                "fill_price_out_of_range",
                {
                    "quote": quote,
                    "fill_price": entry_fill.fill_price,
                    "slippage_bps": self.slippage_bps,
                },
            )

        return Fill(
            fill_price=entry_fill.fill_price,
            quote_price=quote,
            shares=entry_fill.shares,
            cost=sizing.notional,
            fee=entry_fill.fee,
        )
