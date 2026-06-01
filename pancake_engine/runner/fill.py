"""SimFillRouter — the rule-145 bar-close fill, Python-side (ADR-0031 / ADR-0035).

The per-tick analogue of the backtest's ``_process_decision`` fill math. Given a
bar close and the account state, it sizes the order (``fixed_fraction`` ×
available_cash), applies multiplicative slippage and a notional fee, and returns
the fill. The only difference from the backtest fill is the *price source*: the
quote comes from the snapshot ``bar.close`` (rule 145), side-adjusted for NO
(``1 - yes_close``), instead of a dataset ``entry_price`` column.

ADR-0031 keeps this math Python-side (no dual-math drift). The same ``(0, 1)``
price guards and ``fixed_fraction`` sizing as the backtest apply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    """

    def __init__(self, *, slippage_bps: float, fee_bps: float) -> None:
        self.slippage_bps = slippage_bps
        self.fee_bps = fee_bps

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

        fill_price = quote * (1.0 + self.slippage_bps / BPS_DIVISOR)
        if not (0.0 < fill_price < 1.0):
            return FillRejection(
                "fill_price_out_of_range",
                {"quote": quote, "fill_price": fill_price, "slippage_bps": self.slippage_bps},
            )

        fee = sizing.notional * (self.fee_bps / BPS_DIVISOR)
        investable = sizing.notional - fee
        shares = investable / fill_price
        return Fill(
            fill_price=fill_price,
            quote_price=quote,
            shares=shares,
            cost=sizing.notional,
            fee=fee,
        )
