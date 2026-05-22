"""Event-time ledger for capital accounting (architecture §Capital accounting).

Identities maintained:

```
cash(t)         = starting_capital − open_cost(t) + realized_pnl(t)
mark_value(t)   = Σ shares × entry_fill_price   (mark_at_cost)
equity(t)       = cash(t) + mark_value(t)
```

Under ``mark_at_cost``, ``mark_value`` per position equals
``notional − fee``; equity therefore drops by ``fee`` at the decision-time
event (fee realized at entry; B-1 of math bash).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .position import Position
from .trade import Trade

__all__ = ["Ledger"]


@dataclass
class Ledger:
    starting_capital: float
    cash: float = field(init=False)
    realized_pnl: float = 0.0
    open_positions: dict[int, Position] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = float(self.starting_capital)

    def open(self, position: Position) -> None:
        """Open a position. Cash drops by ``position.cost`` (notional)."""
        self.cash -= position.cost
        self.open_positions[position.id] = position

    def close(self, position_id: int, settle_value: float, days_held: int) -> Trade:
        """Close an open position. Cash rises by ``proceeds = shares × settle_value``."""
        position = self.open_positions.pop(position_id)
        proceeds = position.shares * settle_value
        pnl = proceeds - position.cost
        self.cash += proceeds
        self.realized_pnl += pnl
        return_pct = (pnl / position.cost) if position.cost > 0 else 0.0

        trade = Trade(
            market_slug=position.market_link,
            outcome=position.side,
            entry_t=position.decision_time,
            entry_price=position.entry_price,
            entry_price_quote=position.entry_price_quote,
            exit_t=position.resolution_time,
            exit_price=float(settle_value),
            exit_price_quote=float(settle_value),
            exit_reason="hold_to_resolution",
            shares=position.shares,
            cost=position.cost,
            proceeds=proceeds,
            pnl=pnl,
            return_pct=return_pct,
            days_held=days_held,
            resolved_outcome=None,
        )
        self.trades.append(trade)
        return trade

    # --- queries ---

    @property
    def open_cost(self) -> float:
        return sum(p.cost for p in self.open_positions.values())

    def mark_value(self) -> float:
        """``mark_at_cost``: mark per share = entry_fill_price → mark_value = Σ shares × entry_price."""
        return sum(p.shares * p.entry_price for p in self.open_positions.values())

    def equity(self) -> float:
        return self.cash + self.mark_value()
