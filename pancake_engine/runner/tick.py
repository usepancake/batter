"""Single-bar paper ``tick`` — the engine half of ADR-0035 (the locked `/tick`
contract). Amended by ``0035-amendment-engine-confirmation.md`` (A+A):

- The engine is a pure deterministic fill + valuation function: no clock, no
  scheduler. It does NOT return ``next_check_at`` (dispatcher owns scheduling).
- ``new_equity`` is engine-authoritative, marked **at market** (``bar.close``;
  NO side = ``1 - yes_close``). The dispatcher persists it, never recomputes.
- Settlement is a bar-carried resolution marker (``MarketBar.resolution``);
  the venue-declared ``resolved_outcome`` is authoritative (NOT the spec's
  ``yes_payoff``). Reuses the backtest close math.
- No look-ahead (rule 139): every input is as-of ``tick_cursor``.

A ``/tick`` is a single-bar step with no resolution by default: positions open,
then are held + marked to market until the venue resolves them. This is distinct
from ``run_backtest`` (a full DECISION→RESOLUTION walk); it reuses the same
primitives (``compute_sizing``, the ``_process_decision`` fill math via
``SimFillRouter``) but is its own one-bar path.

Times are unix **seconds** (int) throughout, matching the runner. The HTTP shim
converts ISO-8601 ↔ epoch at the boundary (as ``/run`` already does for ms).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from ..__version__ import ENGINE_VERIFICATION_GRADE
from ..compile import compile_spec
from ..types import EvidenceSpec
from .fill import Fill, FillRejection, SimFillRouter

__all__ = [
    "ResolutionMarker",
    "MarketBar",
    "TickPosition",
    "PaperEvent",
    "VerificationBoundary",
    "TickRequest",
    "TickResponse",
    "TickError",
    "tick",
]


# ---------------------------------------------------------------------------
# Wire types (ADR-0035 §2.1 / §2.2 as amended)
# ---------------------------------------------------------------------------


class ResolutionMarker(BaseModel):
    """Bar-carried settlement fact (amendment §6). YES-perspective outcome."""

    resolved_at: int
    resolved_outcome: int  # 0 or 1

    @field_validator("resolved_outcome")
    @classmethod
    def _binary(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("resolved_outcome must be 0 or 1 (YES-perspective)")
        return v


class MarketBar(BaseModel):
    """A snapshot bar as-of ``tick_cursor``. ``close`` is the YES contract price
    (the rule-145 fill source). Extra feature columns the data plane attaches are
    preserved (``extra='allow'``) so the compiled IR conditions can read them."""

    model_config = ConfigDict(extra="allow")

    instrument_id: str
    observed_at: int
    close: float
    open: float | None = None
    high: float | None = None
    low: float | None = None
    volume: float | None = None
    source_manifest_id: str | None = None
    resolution: ResolutionMarker | None = None


class TickPosition(BaseModel):
    """Row-independent open position (amendment §5). Round-tripped by the
    dispatcher verbatim. ``last_mark`` is engine-managed (carry-forward mark per
    share for the stale-mark path, §3); the dispatcher treats it as opaque."""

    instrument_id: str
    side: str  # "YES" | "NO"
    shares: float
    entry_price: float
    cost: float
    fee: float
    opened_at: int
    last_mark: float | None = None


class PaperEvent(BaseModel):
    """A ledger event the dispatcher persists append-only into ``paper_events``.
    ``payload`` is opaque to the dispatcher. ``observed_at`` / ``tick_cursor`` are
    epoch seconds here; the shim renders ISO-8601 at the wire."""

    event_kind: str
    observed_at: int
    tick_cursor: int | None = None
    payload: dict[str, Any]
    source_manifest_id: str | None = None


class VerificationBoundary(BaseModel):
    """Rule 159 — the engine self-identifies which engine minted the tick."""

    verification_grade: str


class TickRequest(BaseModel):
    """ADR-0035 §2.1 — dispatcher → engine (no ``tick_cadence``; engine has no
    scheduler)."""

    deployment_id: str
    mode: str = "paper"
    strategy_spec_ir: EvidenceSpec
    tick_cursor: int
    market_snapshot: list[MarketBar] = []
    universe_state: Any = None
    current_cash: float
    current_positions: dict[str, TickPosition] = {}


class TickResponse(BaseModel):
    """ADR-0035 §2.2 as amended — engine → dispatcher. No ``next_check_at``."""

    events: list[PaperEvent]
    new_cash: float
    new_positions: dict[str, TickPosition]
    new_equity: float
    verification_boundary: VerificationBoundary
    suggested_next_check: int | None = None


class TickError(Exception):
    """ADR-0031 structured-error envelope ``{code, message, retryable}`` — never a
    bare string. The shim serializes ``.envelope`` to the HTTP error body."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable

    @property
    def envelope(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}


# ---------------------------------------------------------------------------
# tick()
# ---------------------------------------------------------------------------


def tick(request: TickRequest) -> TickResponse:
    """Run one paper tick. Pure function of ``request``; deterministic.

    Order within a tick (deterministic, sorted by ``instrument_id``):
      1. Held positions → settle (if resolved as-of t) or mark-to-market.
      2. Entries → evaluate the IR entry condition on each non-held, non-resolved
         bar; open via ``SimFillRouter`` at ``bar.close``.
      3. ``new_equity`` = cash + Σ mark(open positions).
    """
    if request.mode != "paper":
        raise TickError(
            "UNSUPPORTED_MODE", f"mode {request.mode!r} is not supported", retryable=False
        )

    t = request.tick_cursor
    try:
        compiled = compile_spec(request.strategy_spec_ir)
    except Exception as exc:  # spec/IR invalid → structured error, not a bare raise
        raise TickError(
            "INVALID_SPEC_IR", f"strategy_spec_ir failed to compile: {exc}", retryable=False
        ) from exc

    router = SimFillRouter(slippage_bps=compiled.slippage_bps, fee_bps=compiled.fee_bps)
    bars: dict[str, MarketBar] = {b.instrument_id: b for b in request.market_snapshot}

    # No look-ahead (rule 139): the engine refuses any input dated after t.
    for b in request.market_snapshot:
        if b.observed_at > t:
            raise TickError(
                "LOOKAHEAD",
                f"bar {b.instrument_id!r} observed_at {b.observed_at} > tick_cursor {t}",
                retryable=False,
            )
        if b.resolution is not None and b.resolution.resolved_at > t:
            ra = b.resolution.resolved_at
            raise TickError(
                "LOOKAHEAD",
                f"resolution {b.instrument_id!r} resolved_at {ra} > cursor {t}",
                retryable=False,
            )

    cash = float(request.current_cash)
    events: list[PaperEvent] = []
    open_positions: dict[str, TickPosition] = {}
    settled_ids: set[str] = set()

    entry_price_col = _entry_price_col(request.strategy_spec_ir)

    # 1. Settle or mark held positions.
    for iid in sorted(request.current_positions):
        pos = request.current_positions[iid]
        bar = bars.get(iid)

        if bar is not None and bar.resolution is not None:
            outcome = bar.resolution.resolved_outcome
            strategy_wins = (outcome == 1) if pos.side == "YES" else (outcome == 0)
            settle_value = 1.0 if strategy_wins else 0.0
            proceeds = pos.shares * settle_value
            pnl = proceeds - pos.cost
            cash += proceeds
            settled_ids.add(iid)
            events.append(PaperEvent(
                event_kind="position_closed",
                observed_at=t,
                tick_cursor=t,
                payload={
                    "instrument_id": iid,
                    "side": pos.side,
                    "settle_value": settle_value,
                    "shares": pos.shares,
                    "proceeds": proceeds,
                    "pnl": pnl,
                    "return_pct": (pnl / pos.cost) if pos.cost > 0 else 0.0,
                    "resolved_at": bar.resolution.resolved_at,
                    "resolved_outcome": outcome,
                },
                source_manifest_id=bar.source_manifest_id,
            ))
            continue

        # Hold → mark to market (carry forward if the instrument is absent).
        mark_per_share, stale = _mark_per_share(pos, bar)
        events.append(PaperEvent(
            event_kind="mark_to_market",
            observed_at=t,
            tick_cursor=t,
            payload={
                "instrument_id": iid,
                "mark_price": mark_per_share,
                "mark_value": pos.shares * mark_per_share,
                "stale_mark": stale,
            },
            source_manifest_id=(bar.source_manifest_id if bar is not None else None),
        ))
        open_positions[iid] = pos.model_copy(update={"last_mark": mark_per_share})

    # 2. Entries: non-held, non-resolved candidates whose entry condition fires.
    for iid in sorted(bars):
        bar = bars[iid]
        if iid in open_positions or iid in settled_ids:
            continue
        if bar.resolution is not None:
            continue  # resolved instrument is terminal — never enterable
        if not compiled.entry_condition(_bar_to_row(bar, entry_price_col, compiled.side)):
            continue

        result = router.fill(
            side=compiled.side,
            yes_close=bar.close,
            available_cash=cash,
            sizing_value=compiled.sizing_value,
        )
        if isinstance(result, FillRejection):
            events.append(PaperEvent(
                event_kind="order_rejected",
                observed_at=t,
                tick_cursor=t,
                payload={
                    "instrument_id": iid,
                    "side": compiled.side,
                    "reason": result.reason,
                    **result.detail,
                },
                source_manifest_id=bar.source_manifest_id,
            ))
            continue

        fill: Fill = result
        cash -= fill.cost
        events.append(PaperEvent(
            event_kind="order_placed",
            observed_at=t,
            tick_cursor=t,
            payload={"instrument_id": iid, "side": compiled.side, "notional": fill.cost},
            source_manifest_id=bar.source_manifest_id,
        ))
        events.append(PaperEvent(
            event_kind="order_filled",
            observed_at=t,
            tick_cursor=t,
            payload={
                "instrument_id": iid,
                "side": compiled.side,
                "fill_price": fill.fill_price,
                "quote_price": fill.quote_price,
                "shares": fill.shares,
                "fee": fill.fee,
                "cost": fill.cost,
            },
            source_manifest_id=bar.source_manifest_id,
        ))
        open_positions[iid] = TickPosition(
            instrument_id=iid,
            side=compiled.side,
            shares=fill.shares,
            entry_price=fill.fill_price,
            cost=fill.cost,
            fee=fill.fee,
            opened_at=t,
            last_mark=fill.quote_price,
        )
        events.append(PaperEvent(
            event_kind="position_opened",
            observed_at=t,
            tick_cursor=t,
            payload={
                "instrument_id": iid,
                "side": compiled.side,
                "shares": fill.shares,
                "entry_price": fill.fill_price,
                "cost": fill.cost,
                "fee": fill.fee,
            },
            source_manifest_id=bar.source_manifest_id,
        ))

    # 3. Equity = cash + Σ mark(open positions) at market.
    mark_total = 0.0
    for iid, pos in open_positions.items():
        mark_per_share, _ = _mark_per_share(pos, bars.get(iid))
        mark_total += pos.shares * mark_per_share
    new_equity = cash + mark_total

    return TickResponse(
        events=events,
        new_cash=cash,
        new_positions=open_positions,
        new_equity=new_equity,
        verification_boundary=VerificationBoundary(verification_grade=ENGINE_VERIFICATION_GRADE),
        suggested_next_check=None,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mark_per_share(pos: TickPosition, bar: MarketBar | None) -> tuple[float, bool]:
    """Mark-at-market per share. NO side = ``1 - yes_close``. Returns
    ``(mark_per_share, stale)``. When the instrument is absent from the snapshot,
    carry the last known mark forward (stale); if never marked, fall back to the
    entry cost basis."""
    if bar is not None:
        mark = bar.close if pos.side == "YES" else 1.0 - bar.close
        return mark, False
    if pos.last_mark is not None:
        return pos.last_mark, True
    return pos.entry_price, True


def _bar_to_row(bar: MarketBar, entry_price_col: str | None, side: str = "YES") -> dict[str, Any]:
    """Project a bar onto the row the compiled IR conditions read. Feature
    columns (open/high/low/close/volume + data-plane extras) pass through; the
    entry_price semantic-role column is mapped to the side-appropriate contract
    price — ``bar.close`` (the YES price) for YES, ``1 - bar.close`` (the NO
    price) for NO — so the entry condition evaluates against the same price domain
    run_backtest uses. run_backtest reads the entry_price column straight from the
    dataset, where the convention is the literal side price (see
    test_case_03_no_at_096_wins); without the NO inversion here, paper/live entry
    gates diverge from the backtest for every NO-side spec."""
    row: dict[str, Any] = {
        k: v
        for k, v in bar.model_dump(exclude_none=True).items()
        if k not in ("instrument_id", "source_manifest_id", "resolution")
    }
    if entry_price_col is not None and entry_price_col not in row:
        row[entry_price_col] = bar.close if side == "YES" else 1.0 - bar.close
    return row


def _entry_price_col(spec: EvidenceSpec) -> str | None:
    """The schema column name carrying the ``entry_price`` semantic role, if any."""
    for req in spec.schema_requirements.required_columns:
        if req.semantic_role == "entry_price":
            return req.name
    return None
