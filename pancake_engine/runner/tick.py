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
from ..compile.condition import Condition, compile_condition
from ..crypto_ohlcv.compile import compile_crypto_ohlcv_spec
from ..crypto_ohlcv.indicators import compute_indicator
from ..crypto_ohlcv.types import CryptoOhlcvSpec
from ..types import EvidenceSpec
from .fill import BPS_DIVISOR, Fill, FillRejection, SimFillRouter

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
    "CryptoTickBar",
    "CryptoTickPosition",
    "CryptoTickRequest",
    "CryptoTickResponse",
    "tick_crypto",
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
    # 0.9 paper_guard state — optional with safe defaults so existing callers are untouched.
    peak_equity: float | None = None
    consecutive_losses: int = 0
    cooldown_remaining: int = 0


class TickResponse(BaseModel):
    """ADR-0035 §2.2 as amended — engine → dispatcher. No ``next_check_at``."""

    events: list[PaperEvent]
    new_cash: float
    new_positions: dict[str, TickPosition]
    new_equity: float
    verification_boundary: VerificationBoundary
    suggested_next_check: int | None = None
    # 0.9 paper_guard state threads through (optional; None when not active).
    peak_equity: float | None = None
    consecutive_losses: int = 0
    cooldown_remaining: int = 0


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
         Guard state (peak_equity, consecutive_losses) is updated here.
      2. paper_guard check → if tripped or cooldown_remaining > 0, skip all entries.
         If just tripped: emit guard_suspended event, set cooldown.
         If in existing cooldown: decrement, skip silently.
      3. Entries → evaluate the IR entry condition on each non-held, non-resolved
         bar; open via ``SimFillRouter`` at ``bar.close``.
      4. ``new_equity`` = cash + Σ mark(open positions).
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

    # Compile exit condition if present (paper-lane only).
    exit_condition: Condition | None = None
    exit_cfg = request.strategy_spec_ir.strategy.exit
    if isinstance(exit_cfg, dict):
        exit_when = exit_cfg.get("when")
        if isinstance(exit_when, dict):
            exit_condition = compile_condition(exit_when)

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

    # Guard state — thread from request; updated during settlements
    consecutive_losses = request.consecutive_losses
    cooldown_remaining = request.cooldown_remaining

    # Compute start-of-tick equity for guard evaluation (before settlements).
    # This ensures a winning settlement doesn't retroactively prevent a guard trip.
    start_mark_total = 0.0
    for pos in request.current_positions.values():
        bar_ = {b.instrument_id: b for b in request.market_snapshot}.get(pos.instrument_id)
        mps_, _ = _mark_per_share(pos, bar_)
        start_mark_total += pos.shares * mps_
    start_equity = cash + start_mark_total

    # Update equity peak from start-of-tick equity.
    peak_equity = request.peak_equity
    if peak_equity is None:
        peak_equity = start_equity
    else:
        peak_equity = max(peak_equity, start_equity)

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
            # Update consecutive_losses: a full-loss settlement (settle_value == 0) increments;
            # a winning settlement resets. Only full losses count (settle_value == 0 exactly).
            if request.strategy_spec_ir.strategy.paper_guard is not None:
                if settle_value == 0.0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
            continue

        # Exit condition: evaluate against current bar before falling through to mark.
        mark_per_share, stale = _mark_per_share(pos, bar)
        if (
            exit_condition is not None
            and bar is not None
            and exit_condition(_bar_to_row(bar, entry_price_col, pos.side))
        ):
            # Close the position at mark price, applying sell-side slippage + fee.
            # sell-side slippage: proceeds are reduced (price moves against us on exit).
            exit_price = mark_per_share * (1.0 - compiled.slippage_bps / BPS_DIVISOR)
            notional = pos.shares * mark_per_share
            fee = notional * (compiled.fee_bps / BPS_DIVISOR)
            proceeds = pos.shares * exit_price - fee
            cash += proceeds
            settled_ids.add(iid)
            events.append(PaperEvent(
                event_kind="position_closed",
                observed_at=t,
                tick_cursor=t,
                payload={
                    "instrument_id": iid,
                    "side": pos.side,
                    "reason": "exit",
                    "exit_price": exit_price,
                    "mark_price": mark_per_share,
                    "shares": pos.shares,
                    "proceeds": proceeds,
                    "fee": fee,
                    "pnl": proceeds - pos.cost,
                },
                source_manifest_id=bar.source_manifest_id,
            ))
            continue

        # Hold → mark to market (carry forward if the instrument is absent).
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

    # 2. paper_guard — decide whether to skip entries.
    # Guard uses start_equity / peak_equity (computed before settlements above).
    guard = request.strategy_spec_ir.strategy.paper_guard
    skip_entries = False
    just_tripped = False
    tripped_guard_name: str = ""
    tripped_observed: float = 0.0
    tripped_threshold: float = 0.0

    if cooldown_remaining > 0:
        # Already in cooldown — silently skip entries and decrement.
        skip_entries = True
        cooldown_remaining -= 1
    elif guard is not None:
        # Check each configured guard condition.
        if not skip_entries and "max_drawdown_pct" in guard:
            threshold = float(guard["max_drawdown_pct"])
            if peak_equity > 0:
                observed_dd = (peak_equity - start_equity) / peak_equity
            else:
                observed_dd = 0.0
            if observed_dd >= threshold:
                skip_entries = True
                just_tripped = True
                tripped_guard_name = "max_drawdown_pct"
                tripped_observed = observed_dd
                tripped_threshold = threshold

        if not skip_entries and "max_consecutive_losses" in guard:
            threshold_int = int(guard["max_consecutive_losses"])
            if consecutive_losses >= threshold_int:
                skip_entries = True
                just_tripped = True
                tripped_guard_name = "max_consecutive_losses"
                tripped_observed = float(consecutive_losses)
                tripped_threshold = float(threshold_int)

        if just_tripped:
            cooldown_bars = int(guard.get("cooldown_bars", 1))
            cooldown_remaining = cooldown_bars
            events.append(PaperEvent(
                event_kind="guard_suspended",
                observed_at=t,
                tick_cursor=t,
                payload={
                    "guard": tripped_guard_name,
                    "observed": tripped_observed,
                    "threshold": tripped_threshold,
                },
            ))

    # 3. Entries: non-held, non-resolved candidates whose entry condition fires.
    if not skip_entries:
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

    # 4. Equity = cash + Σ mark(open positions) at market.
    mark_total = 0.0
    for iid, pos in open_positions.items():
        mark_per_share, _ = _mark_per_share(pos, bars.get(iid))
        mark_total += pos.shares * mark_per_share
    new_equity = cash + mark_total

    # Update peak equity one final time with the post-entry equity.
    peak_equity = max(peak_equity, new_equity)

    return TickResponse(
        events=events,
        new_cash=cash,
        new_positions=open_positions,
        new_equity=new_equity,
        verification_boundary=VerificationBoundary(verification_grade=ENGINE_VERIFICATION_GRADE),
        suggested_next_check=None,
        peak_equity=peak_equity,
        consecutive_losses=consecutive_losses,
        cooldown_remaining=cooldown_remaining,
    )


# ---------------------------------------------------------------------------
# tick_crypto() — single-bar paper evaluation for the crypto-OHLCV family
# ---------------------------------------------------------------------------


class CryptoTickBar(BaseModel):
    """One OHLCV bar delivered to ``tick_crypto``.  ``t`` is the bar-OPEN epoch
    seconds (UTC), matching ``OhlcvBar.t``; all OHLCV fields are required.

    The bar carries the previous bar's context as well (``prev_open`` … ``prev_close``)
    so that cross-above / cross-below conditions can be evaluated without storing
    full history on the dispatcher side.  Set all ``prev_*`` fields to ``None``
    on the very first bar (no prior context → cross conditions are False, identical
    to the backtest warm-up convention)."""

    model_config = ConfigDict(extra="allow")

    t: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    # Previous bar values — required for cross-type conditions; None = first bar.
    prev_open: float | None = None
    prev_high: float | None = None
    prev_low: float | None = None
    prev_close: float | None = None
    prev_volume: float | None = None
    # Pre-computed indicator values at this bar (None = warm-up / not-yet-defined).
    # Keys must match ``CryptoOhlcvSpec.strategy.indicators[*].id``.
    indicator_values: dict[str, float | None] = {}
    prev_indicator_values: dict[str, float | None] = {}


class CryptoTickPosition(BaseModel):
    """Open crypto position threaded through dispatcher ↔ engine.  All monetary
    fields are in the spec's ``currency`` (default USD)."""

    instrument_id: str
    side: str            # "long" | "short"
    qty: float           # signed position size (+ long, − short)
    entry_fill: float    # price at which the position was opened (after slippage)
    entry_quote: float   # pre-slip open price at entry
    notional: float      # cash committed at entry
    entry_fee: float
    opened_at: int       # epoch seconds (bar-OPEN time of the entry bar)


class CryptoTickRequest(BaseModel):
    """Dispatcher → engine for a single crypto-OHLCV paper tick.

    ``tick_cursor`` is the bar-OPEN epoch seconds of ``bar`` (matches ``bar.t``).
    The dispatcher is responsible for building the ``CryptoTickBar`` with the
    correct ``prev_*`` and ``indicator_values`` / ``prev_indicator_values`` fields
    — the engine performs no look-back here (no look-ahead rule applies equally)."""

    deployment_id: str
    mode: str = "paper"
    strategy_spec: CryptoOhlcvSpec
    tick_cursor: int
    bar: CryptoTickBar
    current_cash: float
    current_position: CryptoTickPosition | None = None
    # paper_guard threads identically to PM tick (optional; safe defaults)
    peak_equity: float | None = None
    consecutive_losses: int = 0
    cooldown_remaining: int = 0


class CryptoTickResponse(BaseModel):
    """Engine → dispatcher for a single crypto-OHLCV paper tick.

    ``paper_fill_convention`` is ALWAYS ``"bar_close"`` for paper/live crypto.

    **Deliberate backtest ↔ paper divergence (documented here and in the field):**
    The crypto backtest (``run_crypto_ohlcv``) fills at NEXT BAR OPEN — this is the
    correct historical-simulation choice (you can't trade at a bar's open before the
    bar opens).  In the paper / live-feed path the concept of "next bar" does not
    exist at tick time: the live agent receives the JUST-CLOSED bar and must act NOW.
    Filling at the CLOSE of the just-delivered bar is the live-trading analogue —
    it represents the market price the moment the signal fired.  Callers MUST NOT
    compare paper PnL directly to backtest PnL without accounting for this
    convention difference; ``paper_fill_convention="bar_close"`` is surfaced in the
    response so receipts can never silently conflate the two."""

    events: list[PaperEvent]
    new_cash: float
    new_position: CryptoTickPosition | None
    new_equity: float
    paper_fill_convention: str = "bar_close"
    verification_boundary: VerificationBoundary
    # paper_guard state threaded through (None when not active)
    peak_equity: float | None = None
    consecutive_losses: int = 0
    cooldown_remaining: int = 0


def tick_crypto(request: CryptoTickRequest) -> CryptoTickResponse:
    """Run one crypto-OHLCV paper tick. Pure function of ``request``; deterministic.

    **Fill convention (paper_fill_convention = "bar_close"):**
    Entry and exit fills execute at the delivered bar's CLOSE price (after
    slippage).  This diverges from the backtest's next-bar-open fill
    (``fill_timing="next_bar_open"``).  See ``CryptoTickResponse`` for the full
    rationale; the divergence is intentional and documented in the response field
    ``paper_fill_convention`` so every receipt carries the convention used.

    Order within a tick:
      1. If ``current_position`` is open → evaluate exit condition; close at
         ``bar.close`` (with sell-side slippage + fee) if it fires.
      2. paper_guard check (identical logic to PM ``tick()``): if tripped or
         ``cooldown_remaining > 0``, skip entry.
      3. If flat after step 1 → evaluate entry condition; open at ``bar.close``
         (with buy-side slippage + fee) if it fires.
      4. ``new_equity`` = cash + mark-at-close of open position (if any).

    Validation:
      - ``request.mode`` must be ``"paper"``.
      - ``request.strategy_spec.spec_family`` must be ``"crypto-ohlcv-spec"``.
      - ``request.bar.t`` must equal ``request.tick_cursor`` (no look-ahead).
    """
    if request.mode != "paper":
        raise TickError(
            "UNSUPPORTED_MODE",
            f"mode {request.mode!r} is not supported",
            retryable=False,
        )
    if request.strategy_spec.spec_family != "crypto-ohlcv-spec":
        raise TickError(
            "INVALID_SPEC_FAMILY",
            f"tick_crypto requires spec_family='crypto-ohlcv-spec', "
            f"got {request.strategy_spec.spec_family!r}",
            retryable=False,
        )
    if request.bar.t != request.tick_cursor:
        raise TickError(
            "LOOKAHEAD",
            f"bar.t {request.bar.t} != tick_cursor {request.tick_cursor}",
            retryable=False,
        )

    t = request.tick_cursor
    spec = request.strategy_spec
    compiled = compile_crypto_ohlcv_spec(spec)

    slip = compiled.slippage_bps / BPS_DIVISOR
    fee_rate = compiled.fee_bps / BPS_DIVISOR

    bar = request.bar
    # Build evaluation contexts from the dispatcher-supplied indicator values.
    cur: dict[str, float] = {
        "open": bar.open, "high": bar.high,
        "low": bar.low, "close": bar.close,
    }
    for k, v in bar.indicator_values.items():
        if v is not None:
            cur[k] = v

    prev: dict[str, float] | None = None
    if bar.prev_close is not None:
        prev = {
            "open": bar.prev_open or bar.open,
            "high": bar.prev_high or bar.high,
            "low": bar.prev_low or bar.low,
            "close": bar.prev_close,
        }
        for k, v in bar.prev_indicator_values.items():
            if v is not None:
                prev[k] = v

    cash = float(request.current_cash)
    events: list[PaperEvent] = []
    current_position = request.current_position

    # Guard state — thread from request
    consecutive_losses = request.consecutive_losses
    cooldown_remaining = request.cooldown_remaining

    # Start-of-tick equity for guard evaluation (before any close).
    start_equity = _crypto_equity(cash, current_position, bar.close)
    peak_equity = request.peak_equity
    if peak_equity is None:
        peak_equity = start_equity
    else:
        peak_equity = max(peak_equity, start_equity)

    # 1. Exit evaluation — if position open and exit condition fires, close now.
    if current_position is not None:
        if compiled.exit(prev, cur):
            # Sell-side exit: proceeds are reduced by slippage.
            qty = abs(current_position.qty)
            direction = 1.0 if current_position.side == "long" else -1.0
            if current_position.side == "long":
                exit_fill = bar.close * (1.0 - slip)
                gross = qty * exit_fill
                exit_fee = gross * fee_rate
                proceeds = gross - exit_fee
                cash += proceeds
            else:  # short → buy back
                exit_fill = bar.close * (1.0 + slip)
                gross = qty * exit_fill
                exit_fee = gross * fee_rate
                cash -= gross + exit_fee
                proceeds = -gross - exit_fee  # cash delta from closing short
            pnl = (
                direction * qty * (exit_fill - current_position.entry_fill)
                - current_position.entry_fee
                - exit_fee
            )
            events.append(PaperEvent(
                event_kind="position_closed",
                observed_at=t,
                tick_cursor=t,
                payload={
                    "instrument_id": current_position.instrument_id,
                    "side": current_position.side,
                    "reason": "exit",
                    "exit_fill": exit_fill,
                    "exit_price_quote": bar.close,
                    "shares": qty,
                    "exit_fee": exit_fee,
                    "pnl": pnl,
                    "paper_fill_convention": "bar_close",
                },
            ))
            # Guard: full-loss on position closed via exit → track consecutive losses
            # (a loss = negative pnl)
            if spec.strategy.paper_guard is not None:
                if pnl < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
            current_position = None

    # 2. paper_guard — decide whether to skip new entry.
    guard = spec.strategy.paper_guard
    skip_entry = False
    just_tripped = False
    tripped_guard_name = ""
    tripped_observed = 0.0
    tripped_threshold = 0.0

    if cooldown_remaining > 0:
        skip_entry = True
        cooldown_remaining -= 1
    elif guard is not None:
        if not skip_entry and "max_drawdown_pct" in guard:
            threshold = float(guard["max_drawdown_pct"])
            observed_dd = (peak_equity - start_equity) / peak_equity if peak_equity > 0 else 0.0
            if observed_dd >= threshold:
                skip_entry = True
                just_tripped = True
                tripped_guard_name = "max_drawdown_pct"
                tripped_observed = observed_dd
                tripped_threshold = threshold
        if not skip_entry and "max_consecutive_losses" in guard:
            threshold_int = int(guard["max_consecutive_losses"])
            if consecutive_losses >= threshold_int:
                skip_entry = True
                just_tripped = True
                tripped_guard_name = "max_consecutive_losses"
                tripped_observed = float(consecutive_losses)
                tripped_threshold = float(threshold_int)

        if just_tripped:
            cooldown_bars = int(guard.get("cooldown_bars", 1))
            cooldown_remaining = cooldown_bars
            events.append(PaperEvent(
                event_kind="guard_suspended",
                observed_at=t,
                tick_cursor=t,
                payload={
                    "guard": tripped_guard_name,
                    "observed": tripped_observed,
                    "threshold": tripped_threshold,
                },
            ))

    # 3. Entry evaluation — only when flat and not guard-suspended.
    if not skip_entry and current_position is None:
        if compiled.entry(prev, cur):
            notional = cash * compiled.sizing_value
            if notional <= 0:
                events.append(PaperEvent(
                    event_kind="order_rejected",
                    observed_at=t,
                    tick_cursor=t,
                    payload={
                        "instrument_id": spec.instrument_id,
                        "side": compiled.side,
                        "reason": "sizing_zero",
                        "available_cash": cash,
                    },
                ))
            else:
                side_str = compiled.side
                if side_str == "long":
                    fill_price = bar.close * (1.0 + slip)
                    entry_fee = notional * fee_rate
                    qty = (notional - entry_fee) / fill_price
                    cash -= notional
                    new_qty = qty
                else:  # short
                    fill_price = bar.close * (1.0 - slip)
                    if fill_price <= 0:
                        fill_price = 0.0
                        events.append(PaperEvent(
                            event_kind="order_rejected",
                            observed_at=t,
                            tick_cursor=t,
                            payload={
                                "instrument_id": spec.instrument_id,
                                "side": side_str,
                                "reason": "non_positive_short_fill",
                                "bar_close": bar.close,
                            },
                        ))
                        fill_price = -1.0  # sentinel so we skip below
                    if fill_price > 0:
                        entry_fee = notional * fee_rate
                        qty = -(notional / fill_price)
                        cash += notional - entry_fee
                        new_qty = qty
                    else:
                        notional = 0.0
                        entry_fee = 0.0
                        new_qty = 0.0

                if notional > 0:
                    events.append(PaperEvent(
                        event_kind="order_placed",
                        observed_at=t,
                        tick_cursor=t,
                        payload={
                            "instrument_id": spec.instrument_id,
                            "side": side_str,
                            "notional": notional,
                            "paper_fill_convention": "bar_close",
                        },
                    ))
                    events.append(PaperEvent(
                        event_kind="order_filled",
                        observed_at=t,
                        tick_cursor=t,
                        payload={
                            "instrument_id": spec.instrument_id,
                            "side": side_str,
                            "fill_price": fill_price,
                            "quote_price": bar.close,
                            "qty": new_qty,
                            "notional": notional,
                            "entry_fee": entry_fee,
                            "paper_fill_convention": "bar_close",
                        },
                    ))
                    events.append(PaperEvent(
                        event_kind="position_opened",
                        observed_at=t,
                        tick_cursor=t,
                        payload={
                            "instrument_id": spec.instrument_id,
                            "side": side_str,
                            "qty": new_qty,
                            "entry_fill": fill_price,
                            "entry_quote": bar.close,
                            "notional": notional,
                            "entry_fee": entry_fee,
                            "paper_fill_convention": "bar_close",
                        },
                    ))
                    current_position = CryptoTickPosition(
                        instrument_id=spec.instrument_id,
                        side=side_str,
                        qty=new_qty,
                        entry_fill=fill_price,
                        entry_quote=bar.close,
                        notional=notional,
                        entry_fee=entry_fee,
                        opened_at=t,
                    )

    # 4. new_equity = cash + mark-at-close of open position (if any).
    new_equity = _crypto_equity(cash, current_position, bar.close)
    peak_equity = max(peak_equity, new_equity)

    return CryptoTickResponse(
        events=events,
        new_cash=cash,
        new_position=current_position,
        new_equity=new_equity,
        paper_fill_convention="bar_close",
        verification_boundary=VerificationBoundary(
            verification_grade=ENGINE_VERIFICATION_GRADE
        ),
        peak_equity=peak_equity,
        consecutive_losses=consecutive_losses,
        cooldown_remaining=cooldown_remaining,
    )


def _crypto_equity(cash: float, pos: CryptoTickPosition | None, bar_close: float) -> float:
    """Mark an open crypto position at bar.close. long: cash + qty*close;
    short: cash + qty*close (qty is negative, so this subtracts the mark-to-market
    loss/gain correctly)."""
    if pos is None:
        return cash
    return cash + pos.qty * bar_close


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
