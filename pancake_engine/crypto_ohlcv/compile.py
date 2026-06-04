"""Compile a CryptoOhlcvSpec to a runner-ready form (ADR-0043 §2.3, P2).

Mirrors `pancake_engine.compile.spec`: a frozen `CompiledCryptoOhlcvSpec` with a
`compiled_spec_hash` (SHA-256 of the canonicalized raw spec) + compiled
entry/exit conditions.

A compiled condition is `(prev_ctx | None, cur_ctx) -> bool`:
- threshold ops (gt/lt/gte/lte) read only `cur_ctx`;
- cross ops (cross_above/cross_below) read prev + cur, so the FIRST bar (no
  prev) never crosses. A missing operand value → the condition is False (no
  guessing). This is the no-look-ahead contract the runner (P3) relies on.

`Ctx` is the per-bar evaluation context the runner builds: price fields
(open/high/low/close) + indicator-id → value.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..hash import sha256_canonical
from .types import Condition, CryptoOhlcvSpec, Indicator, Operand

__all__ = [
    "Ctx",
    "CompiledCondition",
    "CompiledCryptoOhlcvSpec",
    "compile_operand",
    "compile_condition",
    "compile_crypto_ohlcv_spec",
]

Ctx = dict[str, float]
CompiledCondition = Callable[["Ctx | None", Ctx], bool]


def compile_operand(op: Operand) -> Callable[[Ctx], float | None]:
    """Compile an operand to `ctx -> value | None` (None when the referenced
    field/indicator isn't present in the context yet)."""
    if op.ref == "const":
        assert op.value is not None  # guaranteed by Operand validator
        const = op.value
        return lambda _ctx: const
    if op.ref == "price":
        assert op.field is not None  # guaranteed by Operand validator
        field = op.field
        return lambda ctx: ctx.get(field)
    assert op.indicator_id is not None  # guaranteed by Operand validator
    iid = op.indicator_id
    return lambda ctx: ctx.get(iid)


def compile_condition(cond: Condition) -> CompiledCondition:
    left = compile_operand(cond.left)
    right = compile_operand(cond.right)
    op = cond.op

    if op in ("gt", "lt", "gte", "lte"):

        def threshold(_prev: Ctx | None, cur: Ctx) -> bool:
            lv, rv = left(cur), right(cur)
            if lv is None or rv is None:
                return False
            if op == "gt":
                return lv > rv
            if op == "lt":
                return lv < rv
            if op == "gte":
                return lv >= rv
            return lv <= rv  # lte

        return threshold

    def cross(prev: Ctx | None, cur: Ctx) -> bool:
        if prev is None:
            return False
        lp, rp, lc, rc = left(prev), right(prev), left(cur), right(cur)
        if lp is None or rp is None or lc is None or rc is None:
            return False
        if op == "cross_above":
            return lp <= rp and lc > rc
        return lp >= rp and lc < rc  # cross_below

    return cross


@dataclass(frozen=True)
class CompiledCryptoOhlcvSpec:
    raw: CryptoOhlcvSpec
    compiled_spec_hash: str
    entry: CompiledCondition
    exit: CompiledCondition
    indicators: tuple[Indicator, ...]

    @property
    def side(self) -> str:
        return self.raw.strategy.side

    @property
    def sizing_value(self) -> float:
        return self.raw.strategy.sizing.value

    @property
    def slippage_bps(self) -> float:
        return self.raw.costs.slippage_bps

    @property
    def fee_bps(self) -> float:
        return self.raw.costs.fee_bps

    @property
    def starting_capital(self) -> float:
        return self.raw.starting_capital

    @property
    def fill_timing(self) -> str:
        return self.raw.fill_timing

    @property
    def instrument_id(self) -> str:
        return self.raw.instrument_id


def compile_crypto_ohlcv_spec(spec: CryptoOhlcvSpec) -> CompiledCryptoOhlcvSpec:
    raw_dict = spec.model_dump(by_alias=True, exclude_none=True, mode="python")
    return CompiledCryptoOhlcvSpec(
        raw=spec,
        compiled_spec_hash=sha256_canonical(raw_dict),
        entry=compile_condition(spec.strategy.entry),
        exit=compile_condition(spec.strategy.exit),
        indicators=tuple(spec.strategy.indicators),
    )
