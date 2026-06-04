"""Crypto-OHLCV spec + dataset contract (ADR-0043 §2.3, P1).

Pydantic v2 models, parallel to `pancake_engine.types` (the evidence family) but
isolated. Validators reject the dishonest / malformed classes at load time:
negative costs, non-positive capital, bad OHLC bars, unknown indicator refs,
out-of-range sizing. Compiler (P2) and runner (P3) consume these; no math here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "OhlcvBar",
    "OhlcvDataset",
    "Indicator",
    "Operand",
    "Condition",
    "OhlcvSizing",
    "OhlcvCosts",
    "CryptoOhlcvStrategy",
    "CryptoOhlcvSpec",
]

_ERR = "E_CRYPTO_OHLCV_SPEC_INVALID"


# ---------------------------------------------------------------------------
# Dataset — an OHLCV bar series for one instrument
# ---------------------------------------------------------------------------


class OhlcvBar(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: int  # bar-OPEN epoch seconds (UTC), aligned to the dataset bar_period
    open: float
    high: float
    low: float
    close: float
    volume: float

    @field_validator("volume")
    @classmethod
    def _volume_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"{_ERR}: bar volume must be >= 0 (got {v})")
        return v

    @model_validator(mode="after")
    def _ohlc_invariant(self) -> OhlcvBar:
        if not (self.high >= self.low):
            raise ValueError(f"{_ERR}: bar high < low (high={self.high} low={self.low})")
        lo, hi = self.low, self.high
        if not (lo <= self.open <= hi and lo <= self.close <= hi):
            raise ValueError(
                f"{_ERR}: bar OHLC invariant violated "
                f"(o={self.open} h={hi} l={lo} c={self.close})"
            )
        return self


class OhlcvDataset(BaseModel):
    model_config = ConfigDict(extra="allow")

    instrument_id: str
    currency: str = "USD"
    bar_period: Literal["1m"] = "1m"  # v1 locked to price_bars 1-min
    bars: list[OhlcvBar]

    @model_validator(mode="after")
    def _bars_ordered_nonempty(self) -> OhlcvDataset:
        if len(self.bars) == 0:
            raise ValueError(f"{_ERR}: dataset has no bars")
        prev = None
        for b in self.bars:
            if prev is not None and not (b.t > prev):
                raise ValueError(
                    f"{_ERR}: bars must be strictly increasing in t "
                    f"(got {b.t} after {prev})"
                )
            prev = b.t
        return self


# ---------------------------------------------------------------------------
# Strategy DSL — minimal (SMA/EMA/RSI + cross/threshold + fixed-fraction)
# ---------------------------------------------------------------------------

IndicatorKind = Literal["sma", "ema", "rsi"]
PriceField = Literal["open", "high", "low", "close"]


class Indicator(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str  # referenceable handle, e.g. "sma_fast"
    kind: IndicatorKind
    period: int
    source: PriceField = "close"

    @field_validator("period")
    @classmethod
    def _period_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"{_ERR}: indicator period must be > 0 (got {v})")
        return v


class Operand(BaseModel):
    """A comparison operand: a live price field, a named indicator, or a const."""

    model_config = ConfigDict(extra="forbid")

    ref: Literal["price", "indicator", "const"]
    field: PriceField | None = None
    indicator_id: str | None = None
    value: float | None = None

    @model_validator(mode="after")
    def _operand_consistent(self) -> Operand:
        if self.ref == "price" and self.field is None:
            raise ValueError(f"{_ERR}: price operand requires `field`")
        if self.ref == "indicator" and not self.indicator_id:
            raise ValueError(f"{_ERR}: indicator operand requires `indicator_id`")
        if self.ref == "const" and self.value is None:
            raise ValueError(f"{_ERR}: const operand requires `value`")
        return self


CompareOp = Literal["gt", "lt", "gte", "lte", "cross_above", "cross_below"]


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: CompareOp
    left: Operand
    right: Operand


class OhlcvSizing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["fixed_fraction"]  # v1 minimal
    value: float

    @field_validator("value")
    @classmethod
    def _fraction_range(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"{_ERR}: fixed_fraction value must be in (0, 1] (got {v})")
        return v


class OhlcvCosts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slippage_bps: float
    fee_bps: float

    @field_validator("slippage_bps")
    @classmethod
    def _slip_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(
                f"{_ERR}: slippage_bps must be >= 0 (got {v}); "
                "negative slippage represents favorable fills (free money)"
            )
        return v

    @field_validator("fee_bps")
    @classmethod
    def _fee_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(
                f"{_ERR}: fee_bps must be >= 0 (got {v}); "
                "negative fees represent rebates (free money)"
            )
        return v


class CryptoOhlcvStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    side: Literal["long", "short"] = "long"
    indicators: list[Indicator] = Field(default_factory=list)
    entry: Condition
    exit: Condition
    sizing: OhlcvSizing

    @model_validator(mode="after")
    def _refs_resolve(self) -> CryptoOhlcvStrategy:
        ids = [i.id for i in self.indicators]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{_ERR}: duplicate indicator id(s) in {ids}")
        known = set(ids)
        for cond_name, cond in (("entry", self.entry), ("exit", self.exit)):
            for side_name, operand in (("left", cond.left), ("right", cond.right)):
                if operand.ref == "indicator" and operand.indicator_id not in known:
                    raise ValueError(
                        f"{_ERR}: {cond_name}.{side_name} references unknown "
                        f"indicator '{operand.indicator_id}' (known: {sorted(known)})"
                    )
        return self


class CryptoOhlcvSpec(BaseModel):
    """The crypto-OHLCV strategy spec. The dataset is loaded + passed separately
    (run path validates spec.instrument_id == dataset.instrument_id)."""

    model_config = ConfigDict(extra="allow")

    spec_family: Literal["crypto-ohlcv-spec"]
    spec_version: Literal["0.1"]
    name: str
    instrument_id: str
    fill_timing: Literal["next_bar_open"] = "next_bar_open"  # ADR-0043 locked pick
    strategy: CryptoOhlcvStrategy
    costs: OhlcvCosts
    starting_capital: float

    @field_validator("starting_capital")
    @classmethod
    def _starting_capital_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"{_ERR}: starting_capital must be > 0 (got {v})")
        return v
