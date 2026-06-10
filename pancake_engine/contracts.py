"""DatasetContract Seam (Wave 2 + Wave 3 + Wave 4, 0.9.0).

Design doc: docs/design-0.9.0-contracts-and-fills.md §1.

One typed contract per asset domain, consulted by validate_dataset at
dataset registration and run time.  The contracts codify rules that were
previously implicit / ad-hoc in validate/dataset.py.

Wave 2 ships: PredictionMarketContract (PM domain).
Wave 3 ships: CryptoOHLCVContract (bar-series domain; ADR-0043).
Wave 4 ships: MacroSignalContract (reference_series domain).

The refactor is PURE: validate_dataset's observable behavior (error codes,
messages, warning emission, role_lookup) does not change — the existing
validation tests are the safety net.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "RoleSpec",
    "DatasetContract",
    "PredictionMarketContract",
    "CryptoOHLCVContract",
    "MacroSignalContract",
    "contract_for_spec_family",
    "contract_for_domain",
]


@dataclass(frozen=True)
class RoleSpec:
    """Describes one required semantic role in a DatasetContract.

    ``name``          — the SemanticRole literal (e.g. "entry_price").
    ``col_type``      — expected column type (ColumnType literal).
    ``value_range``   — (low, high) or None; None means the contract applies a
                        domain-specific rule instead (see DatasetContract docs).
    ``nullable``      — True when null rows are legal (resolved_outcome_numeric).
    """

    name: str
    col_type: str
    value_range: tuple[float, float] | None = None
    nullable: bool = False


@dataclass(frozen=True)
class DatasetContract:
    """Typed contract for one asset domain.

    Fields mirror the design doc §1 shape.  The contract is consulted by
    validate_dataset; engine dispatch keys off ``time_model``.
    """

    domain: str
    """'prediction_market' | 'crypto_ohlcv' | 'macro_signal' | ..."""

    required_roles: tuple[RoleSpec, ...]
    """Required semantic roles in insertion order."""

    time_model: str
    """'event_resolution' (PM) | 'bar_series' (crypto/macro)."""

    resolution_semantics: str | None
    """PM: 'binary_payout'; bar-series: None."""

    fill_reference: str | None
    """PM: 'entry_price_col'; crypto: 'next_bar_open'; macro: None (no fills)."""


# ---------------------------------------------------------------------------
# PredictionMarketContract — codifies today's implicit PM rules
# ---------------------------------------------------------------------------

PredictionMarketContract = DatasetContract(
    domain="prediction_market",
    required_roles=(
        RoleSpec(name="market_link",               col_type="string"),
        RoleSpec(name="decision_time",             col_type="int"),
        RoleSpec(name="resolution_time",           col_type="int"),
        # entry_price: probability in (0, 1) — the literal SIDE price.
        # The contract does NOT declare a static [0,1] value_range here because
        # validate_dataset applies the spec/dataset-declared range FIRST and only
        # falls back to the contract-level "entry_price is a probability" rule
        # when no range is declared (the 0.7.2 audit gap closure).  This matches
        # the existing validate/dataset.py behavior exactly.
        RoleSpec(name="entry_price",               col_type="number", value_range=None),
        RoleSpec(name="resolved_outcome_numeric",  col_type="int",    nullable=True),
    ),
    time_model="event_resolution",
    resolution_semantics="binary_payout",
    fill_reference="entry_price_col",
)


# ---------------------------------------------------------------------------
# CryptoOHLCVContract — codifies the ADR-0043 bar-series domain rules
# ---------------------------------------------------------------------------

CryptoOHLCVContract = DatasetContract(
    domain="crypto_ohlcv",
    required_roles=(
        # instrument_id: string key linking the spec to the dataset.
        RoleSpec(name="instrument_id", col_type="string"),
        # bar_period: locked to "1m" in v1 (price_bars 1-min source, ADR-0043).
        RoleSpec(name="bar_period", col_type="string"),
        # bars: the OHLCV bar series (open/high/low/close/volume per bar-open timestamp).
        # value_range=None because bars is a list of structured objects, not a
        # scalar column; range validation lives in OhlcvBar's pydantic validators.
        RoleSpec(name="bars", col_type="array", value_range=None),
    ),
    time_model="bar_series",
    resolution_semantics=None,          # no binary payout; P&L is continuous
    fill_reference="next_bar_open",     # ADR-0043 locked pick; registry: next_bar_open@1
)


# ---------------------------------------------------------------------------
# MacroSignalContract — FEATURE-PROVIDER domain (Wave 4)
#
# Macro datasets (e.g. FRED reference series) are validated against this
# contract via validate_reference_dataset in pancake_engine/validate/macro.py.
#
# time_model = "reference_series":  a new value meaning "this domain has no
# engine runner".  Macro datasets are joined into evidence rows UPSTREAM by
# the platform; the engine treats the resulting columns as ordinary declared
# features in PM EvidenceSpecs.  No MacroSignalContract runner will ever be
# added to the engine dispatch loop — the Seam is designed so that joining
# happens before the spec reaches the engine.
#
# Platform flow example:
#   1. Ingest FRED series → reference_observations rows (validated here).
#   2. Platform left-joins reference rows into evidence rows on
#      observation_time ≤ decision_time, per series.
#   3. PM EvidenceSpec declares the joined column as a feature; the engine
#      evaluates it identically to any other feature column.
# ---------------------------------------------------------------------------

MacroSignalContract = DatasetContract(
    domain="macro_signal",
    required_roles=(
        # observation_time: epoch seconds (int).  Must be monotone
        # non-decreasing per series_id.  No duplicates per (series_id,
        # observation_time) — the data point at a given timestamp is unique.
        RoleSpec(name="observation_time", col_type="int"),
        # series_id: string key identifying the time series (e.g. "UNRATE").
        RoleSpec(name="series_id", col_type="string"),
        # value: the observed scalar (finite float or int).  No range
        # constraint — macro series can be any real number (e.g. unemployment
        # rates, real interest rates, spreads, price indices).
        RoleSpec(name="value", col_type="number"),
    ),
    time_model="reference_series",
    resolution_semantics=None,   # no binary payout; this is a feature source
    fill_reference=None,         # macro datasets are never traded directly
)


# ---------------------------------------------------------------------------
# Registry: spec_family → DatasetContract  (PM + crypto, keyed by spec_family)
# Domain registry: domain string → DatasetContract  (all domains)
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, DatasetContract] = {
    "pancake-evidence-spec": PredictionMarketContract,
    "crypto-ohlcv-spec": CryptoOHLCVContract,
}

_DOMAIN_REGISTRY: dict[str, DatasetContract] = {
    "prediction_market": PredictionMarketContract,
    "crypto_ohlcv": CryptoOHLCVContract,
    "macro_signal": MacroSignalContract,
}


def contract_for_spec_family(spec_family: str) -> DatasetContract:
    """Return the DatasetContract for a given spec_family.

    The lookup is total for all spec_family values that pass pydantic
    validation ('pancake-evidence-spec' and 'crypto-ohlcv-spec' are the
    currently valid Literals).  An unknown family is therefore unreachable
    at runtime — pydantic rejects it before we ever call this.  The KeyError
    path is left as a defensive assertion so that future spec_family additions
    that forget to register a contract fail loudly.
    """
    contract = _REGISTRY.get(spec_family)
    if contract is None:
        raise KeyError(
            f"No DatasetContract registered for spec_family={spec_family!r}. "
            "Register a contract in pancake_engine/contracts.py."
        )
    return contract


def contract_for_domain(domain: str) -> DatasetContract:
    """Return the DatasetContract for an explicit domain string.

    Used by validate_reference_dataset and any caller that routes by domain
    rather than spec_family (e.g. the macro reference-data validation path,
    which has no spec_family because macro datasets are not directly run).
    Raises KeyError for unknown domains.
    """
    contract = _DOMAIN_REGISTRY.get(domain)
    if contract is None:
        raise KeyError(
            f"No DatasetContract registered for domain={domain!r}. "
            "Register a contract in pancake_engine/contracts.py."
        )
    return contract
