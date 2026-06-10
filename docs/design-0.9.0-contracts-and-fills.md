# 0.9.0 design — DatasetContract Seam + fill-model registry

Status: design locked by PMO 2026-06-10 (ADR-0049 ladder; ADR-0043 locks the crypto
domain choices). These are the two hash-critical shapes of 0.9.0 — everything else
in the release is additive around them. The 0.9.0 hash break = crypto domain + PM
calibration ECE + (registry field presence for specs that use it).

## 1. DatasetContract (the Seam)

One typed contract per asset domain, validated at dataset registration AND at
run time (validate_dataset consults the contract, not ad-hoc rules).

```python
@dataclass(frozen=True)
class DatasetContract:
    domain: str                          # "prediction_market" | "crypto_ohlcv" | "macro_signal" | ...
    required_roles: tuple[RoleSpec, ...] # name → type, semantic_role, value domain
    time_model: str                      # "event_resolution" (PM) | "bar_series" (crypto)
    resolution_semantics: str | None     # PM: "binary_payout"; crypto: None
    fill_reference: str                  # PM: "entry_price_col"; crypto: "next_bar_open" (ADR-0043)
```

- `PredictionMarketContract` codifies today's implicit rules (decision/resolution
  times, entry_price ∈ (0,1) literal SIDE price, resolved_outcome_numeric, the
  look-ahead + monotonicity row invariants currently in validate/dataset.py).
- `CryptoOHLCVContract` (ADR-0043): 1-min OHLCV bars from price_bars; minimal DSL
  (threshold/cross + a few indicators); fills at NEXT BAR OPEN; positions closed at
  window end (exit conditions apply — the tick/paper exit semantics generalize here
  because bar_series domains DO observe mid-life bars in backtest).
- Spec selects its domain via `spec_family` (existing field) → contract lookup.
  Registration-time validation moves into the contract; engine dispatch
  (event loop vs bar loop) keys off `time_model`.
- Macro (`MacroSignalContract`) and sports are later contracts on the same Seam —
  no engine changes, by construction.

## 2. Fill-model registry (declared, versioned, hashed — never user code)

New optional spec field (None default → existing specs hash byte-identically):

```json
"costs": { "slippage_bps": ..., "fee_bps": ...,
           "fill_model": { "name": "static_bps", "version": 1, "params": { } } }
```

- Registry = engine-side dict name→(version→implementation). 0.9.0 ships:
  - `static_bps@1` — exactly today's math (the implicit default; specs that omit
    fill_model get it, and their hashes DO NOT change).
  - `book_replay@1` (PM) — fills walk the contemporaneous captured L2 slice
    (ADR-0041 data): cumulative consumption of ask levels for buys; the book-slice
    dataset id + its rows_sha256 are PINNED in the result envelope (additive
    fields) so replay-fills stay verifiable. Time-to-resolution covariate enters
    as `params: {"ttr_fill_adjustment": true}` — clean-room implementation
    (homerun is AGPL; concept only).
  - `next_bar_open@1` (crypto) — ADR-0043's lock, expressed as a registry entry.
- Hash policy: fill_model is part of the spec → compiled_spec_hash → result_hash.
  Unknown name/version → E_EVIDENCE_SPEC_INVALID (no silent fallback — the 0.6.0
  always-true lesson). Determinism rule applies to model internals (IEEE-exact /
  decimal only).
- SimFillRouter (paper) consumes the same registry → live/paper parity by
  construction; the 0.10.0 live adapter swaps the router, not the models.

## 3. Sequencing inside 0.9.0 (PMO)

Wave 1 (running): paper guards, exit-on-tick, sensitivity heatmap (additive).
Wave 2: registry + static_bps@1 + contract dataclasses + PM contract extraction
        (pure refactor of validate/dataset.py rules into the contract — byte-
        identical behavior, proven by the suite + examples).
Wave 3: crypto-OHLCV P1–P5 on the Seam (THE break; ECE rides it) + golden regen.
Wave 4: book_replay@1 + TTR covariate (additive registry entries; may slip to
        0.9.x without a second break) + verify CLI + R2 bundle + ledger seam +
        PBO/CPCV. Platform lane: rerun pass (mints the 2 pending corrections),
        receipt result_hash persistence (gap found 2026-06-10: receipts never
        stored result_hash — verify needs it).
```
