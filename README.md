# pancake-engine-py

**Pancake Engine 0.3** — deterministic Python research engine over `EvidenceDataset`.

> Engine 0.3 is **correctness-first, not TS parity**. Known TS divergences are documented in
> [`docs/research/pancake-engine-0.3-ts-divergences.md`](https://github.com/usepancake/pancake-production/blob/main/docs/research/pancake-engine-0.3-ts-divergences.md)
> in the `pancake-production` repo. The engine intentionally fixes documented bugs in
> `lib/evidence-runner/runner.ts`; receipts will differ on affected fixtures.

License: Apache-2.0.

## Status

**PR-0 — canonicalization substrate** (committed at `292b03f`):

- `canonicalize(obj) -> bytes` — vendored ECMA-262 NumberToString port
- `sha256_canonical(obj) -> str` — SHA-256 over canonical bytes
- Minimal pydantic v2 models with cost / capital non-negativity validators
- Strict JSON load with duplicate-key detection at parse time
- `pancake-engine hash --dataset|--spec`

**PR-1 — event-time ledger runner**:

- `run_backtest(spec, dataset, config) -> BacktestResult` — pure function, deterministic
- Event-time ledger: no future cash leaks; same-timestamp `DECISION < RESOLUTION` ordering
- `fixed_fraction × available_cash` sizing; `mark_at_cost` mark policy with fees realized at entry
- `multiplicative_bps` slippage; binary $1/$0 frictionless settlement
- Standard metrics (total_return, cagr, sharpe, sortino, max_drawdown, win_rate) + PM metrics
  (Wilson 95% CI on win_rate, brier_crowd; brier_strategy null for rule-based specs)
- Credibility warnings (`IMPLAUSIBLY_HIGH_SHARPE`, `LOW_SAMPLE_SIZE`, `MARK_AT_COST_DRAWDOWN_MUTED`,
  `RUINED`, `NO_TRADES_GENERATED`, etc.)
- `pancake-engine validate --spec --dataset` and `pancake-engine run --spec --dataset --out`
- TS golden parity: 5 fixtures via `ts_runner_oracle.mjs` against real `runEvidenceBacktest`
  from pancake-production. 3 match within `1e-9`; 2 documented divergence (D-1 cash leak,
  D-11 fee realized at entry).

**PR-2 — frozen-spec walk-forward + 4 domain examples**:

- `run_walkforward(spec, dataset, wf_config, bt_config) -> WalkforwardResult` — pure function
- Fold schedule generators: `expanding`, `rolling`, `anchored` (calendar `MS` / `QS`)
- Per-fold `BacktestResult` (each with its own `result_hash`) + aggregate
  (`PooledMetrics`, `FoldMeanMetrics`, `FoldStdMetrics`, dispersion ratios)
- Walk-forward warnings: `EMPTY_FOLD`, `LOW_TRADES_IN_FOLD`, `UNEQUAL_FOLD_SIZE`,
  `WALKFORWARD_DISPERSION_HIGH`, `WALKFORWARD_SIGN_FLIP`, `WALKFORWARD_SINGLE_FOLD_CARRIES`,
  `OVERHANG_SKIPPED`, `FEATURE_LOOKAHEAD_UNCHECKED`, `OVERRIDE_MIN_FOLD_COUNT`
- Overhang policy: `allow_overhang` (default), `skip_overhang`. `truncate_at_window_end`
  raises `E_OVERHANG_TRUNCATION_UNSUPPORTED` until a mark-policy upgrade (PR-3+)
- `aggregate_result_hash` includes `walkforward_version` + `result_kind` to prevent
  collision with vanilla `BacktestResult.result_hash`
- `pancake-engine walkforward --spec --dataset --window-type --test-horizon --step ...`
- 4 domain examples (`examples/toy`, `jakarta_temperature`, `rapture_family`,
  `btc_pred_hedge`) — synthetic, deterministically regenerated via `regen.py`,
  hash-pinned via `expected_result.json`; tested through `run.py` smoke
- One Jupyter notebook (`notebooks/walkforward_tour.ipynb`) — rendering of test-pinned
  facts; executed in CI via `nbclient`
- Per-fold TS parity proven: `ts_walkforward_oracle.mjs` slices a fixture and calls
  real `runEvidenceBacktest` per slice; Engine 0.3 matches within `1e-9`
- `test_no_domain_leak` guards `pancake_engine/` from example-specific tokens

**Not yet shipped** (later PRs):

- Refit walk-forward — 0.4
- ResultEnvelope adapter — PR-4+
- Other sizing / mark / slippage modes, multi-outcome, `fair_probability`, CLV, benchmark,
  cost-sensitivity, bootstrap CIs — PR-3+

## Quickstart

```bash
git clone <repo>
cd pancake-engine-py
pip install -e ".[dev]"
node tests/fixtures/canonical/v8_oracle.js   # regenerate the V8 oracle expected bytes
pytest -v
```

Hash, validate, run, and walk-forward from the CLI:

```bash
pancake-engine hash         --dataset path/to/dataset.json
pancake-engine hash         --spec    path/to/spec.json
pancake-engine validate     --spec path/to/spec.json --dataset path/to/dataset.json
pancake-engine run          --spec path/to/spec.json --dataset path/to/dataset.json \
                            --out result.json --observation-time 1700000000
pancake-engine walkforward  --spec path/to/spec.json --dataset path/to/dataset.json \
                            --window-type anchored --test-horizon QS --step QS \
                            --min-fold-count 3 --observation-time 1730000000 \
                            --out wf_result.json
```

Run an example:

```bash
python examples/toy/run.py
python examples/jakarta_temperature/run.py
python examples/rapture_family/run.py
python examples/btc_pred_hedge/run.py     # uses walk-forward
```

## Canonical-form contract

The canonical bytes of an object are produced by `canonicalize(obj)`:

- `null`, `true`, `false`, integers: literal.
- Floats: ECMA-262 NumberToString (Steele-Dybvig / Ryu shortest). `NaN`, `+Inf`, `-Inf` rejected.
  `-0` normalized to `0`.
- Integers with `|x| > 2**53` rejected (precision loss on JS round-trip).
- Strings: NFC-normalized, JSON-escaped per RFC 8259. Lone surrogates rejected.
- Arrays: order preserved; never sorted.
- Objects: keys sorted by Unicode codepoint, recursively. Duplicate keys rejected at parse time
  via `object_pairs_hook` (Python's default `json.loads` would silently keep the last value).
- `datetime` / unsupported types: rejected. Callers serialize times to unix-integer seconds before
  canonicalize.

The same `(spec, dataset, config)` therefore produces the same `result_hash` on Python 3.11/3.12/3.13
× macOS/Linux/Windows × Node 20/22. CI enforces the full matrix.

## Cross-runtime parity

Python `canonicalize(x)` is byte-equal to V8 `JSON.stringify(x)` on finite values. The
`tests/fixtures/canonical/v8_oracle.js` script writes the V8 baseline; `tests/test_canonical.py`
asserts byte-equality on a 50-case numeric fixture covering integers, decimals, scientific
notation, subnormals, the `1e-7` / `1e21` boundaries, and `Number.MAX_VALUE`.

### TS hash parity

For the substrate (canonicalize + SHA-256), Python `sha256_canonical()` matches the TS
evidence-runner's `hashSchema()` / `hashRows()` from `pancake-production` byte-for-byte. The
expected hashes in `tests/fixtures/canonical/ts_hashes.json` are **computed by the real TS code**
via `tests/fixtures/canonical/ts_hash_oracle.mjs` (which runs `tsx` against a sibling
`pancake-production` checkout). The Python test (`tests/test_ts_hash_parity.py`) is standalone
and never imports anything from `pancake-production`.

Regenerate the oracle only when fixtures change or when `pancake-production`'s
`lib/data/evidence/hash.ts` / `lib/spec/canonicalize.ts` changes:

```bash
# default sibling-checkout path: ../pancake-production
node tests/fixtures/canonical/ts_hash_oracle.mjs

# or specify an absolute path:
PANCAKE_PRODUCTION_ROOT=/abs/path/to/pancake-production \
  node tests/fixtures/canonical/ts_hash_oracle.mjs
```

## Docs

The canonical engine specification lives in `pancake-production/docs/research/`:

- `pancake-engine-0.3-research.md` — open-source engine landscape + license doctrine
- `pancake-engine-0.3-architecture.md` — full architecture, capital ledger, metrics, warnings
- `pancake-engine-0.3-pr-plan.md` — PR-0 / PR-1 / PR-2 / PR-3+ scope
- `pancake-engine-0.3-ts-divergences.md` — every place Engine 0.3 disagrees with the TS runner

The engine is built against those docs. If code and docs disagree, docs are the source of truth;
file a divergence entry and update the code.

## Doctrine

Engine 0.3 is **correctness-first, not TS parity**. Known TS divergences are documented.
