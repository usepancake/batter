# pancake-engine-py

**Pancake Engine 0.3** — deterministic Python research engine over `EvidenceDataset`.

> Engine 0.3 is **correctness-first, not TS parity**. Known TS divergences are documented in
> [`docs/research/pancake-engine-0.3-ts-divergences.md`](https://github.com/usepancake/pancake-production/blob/main/docs/research/pancake-engine-0.3-ts-divergences.md)
> in the `pancake-production` repo. The engine intentionally fixes documented bugs in
> `lib/evidence-runner/runner.ts`; receipts will differ on affected fixtures.

License: Apache-2.0.

## Status

**PR-0 — canonicalization substrate.** Ships:

- `pancake_engine.canonical.canonicalize(obj) -> bytes` — vendored ECMA-262 §6.1.6.1.13 NumberToString
  port, key-sorted, NFC string normalization, NaN/Inf/lone-surrogate/duplicate-key/non-string-key
  rejection. Byte-identical to V8 `JSON.stringify` on finite values.
- `pancake_engine.hash.sha256_canonical(obj) -> str` — SHA-256 over canonical bytes.
- `pancake_engine.types` — minimal pydantic v2 models for `EvidenceDataset` and `EvidenceSpec`
  at the I/O boundary. Validators enforce `slippage_bps ≥ 0`, `fee_bps ≥ 0`, `starting_capital > 0`.
- `pancake_engine.io.load` — strict JSON load with duplicate-key detection at parse time
  (via `object_pairs_hook`).
- CLI: `pancake-engine hash --dataset|--spec FILE`.

**Not yet shipped** (later PRs):

- Runner / event-time ledger / metrics — PR-1
- Walk-forward — PR-2
- Examples / notebooks — PR-2
- ResultEnvelope adapter — PR-4+

## Quickstart

```bash
git clone <repo>
cd pancake-engine-py
pip install -e ".[dev]"
node tests/fixtures/canonical/v8_oracle.js   # regenerate the V8 oracle expected bytes
pytest -v
```

Hash a dataset or spec from the CLI:

```bash
pancake-engine hash --dataset path/to/dataset.json
pancake-engine hash --spec    path/to/spec.json
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
