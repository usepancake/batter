# batter

[![Test](https://github.com/usepancake/batter/actions/workflows/test.yml/badge.svg)](https://github.com/usepancake/batter/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/batter.svg)](https://pypi.org/project/batter/)
[![Python versions](https://img.shields.io/pypi/pyversions/batter.svg)](https://pypi.org/project/batter/)

Built for [Pancake](https://usepancake.com) — receipts at `https://usepancake.com/r/<receipt-id>` use this engine to verify strategy math.

`batter` is a deterministic Python research engine for prediction-market evidence-backed backtests. Given a backtest spec and an `EvidenceDataset`, it produces a canonical `result_hash` — identical bytes across ubuntu / macos / windows on Python 3.12+ — enabling reproducible research and auditability of strategy claims. Engine 0.4 adds Monte Carlo bootstrap confidence intervals and a sign-permutation Sharpe test so credibility signals travel with every result.

The PyPI package is `batter`; the Python module is `pancake_engine` (sklearn-style rename: `pip install batter` then `import pancake_engine`).

## What is this for?

[Pancake](https://usepancake.com/engine) is a prediction-market research platform. `batter` is the math layer, extracted as a standalone package so the formulas can be verified independently of the platform.

When a strategy backtest runs on Pancake, it runs through `batter`. The platform stores the `result_hash`; anyone can reproduce that hash locally by running the same spec and dataset through `pip install batter`. This is the platform connection: batter has origin in Pancake but is genuinely independently usable by anyone doing prediction-market research.

## Install

```bash
pip install batter
```

## Quickstart

```python
import json
from pancake_engine import run_backtest, BacktestSpec, EvidenceDataset, BacktestConfig

spec    = BacktestSpec(**json.load(open("spec.json")))
dataset = EvidenceDataset(**json.load(open("dataset.json")))
config  = BacktestConfig()

result = run_backtest(spec, dataset, config)
print(result.result_hash)   # deterministic SHA-256 over canonical JSON
print(result.metrics.sharpe)
print(result.bootstrap_ci)  # 95% CI on cagr / sharpe / sortino (0.4+)
```

## Determinism

The same `(spec, dataset, config)` produces the same `result_hash` across ubuntu / macos / windows on Python 3.12+. Verification method, fixture set, and numeric bounds are documented in [docs/math-audit-0.4.md §"Verification verdict"](docs/math-audit-0.4.md#verification-verdict).

**Supported Python versions: 3.12 and 3.13.** Python 3.11 is permanently out of scope — `sum()` semantics changed in 3.12 (compensated float accumulation), causing the bootstrap CI values to differ by 1 ULP and producing a different `result_hash`. No code change can reconcile this without reverse-engineering 3.12's exact internal accumulation path. See [docs/py311-investigation-2026-05-27.md](docs/py311-investigation-2026-05-27.md) for the full root-cause analysis and [docs/math-audit-0.4.md §"Known scope qualifier — Python 3.11"](docs/math-audit-0.4.md#known-scope-qualifier--python-311) for the audit entry.

## Cross-platform

CI enforces a 6-cell matrix (ubuntu-latest + macos-latest + windows-latest) × (Python 3.12 + 3.13). See the badge above.

## Cite batter

If you use batter in academic or independent research, please cite:

```bibtex
@software{mustopo2026batter,
  author       = {Mustopo, Michael},
  title        = {batter: Deterministic Python research engine for
                  prediction-market evidence-backed backtests},
  year         = {2026},
  version      = {0.6.0},
  url          = {https://usepancake.com/engine},
  repository   = {https://github.com/usepancake/batter},
  license      = {Apache-2.0},
  note         = {The math layer of usepancake.com. Produces canonical
                  SHA-256 result hashes reproducible across Ubuntu,
                  macOS, and Windows on Python 3.12+.}
}
```

## Verify any receipt

No trusted party required. Given a self-contained bundle (spec + inline dataset + expected hash), `batter verify` re-runs the engine locally and checks that the computed `result_hash` matches the declared one. Anyone can audit a Pancake receipt independently.

```bash
pip install batter

# local bundle file
batter verify --bundle receipt-bundle.json

# or fetch from a URL directly
batter verify --url https://usepancake.com/r/<receipt-id>/bundle.json
```

**Exit codes:** `0` verified · `1` hash or dataset integrity mismatch · `2` input/validation error · `3` unverifiable (pointer dataset — rows not inline, license-gated)

**JSON output** (stdout):

```json
{
  "verified": true,
  "expected": "<sha256>",
  "computed": "<sha256>",
  "engine_version": "0.8.1",
  "num_trades": 42
}
```

**Bundle shapes accepted:**

- **regen-style** (what `examples/*/regen.py` produces): `{spec, dataset, config?, expected_result_hash: "<sha256>"}`
- **fixture-style**: `{spec, dataset, config?, expected: {result_hash: "<sha256>", ...}}`

The dataset must carry `storage_mode: "inline"` with rows present. Pointer datasets (rows held under license) print a clear message and exit 3.

`batter verify` also checks dataset integrity first: it recomputes `rows_sha256` and `schema_sha256` over the bundle's actual bytes and compares them to the declared values. A tampered bundle is caught before the engine even runs.

If the bundle declares an `engine_version` that differs from the installed version, a warning is printed — `result_hash` values are only comparable under the same `ENGINE_VERSION`.

## License

Apache-2.0 — Copyright 2026 Michael Mustopo

## See also

- [usepancake.com](https://usepancake.com) — the prediction-market research platform
- [usepancake.com/engine](https://usepancake.com/engine) — batter's platform page (methodology, JSON-LD, citation)
- [usepancake.com/methodology](https://usepancake.com/methodology) — platform methodology
