# batter

[![Test](https://github.com/usepancake/batter/actions/workflows/test.yml/badge.svg)](https://github.com/usepancake/batter/actions/workflows/test.yml)

`batter` is a deterministic Python research engine for prediction-market evidence-backed backtests. Given a backtest spec and an `EvidenceDataset`, it produces a canonical `result_hash` — identical bytes across ubuntu / macos / windows on Python 3.12+ — enabling reproducible research and auditability of strategy claims. Engine 0.4 adds Monte Carlo bootstrap confidence intervals and a sign-permutation Sharpe test so credibility signals travel with every result.

The PyPI package is `batter`; the Python module is `pancake_engine` (sklearn-style rename: `pip install batter` then `import pancake_engine`).

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

## License

Apache-2.0 — Copyright 2026 Michael Mustopo
