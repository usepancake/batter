# batter

[![Test](https://github.com/usepancake/batter/actions/workflows/test.yml/badge.svg)](https://github.com/usepancake/batter/actions/workflows/test.yml)

`batter` is a deterministic Python research engine for prediction-market evidence-backed backtests. Given a backtest spec and an `EvidenceDataset`, it produces a canonical `result_hash` — identical bytes across ubuntu / macos / windows on Python 3.12+ — enabling reproducible research and auditability of strategy claims. Engine 0.4 adds Monte Carlo bootstrap confidence intervals and a sign-permutation Sharpe test so credibility signals travel with every result.

The PyPI package is `batter`; the Python module is `pancake_engine` (sklearn-style rename: `pip install batter` then `import pancake_engine`).

## Install

```bash
pip install git+https://github.com/usepancake/batter.git@0.4.0
```

> PyPI publish via `pip install batter` lands shortly.

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

Python 3.11 is out of scope for the byte-identity claim — see [docs/math-audit-0.4.md §"Known scope qualifier — Python 3.11"](docs/math-audit-0.4.md#known-scope-qualifier--python-311). Filed as a v1.4 follow-up.

## Cross-platform

CI enforces a 6-cell matrix (ubuntu-latest + macos-latest + windows-latest) × (Python 3.12 + 3.13). See the badge above.

## License

Apache-2.0 — Copyright 2026 Michael Mustopo
