# pancake-math

[![Test](https://github.com/michaelmustopo/pancake-engine-py/actions/workflows/test.yml/badge.svg)](https://github.com/michaelmustopo/pancake-engine-py/actions/workflows/test.yml)

`pancake-math` is a deterministic Python research engine for prediction-market evidence-backed backtests. Given a backtest spec and an `EvidenceDataset`, it produces a canonical `result_hash` — identical bytes on every run, every platform, every Python version — enabling reproducible research and auditability of strategy claims. Engine 0.4 adds Monte Carlo bootstrap confidence intervals and a sign-permutation Sharpe test so credibility signals travel with every result.

## Install

```bash
pip install git+https://github.com/michaelmustopo/pancake-engine-py.git@0.4.0
```

> PyPI publish via `pip install pancake-math` lands shortly.

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

The same `(spec, dataset, config)` produces the same `result_hash` on Python 3.11/3.12 x macOS/Linux. Verification method, fixture set, and numeric bounds are documented in [docs/math-audit-0.4.md §"Verification verdict"](docs/math-audit-0.4.md#verification-verdict).

## Cross-platform

CI enforces a 4-cell matrix (ubuntu-latest + macos-latest) x (Python 3.11 + 3.12). See the badge above.

## License

Apache-2.0 — Copyright 2026 Michael Mustopo
