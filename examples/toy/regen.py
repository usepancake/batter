#!/usr/bin/env python3
"""Regenerate the toy example: 50-row synthetic dataset + spec + expected_result.

Run once, commit the outputs. ``run.py`` then asserts ``result_hash`` matches
the committed ``expected_result.json``.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Allow `python regen.py` from within the example dir
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples._common import base_dataset, base_spec, write_json
from pancake_engine import BacktestConfig, EvidenceDataset, EvidenceSpec, run_backtest

DIR = Path(__file__).parent
DAY = 86_400


def main() -> int:
    rng = random.Random(42)
    rows = []
    for i in range(50):
        alpha = rng.uniform(0.5, 5.0)
        target = rng.randint(0, 1)
        outcome = target if rng.random() > 0.45 else 1 - target  # 55% target=outcome
        rows.append({
            "mkt": "m/TOY",
            "dec_ts": i * DAY,
            "res_ts": i * DAY + DAY // 2,
            "price": round(rng.uniform(0.15, 0.85), 4),
            "outcome": outcome,
            "alpha": round(alpha, 4),
            "target": target,
        })

    spec_dict = base_spec(
        name="toy-example",
        side="YES",
        sizing_value=0.05,
        starting_capital=10000.0,
        entry_alpha_gte=2.0,
    )
    dataset_dict = base_dataset(dataset_id="ex_toy", rows=rows, label="synthetic")

    spec = EvidenceSpec.model_validate(spec_dict)
    dataset = EvidenceDataset.model_validate(dataset_dict)
    result = run_backtest(spec, dataset, BacktestConfig(observation_time=50 * DAY))

    write_json(DIR / "spec.json", spec_dict)
    write_json(DIR / "dataset.json", dataset_dict)
    write_json(DIR / "expected_result.json", {
        "result_hash": result.result_hash,
        "num_trades": result.metrics.standard.num_trades,
        "total_return": result.metrics.standard.total_return,
        "win_rate": result.metrics.standard.win_rate,
    })
    print(f"toy: result_hash={result.result_hash}")
    print(f"toy: num_trades={result.metrics.standard.num_trades} "
          f"total_return={result.metrics.standard.total_return:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
