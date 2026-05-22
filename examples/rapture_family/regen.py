#!/usr/bin/env python3
"""Regenerate the rapture-family example.

LABEL: synthetic — extreme-NO Jesus/rapture-style market family. Stresses
the engine's NO-side semantics, range guards, and small-sample warnings.

Markets: 12 entries with prices clustered near 0.99 (long-tail extreme-NO).
Strategy: NO side, fixed_fraction.
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples._common import base_dataset, base_spec, write_json
from pancake_engine import BacktestConfig, EvidenceDataset, EvidenceSpec, run_backtest

DIR = Path(__file__).parent


def _utc_ts(year: int, month: int = 1, day: int = 1) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())


def main() -> int:
    rng = random.Random(13)
    rows = []
    for i in range(12):
        dec = _utc_ts(2024 + i // 12, (i % 12) + 1, 1)
        res = _utc_ts(2024 + (i + 6) // 12, ((i + 6) % 12) + 1, 1)
        # NO price clustered very close to 1.0 (i.e., market thinks rapture won't happen)
        no_price = round(rng.uniform(0.95, 0.99), 4)
        # In nearly all cases, the rapture doesn't happen → NO wins → strategy wins
        outcome = 0 if rng.random() > 0.05 else 1     # 95% NO wins
        target = 1
        rows.append({
            "mkt": f"m/RAPTURE_Y{2024 + i // 12:04d}",
            "dec_ts": dec,
            "res_ts": res,
            "price": no_price,
            "outcome": outcome,
            "alpha": round(rng.uniform(2.0, 4.0), 4),
            "target": target,
        })

    spec_dict = base_spec(
        name="rapture-family-extreme-no",
        side="NO",
        sizing_value=0.05,
        starting_capital=10000.0,
        entry_alpha_gte=2.0,
    )
    dataset_dict = base_dataset(dataset_id="ex_rapture", rows=rows, label="synthetic")

    spec = EvidenceSpec.model_validate(spec_dict)
    dataset = EvidenceDataset.model_validate(dataset_dict)
    obs_time = _utc_ts(2026, 1, 1)
    result = run_backtest(spec, dataset, BacktestConfig(observation_time=obs_time))

    write_json(DIR / "spec.json", spec_dict)
    write_json(DIR / "dataset.json", dataset_dict)
    write_json(DIR / "expected_result.json", {
        "result_hash": result.result_hash,
        "num_trades": result.metrics.standard.num_trades,
        "total_return": result.metrics.standard.total_return,
        "win_rate": result.metrics.standard.win_rate,
        "observation_time": obs_time,
    })
    print(f"rapture_family: result_hash={result.result_hash}")
    print(f"  num_trades={result.metrics.standard.num_trades} "
          f"total_return={result.metrics.standard.total_return:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
