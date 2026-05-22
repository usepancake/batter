#!/usr/bin/env python3
"""Regenerate the Jakarta temperature example.

LABEL: synthetic — generated deterministically from random.Random(7). The
"temperature" values are NOT real Jakarta weather observations. If a real
weather feed is wired in later, this example becomes "hybrid" (real features,
synthetic resolution rule) and the label is updated to reflect that.

Strategy: bet YES on a "hot day" market when synthetic morning_temp_c >= 32.
Resolution is set by an oracle rule (target column).
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


def _utc_ts(year: int, month: int, day: int, hour: int = 0) -> int:
    return int(datetime(year, month, day, hour, tzinfo=timezone.utc).timestamp())


def main() -> int:
    rng = random.Random(7)
    rows = []
    for d in range(60):
        dec = _utc_ts(2024, 1, 1) + d * 86_400 + 6 * 3600    # 06:00 UTC morning forecast
        res = _utc_ts(2024, 1, 1) + d * 86_400 + 22 * 3600   # 22:00 UTC settle
        # Synthetic morning temp 28-36 C with seasonal swing
        morning_temp = 31.5 + rng.uniform(-3, 4)
        target = 1 if morning_temp >= 32 else 0
        # Market often correlates with target but not always
        outcome = target if rng.random() > 0.35 else 1 - target
        price = round(rng.uniform(0.25, 0.75), 4)
        rows.append({
            "mkt": f"m/JKT_HOT_{d:02d}",
            "dec_ts": dec,
            "res_ts": res,
            "price": price,
            "outcome": outcome,
            "alpha": round(morning_temp, 4),
            "target": target,
        })

    spec_dict = base_spec(
        name="jakarta-temperature-hot-day",
        side="YES",
        sizing_value=0.04,
        starting_capital=10000.0,
        entry_alpha_gte=32.0,
    )
    dataset_dict = base_dataset(dataset_id="ex_jkt_temp", rows=rows, label="synthetic")

    spec = EvidenceSpec.model_validate(spec_dict)
    dataset = EvidenceDataset.model_validate(dataset_dict)
    obs_time = _utc_ts(2024, 5, 1)
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
    print(f"jakarta_temperature: result_hash={result.result_hash}")
    print(f"  num_trades={result.metrics.standard.num_trades} "
          f"total_return={result.metrics.standard.total_return:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
