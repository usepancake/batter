#!/usr/bin/env python3
"""Run the BTC prediction-market hedge example via walk-forward.

Asserts ``aggregate_result_hash`` matches the committed expected value.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples._common import read_json
from pancake_engine import (
    BacktestConfig,
    WalkforwardConfig,
    load_dataset,
    load_spec,
    run_walkforward,
)

DIR = Path(__file__).parent


def main() -> int:
    spec = load_spec(DIR / "spec.json")
    dataset = load_dataset(DIR / "dataset.json")
    expected = read_json(DIR / "expected_result.json")
    wf_config = WalkforwardConfig(
        window_type="anchored",
        test_horizon="2QS",
        step="2QS",
        min_fold_count=2,
    )
    bt_config = BacktestConfig(observation_time=expected["observation_time"])
    result = run_walkforward(spec, dataset, wf_config, bt_config)
    if result.aggregate_result_hash != expected["aggregate_result_hash"]:
        print(f"FAIL: aggregate_result_hash mismatch", file=sys.stderr)
        print(f"  expected: {expected['aggregate_result_hash']}", file=sys.stderr)
        print(f"  got:      {result.aggregate_result_hash}", file=sys.stderr)
        return 1
    print(f"btc_pred_hedge OK: aggregate_result_hash={result.aggregate_result_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
