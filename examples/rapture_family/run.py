#!/usr/bin/env python3
"""Run the rapture-family example and assert ``result_hash``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples._common import read_json
from pancake_engine import BacktestConfig, load_dataset, load_spec, run_backtest

DIR = Path(__file__).parent


def main() -> int:
    spec = load_spec(DIR / "spec.json")
    dataset = load_dataset(DIR / "dataset.json")
    expected = read_json(DIR / "expected_result.json")
    result = run_backtest(spec, dataset, BacktestConfig(observation_time=expected["observation_time"]))
    if result.result_hash != expected["result_hash"]:
        print(f"FAIL: result_hash mismatch", file=sys.stderr)
        print(f"  expected: {expected['result_hash']}", file=sys.stderr)
        print(f"  got:      {result.result_hash}", file=sys.stderr)
        return 1
    print(f"rapture_family OK: result_hash={result.result_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
