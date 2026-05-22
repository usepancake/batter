#!/usr/bin/env python3
"""Regenerate the BTC prediction-market hedge example.

LABEL: synthetic — Fed-cut odds signal + BTC forward returns engineered
deterministically. The COMMITTED dataset.json is the source of truth for the
test. This regen script does NOT require a Pancake data export; if real Fed
odds + BTC prices are wired in later, replace the synthetic generators below
and update the label to "hybrid" or "real".

Strategy: bet YES on a prediction-market binary when a synthetic
"fed_odds_jump" feature crosses threshold; resolution by the market's target.

Walk-forward demo: this is the example we run through ``run_walkforward`` to
exercise the PR-2 layer.
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples._common import base_dataset, base_spec, write_json
from pancake_engine import (
    BacktestConfig,
    EvidenceDataset,
    EvidenceSpec,
    WalkforwardConfig,
    run_walkforward,
)

DIR = Path(__file__).parent


def _utc_ts(year: int, month: int, day: int = 1) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())


def main() -> int:
    rng = random.Random(99)
    rows = []
    # 24 monthly decisions over 2 years
    for i in range(24):
        year = 2023 + i // 12
        month = (i % 12) + 1
        dec = _utc_ts(year, month, 5)
        # Resolve 30 days later
        res_year = year + (month // 12)
        res_month = (month % 12) + 1
        res = _utc_ts(res_year, res_month, 5)

        fed_odds_jump = rng.uniform(0, 0.25)     # synthetic feature, percentage-point jump
        target = 1 if fed_odds_jump >= 0.1 else 0
        outcome = target if rng.random() > 0.4 else 1 - target  # 60% target=outcome
        price = round(rng.uniform(0.3, 0.7), 4)

        rows.append({
            "mkt": f"m/FED_CUT_{year:04d}_{month:02d}",
            "dec_ts": dec,
            "res_ts": res,
            "price": price,
            "outcome": outcome,
            "alpha": round(fed_odds_jump, 4),
            "target": target,
        })

    spec_dict = base_spec(
        name="btc-pred-hedge-fed-cut",
        side="YES",
        sizing_value=0.04,
        starting_capital=10000.0,
        entry_alpha_gte=0.1,
    )
    # Bake provenance flag so feature lookahead warning suppresses cleanly
    dataset_dict = base_dataset(dataset_id="ex_btc_pred", rows=rows, label="synthetic")
    dataset_dict["provenance"]["feature_construction_verified_no_lookahead"] = True

    spec = EvidenceSpec.model_validate(spec_dict)
    dataset = EvidenceDataset.model_validate(dataset_dict)

    # Walk-forward over 4 anchored quarters
    wf_config = WalkforwardConfig(
        window_type="anchored",
        test_horizon="2QS",
        step="2QS",
        min_fold_count=2,        # allow 2-fold; only ~2 years of data
    )
    bt_config = BacktestConfig(observation_time=_utc_ts(2026, 1, 1))
    wf_result = run_walkforward(spec, dataset, wf_config, bt_config)

    write_json(DIR / "spec.json", spec_dict)
    write_json(DIR / "dataset.json", dataset_dict)
    write_json(DIR / "expected_result.json", {
        "aggregate_result_hash": wf_result.aggregate_result_hash,
        "fold_count": wf_result.aggregate.fold_count,
        "pooled_num_trades": wf_result.aggregate.pooled.num_trades,
        "observation_time": bt_config.observation_time,
    })
    print(f"btc_pred_hedge: aggregate_result_hash={wf_result.aggregate_result_hash}")
    print(f"  fold_count={wf_result.aggregate.fold_count} "
          f"pooled_num_trades={wf_result.aggregate.pooled.num_trades}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
