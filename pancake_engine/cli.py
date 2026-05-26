"""Pancake Engine 0.3 CLI.

PR-1 subcommands: ``hash`` (PR-0), ``validate``, ``run``.

Engine 0.3 is correctness-first, not TS parity. Known TS divergences are documented
in pancake-production/docs/research/pancake-engine-0.3-ts-divergences.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .__version__ import ENGINE, ENGINE_VERSION
from .config import BacktestConfig, WalkforwardConfig
from .hash import sha256_canonical
from .io.dump import dump_result
from .io.load import load_dataset, load_json, load_spec
from .runner import run_backtest
from .validate import validate_dataset, validate_spec
from .walkforward import run_walkforward

__all__ = ["main", "build_parser"]


def _parse_horizon(value: str) -> "int | str":
    """Accept '86400' (seconds) or 'MS'/'3MS'/'QS' (calendar anchor)."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def cmd_hash(args: argparse.Namespace) -> int:
    if args.dataset:
        obj = load_json(args.dataset)
        if not isinstance(obj, dict):
            print("error: dataset file must contain a JSON object", file=sys.stderr)
            return 2
        schema = obj.get("schema")
        rows = obj.get("rows_inline")
        out = {
            "engine": ENGINE,
            "engine_version": ENGINE_VERSION,
            "schema_sha256": sha256_canonical(schema) if schema is not None else None,
            "rows_sha256": sha256_canonical(rows) if rows is not None else None,
            "row_count": len(rows) if isinstance(rows, list) else None,
        }
        print(json.dumps(out, indent=2))
        return 0
    if args.spec:
        obj = load_json(args.spec)
        out = {
            "engine": ENGINE,
            "engine_version": ENGINE_VERSION,
            "source_spec_hash": sha256_canonical(obj),
        }
        print(json.dumps(out, indent=2))
        return 0
    print("error: --dataset or --spec required", file=sys.stderr)
    return 2


def cmd_validate(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    dataset = load_dataset(args.dataset)
    verdict = validate_spec(spec)
    dataset_verdict, _ = validate_dataset(dataset, spec)
    verdict.merge(dataset_verdict)
    print(json.dumps(verdict.to_dict(), indent=2))
    return 0 if verdict.ok else 1


def cmd_run(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    dataset = load_dataset(args.dataset)
    config = BacktestConfig(observation_time=args.observation_time)
    result = run_backtest(spec, dataset, config)
    if args.out:
        dump_result(result, args.out, indent=2 if args.pretty else None)
        print(json.dumps({
            "engine": result.engine,
            "engine_version": result.engine_version,
            "engine_mode": result.engine_mode,
            "result_hash": result.result_hash,
            "out": args.out,
            "validation_ok": result.validation.ok,
            "num_trades": result.metrics.standard.num_trades,
            "warnings": len(result.warnings),
        }, indent=2))
    else:
        print(json.dumps(result.to_dict(), indent=2 if args.pretty else None))
    return 0 if result.validation.ok else 1


def cmd_walkforward(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    dataset = load_dataset(args.dataset)
    bt_config = BacktestConfig(observation_time=args.observation_time)
    wf_config = WalkforwardConfig(
        window_type=args.window_type,
        test_horizon=_parse_horizon(args.test_horizon),
        step=_parse_horizon(args.step),
        train_horizon=_parse_horizon(args.train_horizon) if args.train_horizon else None,
        min_fold_count=args.min_fold_count,
        resolution_policy=args.resolution_policy,
    )
    result = run_walkforward(spec, dataset, wf_config, bt_config)
    if args.out:
        from pathlib import Path
        from .canonical import canonical_string
        if args.pretty:
            Path(args.out).write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True),
                                      encoding="utf-8")
        else:
            Path(args.out).write_text(canonical_string(result.to_dict()), encoding="utf-8")
        print(json.dumps({
            "engine": result.engine,
            "engine_version": result.engine_version,
            "walkforward_version": result.walkforward_version,
            "aggregate_result_hash": result.aggregate_result_hash,
            "out": args.out,
            "validation_ok": result.validation.ok,
            "fold_count": result.aggregate.fold_count,
            "pooled_num_trades": result.aggregate.pooled.num_trades,
            "warnings": len(result.warnings),
        }, indent=2))
    else:
        print(json.dumps(result.to_dict(), indent=2 if args.pretty else None))
    return 0 if result.validation.ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batter",
        description=(
            "Pancake Engine 0.3 — correctness-first, not TS parity. "
            "Known TS divergences are documented in "
            "pancake-production/docs/research/pancake-engine-0.3-ts-divergences.md."
        ),
    )
    parser.add_argument("--version", action="version", version=f"{ENGINE} {ENGINE_VERSION}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # hash
    p_hash = sub.add_parser("hash", help="Compute canonical hash of an EvidenceDataset or EvidenceSpec")
    g = p_hash.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", help="Path to EvidenceDataset JSON file")
    g.add_argument("--spec", help="Path to EvidenceSpec JSON file")
    p_hash.set_defaults(func=cmd_hash)

    # validate
    p_val = sub.add_parser("validate", help="Validate an EvidenceSpec against an EvidenceDataset")
    p_val.add_argument("--spec", required=True, help="Path to EvidenceSpec JSON file")
    p_val.add_argument("--dataset", required=True, help="Path to EvidenceDataset JSON file")
    p_val.set_defaults(func=cmd_validate)

    # run
    p_run = sub.add_parser("run", help="Run a backtest")
    p_run.add_argument("--spec", required=True, help="Path to EvidenceSpec JSON file")
    p_run.add_argument("--dataset", required=True, help="Path to EvidenceDataset JSON file")
    p_run.add_argument("--out", help="Path to write the result JSON (default: stdout)")
    p_run.add_argument("--observation-time", type=int, default=None,
                       dest="observation_time",
                       help="Unix seconds; if omitted, derived from dataset (errors if unresolved rows)")
    p_run.add_argument("--pretty", action="store_true", help="Indent JSON output")
    p_run.set_defaults(func=cmd_run)

    # walkforward
    p_wf = sub.add_parser("walkforward", help="Run a frozen-spec walk-forward evaluation")
    p_wf.add_argument("--spec", required=True, help="Path to EvidenceSpec JSON file")
    p_wf.add_argument("--dataset", required=True, help="Path to EvidenceDataset JSON file")
    p_wf.add_argument("--window-type", required=True, choices=["expanding", "rolling", "anchored"],
                      dest="window_type")
    p_wf.add_argument("--test-horizon", required=True, dest="test_horizon",
                      help="Integer seconds or anchor (MS / 3MS / QS / 2QS)")
    p_wf.add_argument("--step", required=True, help="Integer seconds or anchor")
    p_wf.add_argument("--train-horizon", default=None, dest="train_horizon",
                      help="Required for window_type=rolling; integer seconds or anchor")
    p_wf.add_argument("--min-fold-count", type=int, default=3, dest="min_fold_count")
    p_wf.add_argument("--resolution-policy", default="allow_overhang",
                      choices=["allow_overhang", "skip_overhang", "truncate_at_window_end"],
                      dest="resolution_policy")
    p_wf.add_argument("--observation-time", type=int, default=None, dest="observation_time")
    p_wf.add_argument("--out", help="Path to write the result JSON (default: stdout)")
    p_wf.add_argument("--pretty", action="store_true")
    p_wf.set_defaults(func=cmd_walkforward)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
