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
from .config import BacktestConfig
from .hash import sha256_canonical
from .io.dump import dump_result
from .io.load import load_dataset, load_json, load_spec
from .runner import run_backtest
from .validate import validate_dataset, validate_spec

__all__ = ["main", "build_parser"]


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pancake-engine",
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

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
