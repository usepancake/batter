"""Pancake Engine 0.3 CLI.

PR-0 ships only the ``hash`` subcommand. ``validate`` and ``run`` land in PR-1;
``walkforward`` in PR-2.

Engine 0.3 is correctness-first, not TS parity. Known TS divergences are documented
in pancake-production/docs/research/pancake-engine-0.3-ts-divergences.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .__version__ import ENGINE, ENGINE_VERSION
from .hash import sha256_canonical
from .io.load import load_json

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pancake-engine",
        description=(
            "Pancake Engine 0.3 — correctness-first, not TS parity. "
            "Known TS divergences are documented in "
            "pancake-production/docs/research/pancake-engine-0.3-ts-divergences.md."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"{ENGINE} {ENGINE_VERSION}"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_hash = sub.add_parser(
        "hash",
        help="Compute canonical hash of an EvidenceDataset or EvidenceSpec JSON file",
        description=(
            "Compute the canonical SHA-256 hash(es) of an EvidenceDataset or "
            "EvidenceSpec JSON file. For datasets, emits both schema_sha256 and "
            "rows_sha256 (byte-equal to the TS evidence-runner hashes for the same "
            "content). For specs, emits source_spec_hash."
        ),
    )
    g = p_hash.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", help="Path to EvidenceDataset JSON file")
    g.add_argument("--spec", help="Path to EvidenceSpec JSON file")
    p_hash.set_defaults(func=cmd_hash)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
