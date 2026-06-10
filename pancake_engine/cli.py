"""Pancake Engine 0.3 CLI.

PR-1 subcommands: ``hash`` (PR-0), ``validate``, ``run``.
0.9.0 adds: ``verify``.

Engine 0.3 is correctness-first, not TS parity. Known TS divergences are documented
in docs/math-audit-0.4.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Optional

from .__version__ import ENGINE, ENGINE_VERSION
from .config import BacktestConfig, WalkforwardConfig
from .hash import sha256_canonical
from .io.dump import dump_result
from .io.load import load_dataset, load_json, load_spec, parse_json
from .runner import run_backtest
from .types import EvidenceDataset, EvidenceSpec
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


def _extract_expected_hash(bundle: dict) -> "str | None":
    """Sniff both bundle shapes and return the expected result_hash string, or None."""
    # regen-style: top-level string
    if isinstance(bundle.get("expected_result_hash"), str):
        return bundle["expected_result_hash"]
    # fixture-style: {expected: {result_hash: ...}}
    expected = bundle.get("expected")
    if isinstance(expected, dict) and isinstance(expected.get("result_hash"), str):
        return expected["result_hash"]
    return None


def _sniff_engine_version(bundle: dict) -> "str | None":
    """Return a declared engine_version from common bundle locations, or None."""
    # Top level
    if isinstance(bundle.get("engine_version"), str):
        return bundle["engine_version"]
    # Under _fixture_meta
    meta = bundle.get("_fixture_meta")
    if isinstance(meta, dict) and isinstance(meta.get("engine_version"), str):
        return meta["engine_version"]
    # Under expected block
    expected = bundle.get("expected")
    if isinstance(expected, dict) and isinstance(expected.get("engine_version"), str):
        return expected["engine_version"]
    return None


def _dataset_schema_dict_from_dataset(dataset: EvidenceDataset) -> dict:
    """Round-trip the dataset schema through pydantic to get a canonical-shape dict."""
    return dataset.dataset_schema.model_dump(exclude_none=True, mode="python")


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify a self-contained backtest bundle.

    Exit codes:
      0  verified (result_hash matched + dataset integrity confirmed)
      1  mismatch (result_hash or dataset integrity)
      2  input/validation error (malformed bundle, missing required fields)
      3  unverifiable (pointer dataset — rows not inline)
    """
    # ------------------------------------------------------------------ load
    if args.bundle:
        try:
            raw = load_json(args.bundle)
        except Exception as exc:
            print(f"error: could not load bundle: {exc}", file=sys.stderr)
            return 2
    else:
        # URL mode — stdlib urllib only
        timeout = getattr(args, "timeout", 30) or 30
        try:
            req = urllib.request.Request(
                args.url,
                headers={"User-Agent": f"batter/{ENGINE_VERSION}"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode("utf-8")
            raw = parse_json(text)
        except urllib.error.URLError as exc:
            print(f"error: could not fetch URL: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"error: could not load bundle from URL: {exc}", file=sys.stderr)
            return 2

    if not isinstance(raw, dict):
        print("error: bundle must be a JSON object", file=sys.stderr)
        return 2

    # ------------------------------------------------------------------ sniff shape
    spec_raw = raw.get("spec")
    dataset_raw = raw.get("dataset")
    config_raw = raw.get("config")

    if not isinstance(spec_raw, dict):
        print("error: bundle missing 'spec' object", file=sys.stderr)
        return 2
    if not isinstance(dataset_raw, dict):
        print("error: bundle missing 'dataset' object", file=sys.stderr)
        return 2

    expected_hash = _extract_expected_hash(raw)
    if expected_hash is None:
        print(
            "error: bundle missing expected_result_hash (regen-style) "
            "or expected.result_hash (fixture-style)",
            file=sys.stderr,
        )
        return 2

    # ------------------------------------------------------------------ pointer guard
    storage_mode = dataset_raw.get("storage_mode")
    rows_inline = dataset_raw.get("rows_inline")
    if storage_mode == "pointer" or not isinstance(rows_inline, list):
        print(
            "error: bundle does not carry rows "
            "(license-gated replay bundle required)",
            file=sys.stderr,
        )
        return 3

    # ------------------------------------------------------------------ parse types
    try:
        spec = EvidenceSpec.model_validate(spec_raw)
    except Exception as exc:
        print(f"error: invalid spec: {exc}", file=sys.stderr)
        return 2

    try:
        dataset = EvidenceDataset.model_validate(dataset_raw)
    except Exception as exc:
        print(f"error: invalid dataset: {exc}", file=sys.stderr)
        return 2

    # ------------------------------------------------------------------ dataset integrity
    # Recompute schema_sha256 and rows_sha256 over the bundle's actual data.
    # A tampered bundle will show declared != recomputed here.
    schema_dict = _dataset_schema_dict_from_dataset(dataset)
    recomputed_schema_sha256 = sha256_canonical(schema_dict)
    recomputed_rows_sha256 = sha256_canonical(rows_inline)

    declared_schema_sha256 = dataset_raw.get("schema_sha256", "")
    declared_rows_sha256 = dataset_raw.get("rows_sha256", "")

    schema_ok = (recomputed_schema_sha256 == declared_schema_sha256)
    rows_ok = (recomputed_rows_sha256 == declared_rows_sha256)

    if not schema_ok or not rows_ok:
        if not schema_ok:
            print(
                f"error: schema bytes differ from declaration "
                f"(declared={declared_schema_sha256[:16]}… "
                f"recomputed={recomputed_schema_sha256[:16]}…) — "
                "dataset may have been tampered with",
                file=sys.stderr,
            )
        if not rows_ok:
            print(
                f"error: rows bytes differ from declaration "
                f"(declared={declared_rows_sha256[:16]}… "
                f"recomputed={recomputed_rows_sha256[:16]}…) — "
                "dataset may have been tampered with",
                file=sys.stderr,
            )
        out = {
            "verified": False,
            "expected": expected_hash,
            "computed": None,
            "engine_version": ENGINE_VERSION,
            "num_trades": None,
            "integrity_error": True,
        }
        print(json.dumps(out))
        return 1

    # ------------------------------------------------------------------ engine version guard
    declared_engine_version = _sniff_engine_version(raw)
    version_mismatch = (
        declared_engine_version is not None
        and declared_engine_version != ENGINE_VERSION
    )
    if version_mismatch:
        print(
            f"warning: bundle declares engine_version={declared_engine_version!r} "
            f"but current engine is {ENGINE_VERSION!r}; "
            "result_hash values are only comparable under the same ENGINE_VERSION — "
            "attempting re-run anyway",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------ re-run
    observation_time: "int | None" = None
    if isinstance(config_raw, dict):
        ot = config_raw.get("observation_time")
        if isinstance(ot, int):
            observation_time = ot

    config = BacktestConfig(observation_time=observation_time)
    try:
        result = run_backtest(spec, dataset, config)
    except Exception as exc:
        print(f"error: engine run failed: {exc}", file=sys.stderr)
        return 3

    computed_hash = result.result_hash
    verified = (computed_hash == expected_hash)

    # ------------------------------------------------------------------ output
    human_verdict = "VERIFIED" if verified else "MISMATCH"
    print(
        f"{human_verdict}  expected={expected_hash[:16]}…  "
        f"computed={computed_hash[:16]}…  "
        f"engine={ENGINE_VERSION}  trades={result.metrics.standard.num_trades}",
        file=sys.stderr,
    )

    out: dict = {
        "verified": verified,
        "expected": expected_hash,
        "computed": computed_hash,
        "engine_version": ENGINE_VERSION,
        "num_trades": result.metrics.standard.num_trades,
    }
    if version_mismatch:
        out["version_mismatch"] = True

    print(json.dumps(out))
    return 0 if verified else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batter",
        description=(
            "Pancake Engine 0.3 — correctness-first, not TS parity. "
            "Known TS divergences are documented in "
            "docs/math-audit-0.4.md."
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

    # verify
    p_ver = sub.add_parser(
        "verify",
        help="Verify a self-contained backtest bundle (local file or remote URL)",
    )
    g_ver = p_ver.add_mutually_exclusive_group(required=True)
    g_ver.add_argument(
        "--bundle",
        metavar="PATH",
        help="Path to a self-contained bundle JSON file",
    )
    g_ver.add_argument(
        "--url",
        metavar="URL",
        help="URL of a self-contained bundle JSON (stdlib urllib; no new deps)",
    )
    p_ver.add_argument(
        "--timeout",
        type=int,
        default=30,
        metavar="SECONDS",
        help="HTTP timeout in seconds for --url mode (default: 30)",
    )
    p_ver.set_defaults(func=cmd_verify)

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
