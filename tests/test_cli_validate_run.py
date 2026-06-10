"""CLI smoke tests: validate + run subcommands."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


SCHEMA_COLS = [
    {"name": "mkt", "type": "string", "semantic_role": "market_link"},
    {"name": "dec_ts", "type": "int", "semantic_role": "decision_time"},
    {"name": "res_ts", "type": "int", "semantic_role": "resolution_time"},
    {"name": "price", "type": "number", "semantic_role": "entry_price", "range": [0, 1]},
    {"name": "outcome", "type": "int", "semantic_role": "resolved_outcome_numeric"},
    {"name": "alpha", "type": "number", "semantic_role": "feature"},
    {"name": "target", "type": "int", "semantic_role": "feature"},
]


def _spec() -> dict[str, Any]:
    return {
        "spec_family": "pancake-evidence-spec",
        "spec_version": "0.1",
        "name": "cli-test",
        "schema_requirements": {"required_columns": SCHEMA_COLS},
        "strategy": {
            "side": "YES",
            "entry": {"when": {"feature": "alpha", "gte": 2.0}},
            "yes_payoff": {"when": {"feature_equal": {"a": "target", "b": "outcome"}}},
            "sizing": {"mode": "fixed_fraction", "value": 0.1},
        },
        "costs": {"slippage_bps": 0, "fee_bps": 0},
        "starting_capital": 1000.0,
    }


def _dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "ds_cli",
        "schema": {"columns": SCHEMA_COLS},
        "schema_sha256": "0" * 64,
        "storage_mode": "inline",
        "rows_inline": rows,
        "rows_sha256": "0" * 64,
        "row_count": len(rows),
    }


def test_cli_validate_ok(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    dataset_path = tmp_path / "dataset.json"
    _write_json(spec_path, _spec())
    _write_json(dataset_path, _dataset([
        {"mkt": "m/A", "dec_ts": 100, "res_ts": 200, "price": 0.5,
         "outcome": 1, "alpha": 3.0, "target": 1},
    ]))
    proc = subprocess.run(
        [sys.executable, "-m", "pancake_engine.cli", "validate",
         "--spec", str(spec_path), "--dataset", str(dataset_path)],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["ok"] is True


def test_cli_validate_blocks_on_range(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    dataset_path = tmp_path / "dataset.json"
    _write_json(spec_path, _spec())
    _write_json(dataset_path, _dataset([
        {"mkt": "m/A", "dec_ts": 100, "res_ts": 200, "price": 1.5,
         "outcome": 1, "alpha": 3.0, "target": 1},  # price > 1
    ]))
    proc = subprocess.run(
        [sys.executable, "-m", "pancake_engine.cli", "validate",
         "--spec", str(spec_path), "--dataset", str(dataset_path)],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 1
    data = json.loads(proc.stdout)
    assert data["ok"] is False
    codes = {e["code"] for e in data["errors"]}
    assert "E_EVIDENCE_RANGE" in codes


def test_cli_run_writes_result(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    dataset_path = tmp_path / "dataset.json"
    out_path = tmp_path / "result.json"
    _write_json(spec_path, _spec())
    _write_json(dataset_path, _dataset([
        {"mkt": "m/A", "dec_ts": 100, "res_ts": 200, "price": 0.5,
         "outcome": 1, "alpha": 3.0, "target": 1},
    ]))
    proc = subprocess.run(
        [sys.executable, "-m", "pancake_engine.cli", "run",
         "--spec", str(spec_path), "--dataset", str(dataset_path),
         "--out", str(out_path), "--observation-time", "300"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert out_path.exists()
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert result["engine"] == "batter"
    assert result["engine_version"] == "0.8.0"
    assert result["engine_mode"] == "event_time_v1"
    assert result["hashes"]["result_hash"] != ""
    assert result["metrics"]["standard"]["num_trades"] == 1
