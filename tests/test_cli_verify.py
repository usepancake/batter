"""CLI tests: verify subcommand.

TDD — these tests drive the design of ``batter verify``.

Exit codes:
  0  verified (result_hash matched + dataset integrity confirmed)
  1  mismatch (result_hash or dataset integrity)
  2  input/validation error (malformed bundle, missing required fields)
  3  unverifiable (pointer dataset — rows not inline)

Bundle shapes accepted:
  (a) regen-style:   {spec, dataset, config?, expected_result_hash}
  (b) fixture-style: {spec, dataset, config?, expected: {result_hash, ...}}
"""

from __future__ import annotations

import http.server
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SCHEMA_COLS = [
    {"name": "mkt", "type": "string", "semantic_role": "market_link"},
    {"name": "dec_ts", "type": "int", "semantic_role": "decision_time"},
    {"name": "res_ts", "type": "int", "semantic_role": "resolution_time"},
    {"name": "price", "type": "number", "semantic_role": "entry_price", "range": [0, 1]},
    {"name": "outcome", "type": "int", "semantic_role": "resolved_outcome_numeric"},
    {"name": "alpha", "type": "number", "semantic_role": "feature"},
    {"name": "target", "type": "int", "semantic_role": "feature"},
]

ROWS = [
    {"mkt": "m/A", "dec_ts": 100, "res_ts": 200, "price": 0.5,
     "outcome": 1, "alpha": 3.0, "target": 1},
    {"mkt": "m/B", "dec_ts": 300, "res_ts": 400, "price": 0.4,
     "outcome": 0, "alpha": 2.5, "target": 0},
]


def _spec() -> dict[str, Any]:
    return {
        "spec_family": "pancake-evidence-spec",
        "spec_version": "0.1",
        "name": "verify-test",
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


def _run_verify(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pancake_engine.cli", "verify"] + args,
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )


def _build_inline_bundle(
    rows: list[dict[str, Any]],
    *,
    observation_time: int = 500,
    fixture_style: bool = False,
) -> dict[str, Any]:
    """Build a correct self-contained bundle from scratch by running the engine."""
    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT))
    from pancake_engine import BacktestConfig, EvidenceDataset, EvidenceSpec, run_backtest
    from pancake_engine.hash import sha256_canonical

    spec_dict = _spec()
    schema_cols = spec_dict["schema_requirements"]["required_columns"]
    schema_sha256 = sha256_canonical({"columns": schema_cols})
    rows_sha256 = sha256_canonical(rows)
    dataset_dict: dict[str, Any] = {
        "id": "ds_verify_test",
        "schema": {"columns": schema_cols},
        "schema_sha256": schema_sha256,
        "storage_mode": "inline",
        "rows_inline": rows,
        "rows_sha256": rows_sha256,
        "row_count": len(rows),
    }

    spec = EvidenceSpec.model_validate(spec_dict)
    dataset = EvidenceDataset.model_validate(dataset_dict)
    result = run_backtest(spec, dataset, BacktestConfig(observation_time=observation_time))

    if fixture_style:
        expected_block: Any = {
            "result_hash": result.result_hash,
            "num_trades": result.metrics.standard.num_trades,
        }
    else:
        expected_block = result.result_hash  # regen-style: top-level string

    bundle: dict[str, Any] = {
        "spec": spec_dict,
        "dataset": dataset_dict,
        "config": {"observation_time": observation_time},
    }
    if fixture_style:
        bundle["expected"] = expected_block
    else:
        bundle["expected_result_hash"] = expected_block

    return bundle


# ---------------------------------------------------------------------------
# Happy-path: regen-style bundle (exit 0)
# ---------------------------------------------------------------------------


def test_verify_regen_bundle_ok(tmp_path: Path) -> None:
    bundle = _build_inline_bundle(ROWS)
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    proc = _run_verify(["--bundle", str(bundle_path)])
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["verified"] is True
    assert out["expected"] == out["computed"]
    assert "engine_version" in out
    assert "num_trades" in out


# ---------------------------------------------------------------------------
# Happy-path: fixture-style bundle {expected: {result_hash, ...}} (exit 0)
# ---------------------------------------------------------------------------


def test_verify_fixture_bundle_ok(tmp_path: Path) -> None:
    bundle = _build_inline_bundle(ROWS, fixture_style=True)
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    proc = _run_verify(["--bundle", str(bundle_path)])
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["verified"] is True


# ---------------------------------------------------------------------------
# Hash mismatch: wrong expected hash → exit 1
# ---------------------------------------------------------------------------


def test_verify_hash_mismatch(tmp_path: Path) -> None:
    bundle = _build_inline_bundle(ROWS)
    bundle["expected_result_hash"] = "a" * 64  # wrong hash
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    proc = _run_verify(["--bundle", str(bundle_path)])
    assert proc.returncode == 1, proc.stderr
    out = json.loads(proc.stdout)
    assert out["verified"] is False
    assert out["expected"] != out["computed"]


# ---------------------------------------------------------------------------
# Tamper detection: mutate one row → dataset integrity fails → exit 1
# ---------------------------------------------------------------------------


def test_verify_tamper_detection(tmp_path: Path) -> None:
    bundle = _build_inline_bundle(ROWS)
    # Tamper: change a price in rows AFTER hashes were computed
    bundle["dataset"]["rows_inline"][0]["price"] = 0.99
    # expected_result_hash still points to the original run — but the dataset
    # integrity check must fire BEFORE we even re-run.
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    proc = _run_verify(["--bundle", str(bundle_path)])
    assert proc.returncode == 1, proc.stderr
    out = json.loads(proc.stdout)
    assert out["verified"] is False
    # stderr must mention dataset integrity
    assert "dataset" in proc.stderr.lower() or "tamper" in proc.stderr.lower() or "differ" in proc.stderr.lower()


# ---------------------------------------------------------------------------
# Pointer dataset → exit 3
# ---------------------------------------------------------------------------


def test_verify_pointer_dataset_exit3(tmp_path: Path) -> None:
    bundle = _build_inline_bundle(ROWS)
    # Switch storage_mode to pointer and remove rows_inline
    bundle["dataset"]["storage_mode"] = "pointer"
    bundle["dataset"].pop("rows_inline", None)
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    proc = _run_verify(["--bundle", str(bundle_path)])
    assert proc.returncode == 3, proc.stderr
    assert "rows" in proc.stderr.lower() or "inline" in proc.stderr.lower() or "license" in proc.stderr.lower()


# ---------------------------------------------------------------------------
# Input error: missing expected hash → exit 2
# ---------------------------------------------------------------------------


def test_verify_missing_expected_hash_exit2(tmp_path: Path) -> None:
    bundle = _build_inline_bundle(ROWS)
    del bundle["expected_result_hash"]
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    proc = _run_verify(["--bundle", str(bundle_path)])
    assert proc.returncode == 2, proc.stderr


# ---------------------------------------------------------------------------
# Input error: malformed JSON → exit 2
# ---------------------------------------------------------------------------


def test_verify_malformed_json_exit2(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text("{not valid json", encoding="utf-8")

    proc = _run_verify(["--bundle", str(bundle_path)])
    assert proc.returncode == 2, proc.stderr


# ---------------------------------------------------------------------------
# Engine version mismatch warning path
# ---------------------------------------------------------------------------


def test_verify_engine_version_mismatch_warning(tmp_path: Path) -> None:
    bundle = _build_inline_bundle(ROWS)
    # Declare an old engine_version that won't match current ENGINE_VERSION
    bundle["engine_version"] = "0.1.0"
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    proc = _run_verify(["--bundle", str(bundle_path)])
    # Attempt is still made; result depends on actual hash match/mismatch,
    # but the JSON output must signal version_mismatch
    out = json.loads(proc.stdout)
    assert out.get("version_mismatch") is True
    # stderr must warn about version difference
    assert "engine_version" in proc.stderr.lower() or "version" in proc.stderr.lower()


# ---------------------------------------------------------------------------
# URL mode: serve bundle from a local HTTP server → exit 0
# ---------------------------------------------------------------------------


def test_verify_url_mode_ok(tmp_path: Path) -> None:
    bundle = _build_inline_bundle(ROWS)
    bundle_bytes = json.dumps(bundle).encode("utf-8")

    # Find a free port
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(bundle_bytes)))
            self.end_headers()
            self.wfile.write(bundle_bytes)

        def log_message(self, *args: Any) -> None:  # suppress output
            pass

    server = http.server.HTTPServer(("127.0.0.1", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        proc = _run_verify(["--url", f"http://127.0.0.1:{port}/bundle.json"])
        assert proc.returncode == 0, proc.stderr
        out = json.loads(proc.stdout)
        assert out["verified"] is True
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# URL mode: pointer dataset from URL → exit 3
# ---------------------------------------------------------------------------


def test_verify_url_pointer_exit3(tmp_path: Path) -> None:
    bundle = _build_inline_bundle(ROWS)
    bundle["dataset"]["storage_mode"] = "pointer"
    bundle["dataset"].pop("rows_inline", None)
    bundle_bytes = json.dumps(bundle).encode("utf-8")

    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(bundle_bytes)))
            self.end_headers()
            self.wfile.write(bundle_bytes)

        def log_message(self, *args: Any) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        proc = _run_verify(["--url", f"http://127.0.0.1:{port}/bundle.json"])
        assert proc.returncode == 3, proc.stderr
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Bundle without explicit config (observation_time derived from dataset)
# ---------------------------------------------------------------------------


def test_verify_no_config_observation_time_derived(tmp_path: Path) -> None:
    """Bundle with no config block; observation_time derived from max(res_ts)."""
    bundle = _build_inline_bundle(ROWS)
    del bundle["config"]
    # Recompute the correct hash without config (observation_time=None → derived)
    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT))
    from pancake_engine import BacktestConfig, EvidenceDataset, EvidenceSpec, run_backtest

    spec_dict = bundle["spec"]
    dataset_dict = bundle["dataset"]
    spec = EvidenceSpec.model_validate(spec_dict)
    dataset = EvidenceDataset.model_validate(dataset_dict)
    result = run_backtest(spec, dataset, BacktestConfig(observation_time=None))
    bundle["expected_result_hash"] = result.result_hash

    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    proc = _run_verify(["--bundle", str(bundle_path)])
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["verified"] is True
