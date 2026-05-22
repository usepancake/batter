"""TS hash parity tests.

Engine 0.3 is correctness-first, not TS parity — but for the canonicalization
substrate (the bytes that get hashed), Python ``sha256_canonical`` must
byte-equal the TS evidence-runner's ``hashSchema()`` / ``hashRows()``.

This test asserts that against fixtures whose **expected hashes were computed
by the real TS code in pancake-production** (via
``tests/fixtures/canonical/ts_hash_oracle.mjs``). If `ts_hashes.json` ever
needs to be regenerated, run that script with a sibling pancake-production
checkout. The committed `ts_hashes.json` lives in this repo; the test never
imports anything from pancake-production.

If this test fails, one of:
- Python canonicalize has drifted from V8 JSON.stringify
- TS canonicalize in pancake-production has drifted (regenerate oracle)
- A new test fixture was added without re-running the oracle
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pancake_engine.hash import sha256_canonical

CANONICAL_FIXTURES = Path(__file__).parent / "fixtures" / "canonical"
TS_HASHES_PATH = CANONICAL_FIXTURES / "ts_hashes.json"


def _load_fixtures() -> list[dict]:
    if not TS_HASHES_PATH.exists():
        pytest.skip(
            f"{TS_HASHES_PATH.name} missing; regenerate with "
            "`node tests/fixtures/canonical/ts_hash_oracle.mjs` "
            "from a sibling pancake-production checkout."
        )
    return json.loads(TS_HASHES_PATH.read_text(encoding="utf-8"))


def test_ts_hash_parity_schema() -> None:
    """Python `sha256_canonical(schema)` matches TS `hashSchema(schema)` byte-for-byte."""
    fixtures = _load_fixtures()
    failures: list[str] = []
    for f in fixtures:
        py_hash = sha256_canonical(f["schema"])
        if py_hash != f["schema_sha256"]:
            failures.append(
                f"{f['name']}: schema_sha256 mismatch\n"
                f"  TS:  {f['schema_sha256']}\n"
                f"  Py:  {py_hash}"
            )
    assert not failures, "\n".join(["", *failures])


def test_ts_hash_parity_rows() -> None:
    """Python `sha256_canonical(rows)` matches TS `hashRows(rows)` byte-for-byte."""
    fixtures = _load_fixtures()
    failures: list[str] = []
    for f in fixtures:
        py_hash = sha256_canonical(f["rows"])
        if py_hash != f["rows_sha256"]:
            failures.append(
                f"{f['name']}: rows_sha256 mismatch\n"
                f"  TS:  {f['rows_sha256']}\n"
                f"  Py:  {py_hash}"
            )
    assert not failures, "\n".join(["", *failures])


def test_ts_hashes_format_sanity() -> None:
    """Every committed hash is lowercase 64-hex (SHA-256)."""
    fixtures = _load_fixtures()
    for f in fixtures:
        for k in ("schema_sha256", "rows_sha256"):
            h = f[k]
            assert len(h) == 64, f"{f['name']}.{k}: expected 64 chars, got {len(h)}"
            assert all(c in "0123456789abcdef" for c in h), (
                f"{f['name']}.{k}: non-hex / uppercase chars in {h!r}"
            )


def test_ts_fixtures_count() -> None:
    """5 fixtures committed; if you add/remove, update this count and regenerate the oracle."""
    fixtures = _load_fixtures()
    assert len(fixtures) == 5, f"expected 5 TS-hash fixtures, got {len(fixtures)}"
