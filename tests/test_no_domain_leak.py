"""No-domain-leak guard: ``pancake_engine/`` must not reference example-specific tokens.

The engine package is domain-agnostic. Examples live outside it. This grep guard
asserts that no example-specific token has crept into engine source.

If a legitimate use surfaces (e.g., a generic "weather" mention in a comment),
add it to the ALLOWED carve-out below with a clear rationale.
"""

from __future__ import annotations

import re
from pathlib import Path

ENGINE_DIR = Path(__file__).parent.parent / "pancake_engine"

# Tokens that may appear in examples/ but never in pancake_engine/
FORBIDDEN_TOKENS = [
    "jakarta",
    "rapture",
    "jesus",
    "polymarket",
    "dengue",
    "temperature",
    "weather",
    "fed_cut",
    "fed cut",
]

# Substring carve-outs to avoid false positives (be conservative).
# Format: (forbidden_token, escape_substring). Lines containing any
# escape_substring are skipped when checking that forbidden_token.
ALLOWED_CONTEXTS: list[tuple[str, str]] = [
    # None currently. Add with rationale if needed.
]


def test_no_domain_leak_in_engine() -> None:
    failures: list[str] = []
    for py_file in ENGINE_DIR.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            lowered = line.lower()
            for token in FORBIDDEN_TOKENS:
                if token in lowered:
                    # Check carve-outs
                    if any(t == token and esc in lowered for t, esc in ALLOWED_CONTEXTS):
                        continue
                    rel = py_file.relative_to(ENGINE_DIR.parent)
                    failures.append(f"{rel}:{line_no}: contains forbidden token {token!r}\n    {line.strip()}")
    assert not failures, (
        "Engine package contains domain-specific tokens. Move to examples/ or add a carve-out:\n"
        + "\n".join(failures)
    )


def test_examples_dir_exists() -> None:
    """Sanity: examples/ exists with the expected 4 subdirs."""
    examples = Path(__file__).parent.parent / "examples"
    assert examples.is_dir()
    for name in ("toy", "jakarta_temperature", "rapture_family", "btc_pred_hedge"):
        assert (examples / name).is_dir(), f"examples/{name} missing"
        assert (examples / name / "dataset.json").exists()
        assert (examples / name / "spec.json").exists()
        assert (examples / name / "expected_result.json").exists()
        assert (examples / name / "run.py").exists()
        assert (examples / name / "regen.py").exists()
