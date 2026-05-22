"""Examples smoke: each ``examples/*/run.py`` must report ``result_hash`` byte-equal
to the committed ``expected_result.json``.

Examples live outside ``pancake_engine/`` and import only from the engine. If
the engine drifts, examples fail; examples cannot change engine behavior.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLES = ["toy", "jakarta_temperature", "rapture_family", "btc_pred_hedge"]


@pytest.mark.parametrize("name", EXAMPLES)
def test_example_runs_clean(name: str) -> None:
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "examples" / name / "run.py")],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, (
        f"example {name!r} failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "OK" in proc.stdout
