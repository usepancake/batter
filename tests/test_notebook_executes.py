"""Notebook execution gate.

The notebook is a tour, not a spec. It must execute cleanly from top to bottom
without raising. Outputs are NOT diffed against committed outputs (notebook
output metadata is non-deterministic), but successful execution is required.

If any of the prior tests fail (examples + walkforward + parity + no-domain-leak),
this notebook will likely fail too — by design.
"""

from __future__ import annotations

from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "walkforward_tour.ipynb"


@pytest.mark.skipif(not NOTEBOOK.exists(), reason="notebook not yet added")
def test_walkforward_tour_notebook_executes() -> None:
    nbformat = pytest.importorskip("nbformat")
    NotebookClient = pytest.importorskip("nbclient").NotebookClient

    nb = nbformat.read(str(NOTEBOOK), as_version=4)
    client = NotebookClient(nb, timeout=60, kernel_name="python3",
                             resources={"metadata": {"path": str(NOTEBOOK.parent)}})
    client.execute()
