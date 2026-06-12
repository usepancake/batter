"""Guard: the package version is declared in TWO places and must agree.

pyproject.toml's static ``[project].version`` is what the build backend
stamps into the wheel; ``pancake_engine.__version__.__version__`` is what
the runtime (and the verify CLI's ``package_version`` field) reports.

The 0.10.3 release initially failed to publish because only
``__version__.py`` was bumped — the workflow rebuilt batter-0.10.2 and
PyPI 400'd on file reuse. This test makes that drift a CI failure
instead of a release-day surprise.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pancake_engine.__version__ import __version__


def test_pyproject_version_matches_package_version() -> None:
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    assert data["project"]["version"] == __version__, (
        "pyproject.toml [project].version and pancake_engine.__version__ "
        "have drifted — bump BOTH in any release prep commit"
    )
