"""Shared test fixture paths."""

from pathlib import Path

TESTS_ROOT = Path(__file__).parent
FIXTURES = TESTS_ROOT / "fixtures"
CANONICAL_FIXTURES = FIXTURES / "canonical"
