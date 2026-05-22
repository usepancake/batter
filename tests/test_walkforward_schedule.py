"""Fold schedule generation: expanding / rolling / anchored."""

from __future__ import annotations

import pytest

from pancake_engine import WalkforwardConfig
from pancake_engine.validate import validate_dataset
from pancake_engine.walkforward.schedule import build_fold_schedule
from pancake_engine.walkforward.window import parse_anchor

from ._runner_helpers import make_spec
from ._wf_helpers import make_wf_dataset, utc_ts


def _role_lookup(dataset, spec):
    verdict, lookup = validate_dataset(dataset, spec)
    assert verdict.ok, [e.code for e in verdict.errors]
    return lookup


# --- parse_anchor ---


def test_parse_anchor_seconds() -> None:
    assert parse_anchor(86_400) == (86_400, "sec")


def test_parse_anchor_ms() -> None:
    assert parse_anchor("MS") == (1, "MS")
    assert parse_anchor("3MS") == (3, "MS")


def test_parse_anchor_qs() -> None:
    assert parse_anchor("QS") == (1, "QS")
    assert parse_anchor("2QS") == (2, "QS")


def test_parse_anchor_invalid() -> None:
    with pytest.raises(ValueError, match="E_WALKFORWARD_CONFIG"):
        parse_anchor("WS")
    with pytest.raises(ValueError, match="E_WALKFORWARD_CONFIG"):
        parse_anchor("YS")
    with pytest.raises(ValueError, match="E_WALKFORWARD_CONFIG"):
        parse_anchor(-1)
    with pytest.raises(ValueError, match="E_WALKFORWARD_CONFIG"):
        parse_anchor("3WS")


# --- expanding ---


def test_expanding_schedule_simple() -> None:
    """3 folds, step = test_horizon = 30 days, no overlap."""
    spec = make_spec()
    DAY = 86_400
    dataset = make_wf_dataset([
        (i * DAY, i * DAY + 100, {}) for i in range(90)
    ])
    lookup = _role_lookup(dataset, spec)
    config = WalkforwardConfig(
        window_type="expanding", test_horizon=30 * DAY, step=30 * DAY,
    )
    schedule = build_fold_schedule(dataset, config, lookup)
    assert len(schedule) == 3
    # Folds: [0, 30d), [30d, 60d), [60d, 90d)
    assert schedule[0].test_window == (0, 30 * DAY)
    assert schedule[1].test_window == (30 * DAY, 60 * DAY)
    assert schedule[2].test_window == (60 * DAY, 90 * DAY)
    # Expanding train: all starts at 0
    for f in schedule:
        assert f.train_window[0] == 0
    assert schedule[0].train_window == (0, 0)


def test_rolling_schedule_simple() -> None:
    """3 folds, rolling 30-day train + 30-day test, step 30d."""
    spec = make_spec()
    DAY = 86_400
    dataset = make_wf_dataset([
        (i * DAY, i * DAY + 100, {}) for i in range(120)
    ])
    lookup = _role_lookup(dataset, spec)
    config = WalkforwardConfig(
        window_type="rolling", test_horizon=30 * DAY, step=30 * DAY,
        train_horizon=30 * DAY,
    )
    schedule = build_fold_schedule(dataset, config, lookup)
    assert len(schedule) >= 3
    # First fold: train [0, 30d), test [30d, 60d)
    assert schedule[0].train_window == (0, 30 * DAY)
    assert schedule[0].test_window == (30 * DAY, 60 * DAY)
    # Rolling: train shifts forward
    assert schedule[1].train_window == (30 * DAY, 60 * DAY)
    assert schedule[1].test_window == (60 * DAY, 90 * DAY)


def test_anchored_monthly_schedule() -> None:
    """Anchored MS step + 1MS horizon over 6 months."""
    spec = make_spec()
    rows = [
        (utc_ts(2024, m, d), utc_ts(2024, m, d) + 3600, {})
        for m in range(1, 7) for d in (5, 15, 25)
    ]
    dataset = make_wf_dataset(rows)
    lookup = _role_lookup(dataset, spec)
    config = WalkforwardConfig(
        window_type="anchored", test_horizon="MS", step="MS",
    )
    schedule = build_fold_schedule(dataset, config, lookup)
    assert len(schedule) >= 5
    # First test_window: snap to next anchor at or after the first row time.
    # First row is Jan 5; next MS = Feb 1. Test = [Feb 1, Mar 1).
    assert schedule[0].test_window == (utc_ts(2024, 2, 1), utc_ts(2024, 3, 1))
    assert schedule[1].test_window == (utc_ts(2024, 3, 1), utc_ts(2024, 4, 1))


def test_anchored_quarterly_schedule() -> None:
    """Anchored QS step + 1QS horizon over 1.5 years."""
    spec = make_spec()
    rows = [
        (utc_ts(2024, m, 15), utc_ts(2024, m, 15) + 3600, {})
        for m in range(1, 13)
    ]
    dataset = make_wf_dataset(rows)
    lookup = _role_lookup(dataset, spec)
    config = WalkforwardConfig(
        window_type="anchored", test_horizon="QS", step="QS",
    )
    schedule = build_fold_schedule(dataset, config, lookup)
    assert len(schedule) >= 3
    # First row Jan 15 → next QS = Apr 1
    assert schedule[0].test_window == (utc_ts(2024, 4, 1), utc_ts(2024, 7, 1))
    assert schedule[1].test_window == (utc_ts(2024, 7, 1), utc_ts(2024, 10, 1))


def test_anchored_requires_calendar_units() -> None:
    spec = make_spec()
    dataset = make_wf_dataset([(0, 100, {})])
    lookup = _role_lookup(dataset, spec)
    config = WalkforwardConfig(
        window_type="anchored", test_horizon=86_400, step=86_400,
    )
    with pytest.raises(ValueError, match="E_WALKFORWARD_CONFIG"):
        build_fold_schedule(dataset, config, lookup)


def test_rolling_requires_train_horizon() -> None:
    spec = make_spec()
    dataset = make_wf_dataset([(0, 100, {})])
    lookup = _role_lookup(dataset, spec)
    config = WalkforwardConfig(
        window_type="rolling", test_horizon=100, step=100,
        train_horizon=None,
    )
    with pytest.raises(ValueError, match="E_WALKFORWARD_CONFIG"):
        build_fold_schedule(dataset, config, lookup)
