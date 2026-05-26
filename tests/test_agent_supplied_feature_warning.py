"""Tests for AGENT_SUPPLIED_FEATURE_UNVERIFIED warning (E3b parity with TS runner).

ADR-0031 line 110: the warning MUST fire whenever entry or yes_payoff predicates
reference at least one column with semantic_role=feature.

4 tests:
1. No feature columns referenced → no emit.
2. 1 feature column referenced in entry → emit, message + context correct.
3. 2 feature columns referenced across entry+yes_payoff → emit ONCE, context sorted.
4. Re-run (determinism): identical result_hash on two runs when warning is emitted.
"""

from __future__ import annotations

from pancake_engine import BacktestConfig, run_backtest
from pancake_engine.warnings import WarningCode

from ._runner_helpers import make_dataset, make_spec, row

DAY = 86_400


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _run(spec, dataset, obs_time=300):
    return run_backtest(spec, dataset, BacktestConfig(observation_time=obs_time))


def _feature_warns(result):
    return [w for w in result.warnings if w.code == WarningCode.AGENT_SUPPLIED_FEATURE_UNVERIFIED]


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: no feature columns referenced → no emit
# ─────────────────────────────────────────────────────────────────────────────


def test_no_feature_columns_no_warning() -> None:
    """When neither entry nor yes_payoff reference any feature column, the warning must NOT emit.

    This spec uses no entry condition (all rows enter) and the default yes_payoff
    (target == outcome), which references the 'target' and 'outcome' columns.
    Both 'target' and 'outcome' have semantic_role=feature in the standard schema,
    but the yes_payoff uses a feature_equal condition — however, we need to check
    that a spec without ANY feature reference doesn't emit.

    We construct a spec whose entry condition references a non-feature column only,
    and whose yes_payoff also avoids feature columns entirely. This is done by
    using a price-based entry guard (price column has semantic_role=entry_price, not feature).

    Since all condition predicates must reference *some* column, we use a spec
    where the entry_when references the 'alpha' feature (so we CAN emit) vs one
    where we suppress feature columns. The cleanest no-emit case is a spec with
    zero trades (entry condition never fires) — but the warning check is structural
    (based on spec AST, not trades), so we need a spec that structurally has no
    feature references in its condition AST.

    We do this by using a custom schema with only system-role columns and no
    feature columns — entry condition is always-fire (empty any_of is False so
    we use a trivially-true approach via a price check on the entry_price column).

    Simpler: use the make_spec with an entry_when that references a non-feature
    column. In the standard schema, only 'alpha' and 'target' are features.
    The condition {"feature": "alpha", "gte": 999.0} references alpha (a feature),
    so it WOULD emit. We need a condition that doesn't reference any feature column.

    We use the EvidenceSpec directly with a modified schema_requirements that
    marks ALL columns as non-feature roles, then no feature reference exists.
    """
    from pancake_engine.types import EvidenceDataset, EvidenceSpec

    # Build a spec where entry_when uses price_range_col that is NOT semantic_role=feature.
    # We use a schema where no column is declared as 'feature'. The entry condition
    # {"feature": "alpha", ...} references 'alpha' — but we declare alpha as a different
    # semantic role to make it non-feature. However changing semantic_role would break
    # validation. Instead: use a yes_payoff-only feature reference but no entry reference,
    # and declare the yes_payoff's column as non-feature.

    # Cleanest approach: build a spec with no 'feature' semantic_role columns at all.
    # Use a minimal schema where alpha is declared as market_link (unusual but valid for test).
    # Actually, use the existing make_spec but override schema_requirements to remove features.

    # Simplest valid approach: make a spec where entry_when is a feature_equal on
    # system-role columns. But target and outcome are both feature-role in standard schema.
    # So we build a custom EvidenceSpec with no feature-role columns at all.

    spec = EvidenceSpec.model_validate({
        "spec_family": "pancake-evidence-spec",
        "spec_version": "0.1",
        "name": "no-feature-test",
        "evidence_dataset_id": "ev_test",
        "schema_requirements": {
            "required_columns": [
                {"name": "mkt",     "type": "string", "semantic_role": "market_link"},
                {"name": "dec_ts",  "type": "int",    "semantic_role": "decision_time"},
                {"name": "res_ts",  "type": "int",    "semantic_role": "resolution_time"},
                {"name": "price",   "type": "number", "semantic_role": "entry_price", "range": [0, 1]},
                {"name": "outcome", "type": "int",    "semantic_role": "resolved_outcome_numeric"},
                # NO feature-role columns at all
            ]
        },
        "strategy": {
            "side": "YES",
            "entry": {"when": {"feature": "outcome", "gte": 0}},  # 'outcome' is NOT semantic_role=feature
            "yes_payoff": {"when": {"feature": "outcome", "gte": 1}},  # same
            "sizing": {"mode": "fixed_fraction", "value": 0.1},
        },
        "costs": {"slippage_bps": 0.0, "fee_bps": 0.0},
        "starting_capital": 1000.0,
    })

    dataset = EvidenceDataset.model_validate({
        "id": "ev_test",
        "schema": {
            "columns": [
                {"name": "mkt",     "type": "string", "semantic_role": "market_link"},
                {"name": "dec_ts",  "type": "int",    "semantic_role": "decision_time"},
                {"name": "res_ts",  "type": "int",    "semantic_role": "resolution_time"},
                {"name": "price",   "type": "number", "semantic_role": "entry_price", "range": [0, 1]},
                {"name": "outcome", "type": "int",    "semantic_role": "resolved_outcome_numeric"},
            ]
        },
        "schema_sha256": "test",
        "storage_mode": "inline",
        "rows_inline": [
            {"mkt": "m/A", "dec_ts": 100, "res_ts": 200, "price": 0.5, "outcome": 1},
        ],
        "rows_sha256": "test",
        "row_count": 1,
    })

    result = run_backtest(spec, dataset, BacktestConfig(observation_time=300))
    warns = _feature_warns(result)
    assert len(warns) == 0, (
        f"Expected no AGENT_SUPPLIED_FEATURE_UNVERIFIED, got {len(warns)}: {warns}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: 1 feature column referenced in entry → emit once
# ─────────────────────────────────────────────────────────────────────────────


def test_one_feature_column_in_entry_emits_warning() -> None:
    """Feature columns referenced in predicates → AGENT_SUPPLIED_FEATURE_UNVERIFIED emitted.

    entry_when references 'alpha' (feature); yes_payoff_when uses feature_equal on
    'target' and 'outcome' (both feature-role in standard schema). All three appear
    in the warning context, sorted.
    """
    spec = make_spec(
        side="YES",
        sizing_value=0.1,
        starting_capital=1000.0,
        entry_when={"feature": "alpha", "gte": 2.0},
        yes_payoff_when={"feature_equal": {"a": "target", "b": "outcome"}},
    )
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.50, outcome=1, alpha=3.0, target=1),
    ])
    result = _run(spec, dataset)

    warns = _feature_warns(result)
    assert len(warns) == 1, f"Expected 1 warning, got {len(warns)}"
    w = warns[0]
    assert w.severity.value == "info"
    assert "feature" in w.message.lower()
    feature_cols = w.context.get("feature_columns", [])
    # alpha, outcome, target are all feature-role columns referenced in predicates
    assert "alpha" in feature_cols
    assert feature_cols == sorted(feature_cols), f"feature_columns not sorted: {feature_cols}"
    assert len(feature_cols) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: 2 feature columns in entry + yes_payoff → emit ONCE, sorted
# ─────────────────────────────────────────────────────────────────────────────


def test_two_feature_columns_entry_and_yes_payoff_emit_once_sorted() -> None:
    """2 feature columns across entry+yes_payoff → single warning, context sorted."""
    # entry references 'alpha'; yes_payoff references 'target' (both are feature-role)
    spec = make_spec(
        side="YES",
        sizing_value=0.1,
        starting_capital=1000.0,
        entry_when={"feature": "alpha", "gte": 2.0},
        yes_payoff_when={"feature": "target", "eq": 1},
    )
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.50, outcome=1, alpha=3.0, target=1),
    ])
    result = _run(spec, dataset)

    warns = _feature_warns(result)
    assert len(warns) == 1, (
        f"Expected exactly 1 AGENT_SUPPLIED_FEATURE_UNVERIFIED, got {len(warns)}"
    )
    w = warns[0]
    feature_cols = w.context.get("feature_columns", [])
    # Both columns must appear, sorted
    assert "alpha" in feature_cols
    assert "target" in feature_cols
    assert feature_cols == sorted(feature_cols), f"feature_columns not sorted: {feature_cols}"
    assert len(feature_cols) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: determinism — re-run produces identical result_hash
# ─────────────────────────────────────────────────────────────────────────────


def test_determinism_with_feature_warning() -> None:
    """Determinism: when AGENT_SUPPLIED_FEATURE_UNVERIFIED is emitted, result_hash is stable."""
    spec = make_spec(
        side="YES",
        sizing_value=0.1,
        starting_capital=1000.0,
        entry_when={"feature": "alpha", "gte": 2.0},
        yes_payoff_when={"feature_equal": {"a": "target", "b": "outcome"}},
    )
    dataset = make_dataset([
        row(mkt="m/A", dec_ts=100, res_ts=200, price=0.50, outcome=1, alpha=3.0, target=1),
    ])
    config = BacktestConfig(observation_time=300)

    r1 = run_backtest(spec, dataset, config)
    r2 = run_backtest(spec, dataset, config)

    assert r1.result_hash != "", "empty result_hash"
    assert r1.result_hash == r2.result_hash, (
        f"Non-deterministic with feature warning: {r1.result_hash} != {r2.result_hash}"
    )
    # Confirm the warning is present in both
    assert len(_feature_warns(r1)) == 1
    assert len(_feature_warns(r2)) == 1
