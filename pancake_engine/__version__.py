"""Version + engine identity constants for Pancake Engine 0.4.

These constants are written into every result emitted by the runner (PR-1+).
They are also part of `result_hash` — bumping any of them is a deliberate
breaking change to the receipt contract.

0.4.0 adds:
  - MC bootstrap CI for cagr / sharpe / sortino (percentile method, PCG64 RNG)
  - Sign-permutation test for Sharpe null (Good 2005)
  - numpy>=1.26 as hard dependency
  - New warning codes: BOOTSTRAP_INSUFFICIENT, CI_TOO_WIDE, PERMUTATION_P_HIGH

0.4.1 (docs+config patch):
  - Permanently scope-qualifies Python 3.11 after root-cause investigation
    (sum() float accumulation changed in 3.12; 1-ULP CI drift; unfixable).
  - See docs/py311-investigation-2026-05-27.md for full analysis.
  - No math changes; result_hash unchanged for Python 3.12+ users.

0.4.2 (first PyPI release):
  - First release published to PyPI via Trusted Publishing (OIDC).
  - Engine code byte-identical to 0.4.1; result_hash unchanged.
  - Install: `pip install batter` (previously git-only via release tarball).

0.4.3 (paper /tick):
  - Adds the single-bar paper `tick()` + `SimFillRouter` (ADR-0035 +
    engine-confirmation addendum). Additive surface;
    `run_backtest` and its `result_hash` are byte-unchanged.

0.5.0 (daily-returns correction — DELIBERATE result_hash break):
  - Fixes `daily_returns_carry_forward`: on the start UTC day with >=2 equity
    points it used the FIRST point instead of the LAST same-day close,
    corrupting day-0 and thus Sharpe / Sortino / bootstrap-CI / permutation-p.
    A real bug AND a TS<->Python parity divergence (TS used last-same-day); the
    fix restores parity (usepancake/batter#7).
  - ENGINE_VERSION 0.4.0 -> 0.5.0 is a deliberate receipt-contract break: it is
    part of `result_hash`, so EVERY hash changes. Headline P&L is unchanged for
    nearly all receipts — only risk-stats move, and only for backtests with >=2
    equity points on the start UTC day. Published receipts are re-run
    transparently (per-receipt old_hash -> new_hash correction record), never
    silently version-pinned. The moat is "correct + checkable", not "immutable".
    (policy B, Michael 2026-06-04)

0.6.0 (statistics-correctness + hardening — DELIBERATE result_hash break):
  - Permutation p-value is now (count+1)/(n+1) (Phipson & Smyth 2010) and can no
    longer be exactly 0; changes sharpe_p_value (→ result_hash) for any backtest
    with >=10 daily returns.
  - Bootstrap returns (None, None) + BOOTSTRAP_INSUFFICIENT for a degenerate
    zero-width CI instead of a misleading (v, v).
  - Stricter spec compile: typo'd/unknown operator keys and bare feature nodes are
    rejected (were silently always-true); feature_equal requires string columns and
    both sides present.
  - n_resamples / n_permutations capped at 1e6 (public-API DoS guard).
  - Python >=3.12 enforced at import (3.11 sum() float drift silently changes hashes).
  - Also folds in the additive crypto-OHLCV spec family (does not change
    evidence-spec hashes).
  - ENGINE_VERSION 0.5.0 -> 0.6.0 is part of result_hash, so EVERY hash changes;
    published receipts are re-run transparently (per-receipt old_hash -> new_hash
    correction record), never silently version-pinned. (policy B; audit 2026-06-04)

0.7.0 (robustness panel — ADDITIVE, result_hash UNCHANGED):
  - Adds run_sensitivity_analysis: a 7x7 entry-threshold x sizing-fraction Sharpe
    sweep + per-step Monte-Carlo drawdown fan (ADR-0046, usepancake/batter#12+#13),
    exposed at POST /sensitivity. Purely additive — run_backtest and its
    result_hash are byte-unchanged, so ENGINE_VERSION stays 0.6.0 (NO receipt
    break, no correction records). Package __version__ bumps to 0.7.0 as the PyPI
    release vehicle for the new surface (same pattern as 0.4.1/0.4.2/0.4.3).

0.7.1 (sweep perf — ADDITIVE, result_hash UNCHANGED):
  - run_backtest gains a `with_inference: bool = True` EXECUTION arg (not a
    BacktestConfig field, so config_hash/result_hash are untouched). When False
    it skips the bootstrap CIs + permutation test. run_sensitivity_analysis sets
    it False for every cell — the sweep only needs Sharpe, so the per-cell
    inference was ~50× wasted work that blew the request budget. Cuts the sweep
    from minutes to seconds. Default True → every receipt path is byte-identical;
    ENGINE_VERSION stays 0.6.0.
"""

__version__ = "0.7.1"
ENGINE = "batter"
# Deliberately NOT bumped: 0.7.0 is additive (sensitivity); run_backtest math and
# result_hash are unchanged. ENGINE_VERSION is part of result_hash — bumping it
# would force a needless correction record on every existing receipt.
ENGINE_VERSION = "0.6.0"
ENGINE_MODE = "event_time_v1"

# Verification grade the engine self-identifies with (rule 159 / ADR-0035 §2.2).
# This is a CONTRACT TOKEN tied to the engine *generation* (the 0.3 execution
# trust-layer per ADR-0031), intentionally DECOUPLED from ``ENGINE_VERSION``
# (0.4.0 is a metrics-credibility revision of the same trust-layer, not a new
# generation). It must stay in the dispatcher's ``PaperVerificationGrade`` union
# (``engine-0.3-canonical | ts-shim-degraded``); changing it is a deliberate
# cross-repo contract change.
ENGINE_VERIFICATION_GRADE = "engine-0.3-canonical"
