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
  - Adds the single-bar paper `tick()` + `SimFillRouter` (ADR-0035, amended by
    pancake-production 0035-amendment-engine-confirmation.md). Additive surface;
    `run_backtest` and its `result_hash` are byte-unchanged.
"""

__version__ = "0.4.3"
ENGINE = "batter"
ENGINE_VERSION = "0.4.0"
ENGINE_MODE = "event_time_v1"

# Verification grade the engine self-identifies with (rule 159 / ADR-0035 §2.2).
# This is a CONTRACT TOKEN tied to the engine *generation* (the 0.3 execution
# trust-layer per ADR-0031), intentionally DECOUPLED from ``ENGINE_VERSION``
# (0.4.0 is a metrics-credibility revision of the same trust-layer, not a new
# generation). It must stay in the dispatcher's ``PaperVerificationGrade`` union
# (``engine-0.3-canonical | ts-shim-degraded``); changing it is a deliberate
# cross-repo contract change.
ENGINE_VERIFICATION_GRADE = "engine-0.3-canonical"
