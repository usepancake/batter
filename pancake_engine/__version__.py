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
"""

__version__ = "0.4.1"
ENGINE = "batter"
ENGINE_VERSION = "0.4.0"
ENGINE_MODE = "event_time_v1"
