# Changelog

All notable changes to `batter`. The engine version is part of every
`result_hash`, so any version bump is a deliberate receipt-contract change:
published receipts are re-run transparently with an `old_hash → new_hash`
correction record, never silently version-pinned.

## Unreleased (verify CLI labels — ADDITIVE, `result_hash` UNCHANGED)

- **`batter verify` version warning normalized** (#46): the bundle's
  `engine_version` is stripped of its `<package>@` prefix before identity
  comparison — Pancake replay bundles stamp the row format `batter@0.9.0`,
  which false-warned against the bare self-report `0.9.0` on every
  production bundle. Real identity mismatches still warn.
- **`batter verify` JSON output carries both version concepts** under
  distinct names (pancake-production rule 173): `package_version` (PyPI
  release) + `result_hash_identity` (hash identity); `engine_version`
  stays as a deprecated alias of `result_hash_identity`. The human line
  adds `pkg=<package_version>`.
- CLI-output only — `run_backtest` and every hash are byte-unchanged;
  `ENGINE_VERSION` stays 0.9.0.

## 0.6.0 — 2026-06-04 (statistics-correctness + hardening; deliberate `result_hash` break)

- **Permutation p-value** is now `(count + 1) / (n + 1)` (Phipson & Smyth 2010) and
  can never be exactly `0`.
- **Bootstrap** returns `(None, None)` + `BOOTSTRAP_INSUFFICIENT` for a degenerate
  zero-width CI instead of a misleading `(0, 0)`.
- **Stricter spec compile**: typo'd/unknown operator keys (e.g. `gt` for `gte`) and
  bare feature nodes are rejected (previously silently always-true → "enter on
  everything"); `feature_equal` requires string columns and both sides present.
- **Dataset-declared column ranges** are enforced even when the spec omits them
  (restores parity with the reference TypeScript runner).
- **`math.fsum`** in the hashed float-sum path: `result_hash` is now stable across
  all CPython versions, not just ≥3.12.
- **Guards**: `n_resamples` / `n_permutations` capped at 1e6; Python ≥3.12 enforced
  at import.
- `ENGINE_VERSION` 0.5.0 → 0.6.0. Headline P&L of valid backtests is unchanged; only
  significance stats shift and edge-case CIs become honest.

## 0.5.0 — unreleased (internal checkpoint, folded into 0.6.0)

- Fixed `daily_returns_carry_forward` start-day handling (uses the last same-day
  close); restored TS parity for backtests with ≥2 equity points on the start day.

## 0.4.3 — paper `/tick`

- Single-bar paper `tick()` + `SimFillRouter` (additive; `run_backtest` hash unchanged).

## 0.4.2 — first PyPI release

- First release via OIDC Trusted Publishing. Engine byte-identical to 0.4.1.

## 0.4.1 — Python 3.12+ scope qualifier

- Permanently scope-qualifies Python 3.11 (`sum()` float-accumulation drift).
  Superseded by the `math.fsum` fix in 0.6.0.

## 0.4.0 — bootstrap CI + permutation test

- Monte-Carlo bootstrap CIs (cagr / sharpe / sortino; percentile method, PCG64 RNG)
  and a sign-permutation Sharpe test. `numpy >= 1.26`. Renamed to `batter`.
