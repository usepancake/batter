# 0.4.0 bootstrap CI + permutation test — formulas, sources, fixtures

Engine version: `0.4.0-rc1`
Date: 2026-05-26
Author: Engine 0.4 implementation (Pancake Engine)

---

## Overview

Engine 0.4 adds two statistical credibility layers on top of the existing point
estimates (CAGR, Sharpe, Sortino):

1. **Monte Carlo bootstrap confidence intervals** — percentile method, 10 000
   resamples, seed=0 for determinism.
2. **Sign-permutation test for Sharpe** — null hypothesis Sharpe = 0, 10 000
   permutations, seed=0.

These are additive fields on `MetricsStandard` with backward-compatible defaults
`(None, None)` / `None`, so no existing fixtures break.

---

## Bootstrap CI (percentile method)

### Formula

Given a sample of N daily returns `r_1, ..., r_N`:

1. Draw B = 10 000 bootstrap resamples (with replacement) of size N.
2. For each resample, compute the metric `θ*_b = metric_fn(resample_b)`.
3. Sort the B bootstrap statistics.
4. 95% CI: `(θ*_{α/2}, θ*_{1-α/2})` where α = 0.05.
   - ci_low  = 2.5th percentile of the B bootstrap statistics
   - ci_high = 97.5th percentile of the B bootstrap statistics

### Code path

```
pancake_engine/metrics/bootstrap.py::bootstrap_ci()
  └── called from pancake_engine/metrics/standard.py::compute_standard()
        ├── sharpe_ci  = bootstrap_ci(daily_rets, sharpe_ratio)
        ├── sortino_ci = bootstrap_ci(daily_rets, sortino_ratio)
        └── cagr_ci    = bootstrap_ci(daily_rets, _cagr_proxy_fn(...))
```

`_cagr_proxy_fn` approximates CAGR from resampled daily returns via geometric
compounding: `ending_boot = starting_capital × Π(1 + r_i)`.

### Sources

- Efron, B. (1979). "Bootstrap methods: Another look at the jackknife."
  _Annals of Statistics_, 7(1), 1–26. https://doi.org/10.1214/aos/1176344552
- Efron, B., & Tibshirani, R. J. (1993). _An Introduction to the Bootstrap_.
  Chapman & Hall. §13.3 (percentile interval), §14.3 (bias and variance).
- Hyndman, R. J., & Athanasopoulos, G. (2018). _Forecasting: Principles and
  Practice_, 2nd ed. OTexts. §3.5 (bootstrap forecast intervals).

### Why percentile, not BCa?

BCa (bias-corrected accelerated) intervals require the jackknife for acceleration
constant estimation, adding O(N²) computation and code complexity. Percentile is
simpler, correct in the first order, and adequate for the metrics we test (Sharpe,
CAGR). BCa is deferred to a future release.

### Hand-calc fixture A — Sharpe CI on 20-day return series

Input:
```python
returns = [0.01, -0.01, 0.02, -0.02, 0.005, -0.005, 0.015, -0.015, 0.0, 0.008,
           0.010, -0.013, 0.002, 0.021, -0.009, 0.005, 0.013, -0.007, 0.008, 0.016]
n_resamples = 10_000, seed = 0
```

Expected (computed and verified deterministically):
```
observed_sharpe = 3.545810
sharpe_ci = (-3.420357, 12.155009)
```

Note: CI is wide (relative width ≈ 4.4×) due to small N=20. CI_TOO_WIDE does not
fire at 4.4× (threshold is 5×).

The observed Sharpe falls within the CI: `ci_low ≤ sharpe ≤ ci_high`. ✓

This is a structural property of the percentile method (the observed statistic is
always between the percentile bounds unless the bootstrap distribution is degenerate).

### Hand-calc fixture B — CAGR CI on same series

Input: same 20-day returns, `starting_capital=1000, ending_equity=1200, period=30 days`

Expected:
```
cagr_ci = (-0.472686, 5.385626)
```

Wide CI reflects high uncertainty from N=20 and a 30-day period.

### Guards

- Returns `(None, None)` if N < 2, emits `BOOTSTRAP_INSUFFICIENT`.
- Returns `(None, None)` if all returns are identical (zero variance), emits
  `BOOTSTRAP_INSUFFICIENT`.
- Discards non-finite metric values from boot_stats (AF-3 compat: extreme return
  series can overflow float64 during geometric compounding).
- Emits `CI_TOO_WIDE` if `(ci_high - ci_low) / |point_estimate| > 5.0`.

### CI_TOO_WIDE threshold rationale

For a Sharpe ratio of 0.5, a CI width of 2.5 spans from −1.0 to +1.5 — the
entire plausible range for a modest-skill strategy. The 5× threshold was calibrated
against Ding & Martin (2017) "The Sharpe ratio: statistics and applications," where
annual CI widths beyond ~4× the point estimate correspond to p-values > 0.25 in
practice. The 5× threshold is slightly conservative to avoid false alarms on
legitimately volatile strategies.

---

## Permutation test for Sharpe

### Formula

Null hypothesis: the observed Sharpe ratio is indistinguishable from noise (Sharpe = 0).

1. Compute the observed Sharpe: `S_obs = mean(r) / std(r) × sqrt(252)`.
2. For each of P = 10 000 permutations:
   a. Draw a sign vector `s_i ∈ {-1, +1}` uniformly at random.
   b. Compute the permuted Sharpe: `S_perm = _sharpe(s ⊙ r)`.
3. p-value = `count(|S_perm| ≥ |S_obs|) / P`.

### Code path

```
pancake_engine/metrics/permutation.py::permutation_p_sharpe()
  └── called from pancake_engine/metrics/standard.py::compute_standard()
```

### Sources

- Good, P. I. (2005). _Permutation, Parametric and Bootstrap Tests of Hypotheses_,
  3rd ed. Springer. §4.2 (randomisation tests for financial data).
- Bailey, D. H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier."
  _Journal of Risk_, 15(2), 3–44.

### Why sign permutation?

Sign permutation preserves the marginal distribution of |r_t| while scrambling
direction. It is the maximum-entropy null under the symmetry assumption (returns are
equally likely to be positive or negative). Full row shuffle is equivalent under
i.i.d. but breaks for autocorrelated series; sign permutation is more conservative.

### Hand-calc fixtures

**Fixture C — alternating returns (null is true):**
```python
returns = [0.01 if i%2==0 else -0.01 for i in range(30)]  # N=30, mean=0
observed_sharpe = 0.0
p_value = 1.0  # every permutation is as extreme as observed (all |S|=0)
```
Expected: p = 1.0. ✓

**Fixture D — strongly positive returns (signal is strong):**
```python
returns = [0.05 + (i%3)*0.001 for i in range(20)]  # N=20, strong upward drift
observed_sharpe ≈ 979.68
p_value = 0.0000  # no permutation produces |S| ≥ 979.68
```
Expected: p ≈ 0.0 (minimum is 1/10_000 = 0.0001). ✓

**Fixture E — all-identical returns (Sharpe undefined):**
```python
returns = [0.01] * 10  # std = 0 → Sharpe = None
p_value = None  # cannot test undefined statistic
```
Expected: p = None (no `PERMUTATION_P_HIGH` emitted). ✓

### Guards

- Returns `None` if N < 10, emits `PERMUTATION_P_HIGH` with `context.min_n=10`.
- Returns `None` if observed Sharpe is `None` (std=0 case).
- Emits `PERMUTATION_P_HIGH` when p > 0.10 (signal weak vs random).

---

## Seeded RNG choice + cross-platform stability

### Choice: numpy PCG64

Both `bootstrap_ci` and `permutation_p_sharpe` use `numpy.random.default_rng(seed)`.
Since numpy 1.17, the default RNG is **PCG64** (Permuted Congruential Generator,
64-bit output).

### Why PCG64 over stdlib Mersenne Twister?

| Property | numpy PCG64 | stdlib random.Random (MT) |
|---|---|---|
| Cross-platform byte-stable | ✓ (identical uint64 sequences on macOS/Linux/Windows) | ✗ (float draw algorithm differs across platforms) |
| Passes BigCrush | ✓ | Marginal |
| Used by scipy / scikit-learn | ✓ | No |
| Hard dep required | numpy ≥ 1.26 | None (stdlib) |

PCG64 is preferred because:
1. `numpy.random.default_rng(0).integers(0, N, size=(10_000, N))` produces
   identical uint64 arrays on macOS, Linux, and Windows for the same seed.
2. It is the scipy/LEAN convention, making our implementation auditable against
   published references.
3. numpy was already an implicit dep via many downstream packages; making it
   explicit (≥1.26) is honest about the constraint.

### Determinism contract

- Same `seed` → byte-identical `(ci_low, ci_high)` tuples on the same machine.
- Cross-machine (different OS/arch) stability: PCG64 is byte-stable for integer
  draws, which is all we use (`rng.integers`). Float operations downstream
  (percentile, Sharpe formula) use IEEE 754 double precision, which is consistent
  across modern x86_64 and ARM64 with standard compiler flags.

---

## Fixture re-lock (0.3.0-rc1 → 0.4.0-rc1)

### What changed

The new fields `cagr_ci`, `sharpe_ci`, `sortino_ci`, `sharpe_p_value` are
included in `asdict(metrics.standard)`, which is part of `compute_result_hash`.
Additionally, `ENGINE_VERSION` changed from `"0.3.0"` to `"0.4.0-rc1"`, which is
also in `result_hash`. Both changes are deliberate and expected.

**No math-hash drift outside bootstrap fields.** The existing metrics
(cagr, sharpe, sortino, total_return, max_drawdown, win_rate, equity_curve,
drawdown_curve, monthly_returns, trades) are mathematically unchanged.

### Fixture files updated

| Example | 0.3.0-rc1 result_hash | 0.4.0-rc1 result_hash | Changed fields |
|---|---|---|---|
| `examples/toy/expected_result.json` | `90d05a294c89a2e7e73af78907f0ee4ae0cf3c0a0dc39d5a8610065a3abaf361` | `f13036339552fda089fd963c9b0d4d061080e88ec19ee201a87d5faefc6deb9d` | engine_version + CI fields |
| `examples/jakarta_temperature/expected_result.json` | `cf32074cf78e0fc3fd0f92b2aadaf9aeac7197fbc9061a8b1139c206ec86abce` | `48735f96ab8ffcb0d94e851ccd74cbb5f2b6787bbfcf3215d8bb5eaabc53389a` | engine_version + CI fields |
| `examples/rapture_family/expected_result.json` | `a629a675b2a5bdbf2df806040afcf522dc07d4dc1f96d58a399669a8e66bd074` | `761721c37111f9abe384c5cc506b7f504b235357e01f61cc7a29475face0fa4f` | engine_version + CI fields |
| `examples/btc_pred_hedge/expected_result.json` | `6f3e3017b7612d340448337475d0485115df075f630e7ae24cc882f6a198c30b` | `c753ee60c6e4cb54482ccd694e0f93a22374add3da0cdda98d0ae9a9c372398c` | engine_version + CI fields |

The `tests/fixtures/runner/ts_runner_expected.json` is unchanged — it is generated
by the TS oracle and not affected by Python engine version bumps.

### Verification

```bash
uv run pytest -q  # 184 passed
```

All 10 math acceptance test cases (`test_math_acceptance.py::test_case_01` through
`test_case_10`) pass, confirming no math-hash drift outside bootstrap fields.
The `_assert_hash_stable` helper in each case confirms same-seed determinism.

---

## Cross-platform matrix gap

**Status: deferred to PR-B4.**

The full CI matrix (macOS / Linux / Windows, Python 3.11 / 3.12 / 3.13) that
verifies cross-platform byte-stability of `bootstrap_ci` and `permutation_p_sharpe`
is deferred to PR-B4, when the public GitHub repository is created and GitHub
Actions is configured.

This mirrors the precedent set in PR #206 (Engine 0.3 cross-platform matrix was
also deferred to the public repo PR). The local determinism gate
(`tests/test_bootstrap_determinism.py`) confirms same-machine stability, which is
the minimum requirement for the `v1.3` launch. Cross-platform stability is a
property of PCG64 (see §Seeded RNG above) but is not mechanically verified until
PR-B4.

---

## New warning codes (0.4.0-rc1)

| Code | Group | Severity | Condition |
|---|---|---|---|
| `BOOTSTRAP_INSUFFICIENT` | Credibility | warn | N < 2, zero variance, or all metric_fn calls return None |
| `CI_TOO_WIDE` | Credibility | warn | `(ci_high - ci_low) / \|point_estimate\|` > 5.0 |
| `PERMUTATION_P_HIGH` | Credibility | warn | p-value > 0.10, or N < 10 |

All three codes are included in `result_hash` (via `warnings[*].{code, severity}`),
consistent with the hash policy in `result.py`.
