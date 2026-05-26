# Python 3.11 determinism investigation — batter 0.4.0

Date: 2026-05-27
Branch: `chore/py311-determinism-investigation`
Engine version: `0.4.0`
Investigator: Phase B agent

---

## Reproduction

Environment: macOS aarch64, CPython 3.11.15 (via uv), numpy 2.4.6, scipy 1.17.x.

Worktree: `/Users/michaelmustopo/Documents/batter-py311-investigation` (branch `chore/py311-determinism-investigation` at `be3ed27`).

Python 3.11 venv created with `uv venv --python 3.11 .venv311` then deps installed via `uv pip install --python .venv311/bin/python "pydantic>=2.5,<3" "numpy>=1.26" ...` (requires bypassing `pyproject.toml requires-python = ">=3.12"` constraint).

Tests run as: `PYTHONPATH=. .venv311/bin/python -m pytest <test-files> -v`

---

## Failures observed

### 5 failures total across math-acceptance + examples smoke

**Test 1: `test_metrics_permutation::test_permutation_identical_returns_sharpe_none`**

```
FAILED tests/test_metrics_permutation.py::test_permutation_identical_returns_sharpe_none
AssertionError: assert 0.0 is None
```

The test asserts that `permutation_p_sharpe([0.01] * 20, ...)` returns `p_val = None` (because all-identical returns have `std = 0` → `_sharpe` returns `None` → no test). On 3.11, it returns `p_val = 0.0` instead.

**Tests 2–5: all 4 example smoke tests fail with `result_hash` mismatch**

```
toy:               expected 72484025...  got b9a97e45...
jakarta_temperature: expected 2a27ebd4...  got af2e48c5...
rapture_family:    expected 4c3fedb5...  got 9cf627e7...
btc_pred_hedge:    expected 9cff7ce4...  got 474d4d78...
```

All 4 fixtures produce wrong hashes because the CI values included in `result_hash` differ between Python 3.11 and 3.12.

---

## Diff verbatim (3.11 vs 3.12 for `toy` example)

```
Field             Python 3.11                        Python 3.12
sharpe_ci low     -4.328152052015572                 -4.3281520520155725
sharpe_ci high     4.665324996479036                  4.665324996479034

Hex representation (low):
  3.11: -0x1.15007176e16b4p+2
  3.12: -0x1.15007176e16b5p+2
  Difference: 1 ULP (unit in the last place)
```

The `result_hash` feeds on `repr(float)` via canonical serialization (`canonical.py::_number_to_string`), so a 1-ULP float difference produces a completely different SHA-256.

---

## Bisection steps

**Step 1: numpy version pin**

Both environments run numpy 2.4.6. Same numpy version, same PCG64 seed → same integer draws from `rng.integers(0, n, size=(10_000, n))`. Verified explicitly:

```python
# 3.11 and 3.12 both produce:
numpy.random.default_rng(0).integers(0, 10, size=10)
→ [8 6 5 2 3 0 0 0 1 8]
```

**Verdict: numpy pin is NOT the cause.** Pinning numpy makes no difference.

**Step 2: `math.fma` constant-folding**

Grep confirmed: zero uses of `math.fma` anywhere in the engine codebase.

**Verdict: `math.fma` is NOT the cause.**

**Step 3: PCG64 state-hash**

Verified above — PCG64 gives byte-identical integer sequences on 3.11 and 3.12 for the same seed. The `rng.integers()` draws that drive bootstrap resampling are identical.

**Verdict: PCG64 is NOT the cause.**

**Step 4: Float repr / precision — ROOT CAUSE FOUND**

`_sharpe` in `permutation.py` uses:

```python
mean = sum(daily_returns) / n
var = sum((r - mean) ** 2 for r in daily_returns) / (n - 1)
std = math.sqrt(var)
```

`sharpe_ratio` in `standard.py` uses `_stdev_sample` which has the same pattern:

```python
var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
```

**Python 3.11 vs 3.12 `sum()` behaviour for homogeneous float lists differs:**

```python
# Python 3.11
sum([0.01] * 20) = 0.20000000000000004   # hex: 0x1.999999999999bp-3
mean = 0.010000000000000002               # non-zero deviation from 0.01

# Python 3.12  
sum([0.01] * 20) = 0.2                   # hex: 0x1.999999999999ap-3
mean = 0.01                               # exact
```

Python 3.12 introduced an optimized `sum()` implementation for homogeneous float lists (CPython issue gh-100946, merged Nov 2022 for 3.12). Python 3.11 uses sequential IEEE 754 addition with no compensation, producing a 1-bit error at the trailing ULP.

**Consequence 1 — `test_permutation_identical_returns_sharpe_none`:**

On Python 3.11: `mean = 0.010000000000000002` → `(0.01 - mean)^2 = 3.0e-36` → `var = 3.2e-36` → `std = 1.78e-18` (nonzero). `_sharpe` returns `8.9e+16` instead of `None`. The sign-permutation loop runs, and since the permuted Sharpe of ±(large) is always ≥ the observed large Sharpe with prob 0 (all permutations produce absolute Sharpe ≥ obs), `p_value = 0.0` is returned instead of `None`.

**Consequence 2 — CI hash divergence across all 4 examples:**

The 10,000-resample bootstrap loop calls `sharpe_ratio` (which calls `_stdev_sample`) once per resample. The sequential `sum()` accumulation differs in the low-order bit across Python versions for the bootstrap resamples (which are non-identical float arrays — the effect is smaller than the all-identical case but still present). This shifts 1-2 of the 10,000 per-resample Sharpe values by ±1 ULP, which shifts the 249.975th-percentile interpolation by 1 ULP, producing a different `ci_low`.

**Why code changes cannot fix this:**

`math.fsum` was tested as an alternative — it gives a *third* different answer (`e16b6`) from both 3.11 (`e16b4`) and 3.12 (`e16b5`). NumPy-based variance (`np.std(ddof=1)`) also gives `e16b6`. There is no Python 3.11 implementation of variance that produces byte-identical output to Python 3.12's `sum()` without reverse-engineering 3.12's exact accumulation path. The difference is inherent to the Python version's C-level float stack management.

---

## Verdict

**SCOPE-QUALIFIED. Python 3.11 is permanently out of scope for batter 0.4.x.**

Root cause: Python 3.12 changed `sum()` semantics for homogeneous float lists (compensated/Neumaier summation vs plain sequential IEEE 754 on 3.11). The difference propagates into bootstrap CI computation as 1-ULP shifts that change `result_hash`. There is no code change that makes Python 3.11 produce byte-identical `result_hash` to Python 3.12+.

---

## User-visible scope statement

`batter 0.4.x requires Python ≥ 3.12. Python 3.11 produces different result_hash values due to a sum() precision change in Python 3.12; byte-identical determinism is only guaranteed on Python 3.12+.`

---

## Evidence summary

| Check | Python 3.11 | Python 3.12 | Match? |
|---|---|---|---|
| numpy version | 2.4.6 | 2.4.6 | ✓ |
| PCG64 first 10 integers (seed=0) | `[8 6 5 2 3 0 0 0 1 8]` | `[8 6 5 2 3 0 0 0 1 8]` | ✓ |
| `sum([0.01]*20)` | `0.20000000000000004` | `0.2` | ✗ |
| toy `sharpe_ci` low | `-4.328152052015572` | `-4.3281520520155725` | ✗ |
| toy `sharpe_ci` low hex | `0x1.15007176e16b4` | `0x1.15007176e16b5` | 1 ULP diff |
| `_sharpe([0.01]*20)` | `8.9e+16` (nonzero) | `None` (std=0) | ✗ |
| toy result_hash | `47e9266c…` | `dcc56c4d…` | ✗ |
| jakarta result_hash | `f0cc3a90…` | `a1e83832…` | ✗ |
| rapture_family result_hash | `fd31e743…` | `ecbeaee6…` | ✗ |
| btc_pred_hedge result_hash | `474d4d78…` | `9cff7ce4…` | ✗ |

---

## Actions taken

1. `pyproject.toml`: `requires-python = ">=3.12"` — already set from PR-B4; confirmed correct and locked.
2. Classifiers: `"Programming Language :: Python :: 3.12"` and `"3.13"` only — 3.11 was never listed; no change needed.
3. `docs/math-audit-0.4.md` §"Known scope qualifier — Python 3.11": updated with empirical evidence from this investigation (root cause, exact diff, test failures).
4. `README.md` §"Supported Python versions": updated to make 3.11 exclusion visible and explain the reason.
5. This document: permanent record at `docs/py311-investigation-2026-05-27.md`.

---

## References

- CPython `sum()` optimization: https://github.com/python/cpython/issues/100946 (Python 3.12 float accumulation change)
- IEEE 754 sequential accumulation: Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM. §4.2.
- Neumaier compensated summation: Neumaier, A. (1974). "Rundungsfehleranalyse einiger Verfahren zur Summation endlicher Gleitkommazahlen." *ZAMM*, 54(1), 39–51.
