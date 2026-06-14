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

0.7.2 (validation hardening — ADDITIVE, result_hash UNCHANGED):
  - validate_dataset now enforces entry_price ∈ [0, 1] even when neither the spec
    requirement nor the dataset column declares a range. Previously an
    out-of-range entry_price with no declared range was silently skipped at run
    time with an ENTRY_PRICE_OUT_OF_RANGE warning instead of failing pre-flight,
    so the MCP surface reported it as an opaque engine error (error-recovery eval
    2026-06-06). Stricter rejection only — valid datasets are byte-unchanged, so
    ENGINE_VERSION stays 0.6.0.

0.8.0 (credibility release — DELIBERATE result_hash break):
  - MetricsStandard gains `psr` (Probabilistic Sharpe Ratio, Bailey & López de
    Prado 2012) and `min_track_record_length` (MinTRL) — the primary significance
    + sufficiency signals, complementing the sign-permutation sharpe_p_value.
  - The bootstrap CIs (cagr_ci / sharpe_ci / sortino_ci) switch from the 0.4 IID
    resample to a STATIONARY block bootstrap (Politis-Romano 1994), preserving
    serial correlation → honester (wider) intervals for autocorrelated returns.
  - Additive, non-hashed surfaces also land: deflated Sharpe (DSR) + BHY
    false-discovery-rate control on the sensitivity sweep, and transaction-cost
    sensitivity + break-even multiplier.
  - ENGINE_VERSION 0.6.0 -> 0.8.0 is part of result_hash, so EVERY hash changes;
    published receipts are re-run transparently (per-receipt old_hash -> new_hash
    correction record), never silently version-pinned. (policy B; credibility-first
    sequencing 2026-06-10 — receipt break-cost ~0 post the ADR-0045 reset.)

0.8.1 (cross-platform determinism hotfix — DELIBERATE result_hash break of 0.8.0):
  - 0.8.0 computed psr via math.erf (libm): glibc and Apple libm round the last
    ULP differently, so psr — a HASHED field — diverged by 2 ULP between ubuntu
    and macOS and result_hash was platform-ambiguous (caught by the 6-matrix CI;
    field-level bit probe isolated psr as the sole divergent field).
  - Φ and the probit now run in 50-digit decimal (libmpdec; correctly rounded BY
    SPECIFICATION on every platform): erf by Maclaurin series with pinned
    constants; probit = Acklam float seed + Decimal-Newton (quadratic convergence
    erases seed ULPs). Bit-exact contract pinned in tests/test_phi_bit_exact.py.
  - Hashed-path discipline going forward: hashed values may only come from
    IEEE-exact float ops (+,-,*,/,sqrt, fsum) or spec-correctly-rounded decimal.
  - ENGINE_VERSION 0.8.0 -> 0.8.1: psr moves at the ULP level, so hashes change.
    0.8.0 was never deployed to Fly and minted zero receipts; 0.8.0 receipts
    cannot exist consistently (platform-ambiguous), so the break is free.

0.9.0 (proof-layer release — DELIBERATE result_hash break):
  - MetricsPM gains ``calibration_ece: float | None`` (Expected Calibration Error
    over 10 fixed bins on the traded-side (implied_prob_at_entry,
    realized_outcome_for_trade) pairs; None when num_trades < 10). This field is
    HASHED → every receipt hash changes. Pure arithmetic (fsum) — deterministic
    on all platforms without libm.
  - Additive, non-hashed: ``BacktestResult.calibration_bins`` (reliability curve
    per-bin shape; same < 10 threshold; only computed when with_inference=True).
  - Additional surfaces also landing in 0.9.0 on the same break: crypto-OHLCV
    receipts on the DatasetContract Seam (next_bar_open@1 fill reference); fill-
    model registry (static_bps@1 default, book_replay@1, next_bar_open@1);
    paper guards + exit-on-tick semantics; verify CLI; PBO/CPCV walk-forward;
    ledger seam (deflated block); macro contract dataclass; sensitivity heatmap
    grids. All hash changes are governed by policy B (per-receipt rerun).
  - ENGINE_VERSION 0.8.1 -> 0.9.0; every existing hash changes; published
    receipts are re-run transparently (per-receipt old_hash -> new_hash correction
    record), never silently version-pinned. (policy B; 2026-06-10)

0.9.1 (paper + replay surfaces — ADDITIVE, result_hash UNCHANGED):
  - book_replay@1 PM fill model: VWAP walk of captured L2 ask levels
    (ADR-0041 slices via run_backtest(book_dataset=...)); slice-missing /
    depth-insufficient BLOCK (no silent fallback); the book dataset id+shas
    pin into result_hash ONLY for runs that used replay — every existing
    hash is byte-identical (asserted by test). ttr_fill_adjustment is
    declared-but-reserved and hard-rejects.
  - FillBlocked joins the public FillModel Protocol (EntryFill | FillBlocked).
  - SimFillRouter resolves fill models from the registry — paper and backtest
    share one fill implementation (live/paper parity by construction).
  - tick_crypto(): single-bar crypto paper evaluation; bar_close fill
    convention (deliberate, documented divergence from backtest
    next-bar-open; surfaced as paper_fill_convention on every response).
  - _phi lower-tail docstring: |z|>=11 -> 0.0 stated as the deliberate
    symmetric convention (Haiku-swarm finding; zero behavior change).
  - ENGINE_VERSION stays 0.9.0 — no receipt break, no corrections.

0.10.0 (The Chain + The Ledger — ADDITIVE, result_hash UNCHANGED):
  - chain/1: hash-linked deployment records (ChainRecord/ChainBuilder/
    verify_chain + `batter verify --chain`). Genesis pins compiled_spec_hash +
    backtest result_hash + dataset id + starting_cash; verify does EXACT P&L
    roll-forward (E_CHAIN_CASH_MISMATCH on bit mismatch — no heuristics).
    Order-state machine with frozen transition table; cumulative-fill
    monotonicity; every tamper class tested.
  - trials/1: run_many batch orchestration threading an accumulating trial
    history into each run's deflated block (running DSR as the search
    deepens); TrialLedger hashed session record; final_dsr vs the complete
    ledger. Per-run result_hash byte-identical to standalone runs (asserted).
  - portfolio/1: compute_portfolio — joint carry-forward equity, portfolio
    metrics, pairwise correlations, leg result_hash provenance pins, own
    portfolio_hash. regime block (additive non-hashed quartile stability).
    SportsEventContract on the domain Seam.
  - ENGINE_VERSION stays 0.9.0 — new artifact classes carry their own
    format_version; existing receipt hashes never move. No corrections.


0.10.1 (live tick mode — ADDITIVE, result_hash UNCHANGED):
  - TickRequest.mode gains "live" (ADR-0050 L1, Path A — the executor intent
    contract). Identical decision path to paper (settle, mark, guards, entry,
    sizing) to a single branch point at the fill boundary; live mode stops
    there: no fill router, no state mutation. Emits target_positions
    {instrument_id, side, target_shares, signal_price, source_manifest_id}
    (rule-151 provenance echoed from the triggering bar). Guards gate live
    intents exactly as they gate paper fills. Paper callers byte-unchanged.
  - Parity contract: decision SET identical across modes by construction;
    target_shares uses undepleted cash (live has no intra-tick fills), so
    share-quantity parity vs paper holds for the first entry per tick and
    paper cash-exhaustion can suppress marginal fills live still targets —
    the executor owns allocation across simultaneous targets.
  - ENGINE_VERSION stays 0.9.0 — tick is not part of result_hash.


0.10.2 (sweep trial_stats — ADDITIVE, result_hash UNCHANGED):
  - SensitivityResult.trial_stats {n_trials, sharpe_best, dsr_best,
    expected_max_sharpe}: multiple-testing honesty for the sweep surface
    (Slice D engine half). expected_max_sharpe extracted from
    deflated_sharpe_ratio bit-identically (hex-pinned). Base-is-best reuses
    the existing deflated value verbatim (ties prefer base). Null semantics:
    sharpe_best null iff no defined cells; expected_max_sharpe null iff
    n_trials < 2 or zero trial variance; dsr_best null iff either of those
    or the best cell re-run has < 2 daily returns. Sweeps are not receipts —
    no result_hash impact (oracle-pinned).

0.10.3 (verify CLI labels — ADDITIVE, result_hash UNCHANGED):
  - `batter verify` version warning prefix-normalized (#46): bundles stamping
    the row format 'batter@0.9.0' no longer false-warn against the bare
    self-report '0.9.0'; real identity mismatches still warn.
  - verify JSON output (success + integrity-failure paths) names both version
    concepts per pancake-production rule 173: package_version (PyPI release)
    + result_hash_identity (hash identity); engine_version kept as deprecated
    alias. Human line adds pkg=<package_version>.
  - Removes a dead crypto_ohlcv import in runner/tick.py that had main's ruff
    gate red since 0.10.x.
  - CLI-output only: run_backtest and every hash byte-unchanged;
    ENGINE_VERSION stays 0.9.0.

0.10.4 (spec-rejection + metric-overflow hardening — result_hash UNCHANGED):
  - validate_spec now lints condition ASTs (lint_condition): malformed conditions
    (empty all_of, unknown/typo operator keys, bare feature, empty when-nodes,
    malformed feature_equal) return a clean blocked verdict instead of raising an
    uncaught ValueError (which the prod shim turned into HTTP 500). (#49)
  - PM / standard / PSR / permutation metric paths degrade to None on OverflowError
    instead of raising, when a denormal entry_price (e.g. 1e-203) drives a return
    to ~1e+200 and squaring overflows float64. (#49)
  - Robustness-only: run_backtest and every hash byte-unchanged for valid specs;
    ENGINE_VERSION stays 0.9.0.
"""

__version__ = "0.10.4"
ENGINE = "batter"
# 0.9.0 is a DELIBERATE result_hash break: MetricsPM gains calibration_ece (hashed).
ENGINE_VERSION = "0.9.0"
ENGINE_MODE = "event_time_v1"

# Verification grade the engine self-identifies with (rule 159 / ADR-0035 §2.2).
# This is a CONTRACT TOKEN tied to the engine *generation* (the 0.3 execution
# trust-layer per ADR-0031), intentionally DECOUPLED from ``ENGINE_VERSION``
# (0.4.0 is a metrics-credibility revision of the same trust-layer, not a new
# generation). It must stay in the dispatcher's ``PaperVerificationGrade`` union
# (``engine-0.3-canonical | ts-shim-degraded``); changing it is a deliberate
# cross-repo contract change.
ENGINE_VERIFICATION_GRADE = "engine-0.3-canonical"
