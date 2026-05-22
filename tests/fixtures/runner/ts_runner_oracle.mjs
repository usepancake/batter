#!/usr/bin/env node
// Pancake Engine 0.3 — TS runner oracle for golden parity / documented divergence.
//
// Reads 5 curated fixtures (defined below), runs them against the REAL TS
// `runEvidenceBacktest` from pancake-production, and emits `ts_runner_expected.json`
// containing the TS output for each fixture.
//
// The Python test `tests/test_ts_runner_parity.py` then asserts:
//   * For "match" fixtures: Engine 0.3 metrics within 1e-9 of TS metrics.
//   * For "documented_divergence" fixtures: Engine 0.3 differs from TS in the
//     specific way documented in pancake-engine-0.3-ts-divergences.md (D-1, D-3,
//     D-11, D-13, D-14). The divergence shape is asserted explicitly.
//
// Engine 0.3 is correctness-first, not TS parity. This oracle grounds the
// divergence proof in real TS code from pancake-production rather than a port.
//
// Usage (run from inside pancake-engine-py):
//   node tests/fixtures/runner/ts_runner_oracle.mjs
//   PANCAKE_PRODUCTION_ROOT=/abs/path \
//     node tests/fixtures/runner/ts_runner_oracle.mjs

"use strict";

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import os from "node:os";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PANCAKE_PRODUCTION_ROOT =
  process.env.PANCAKE_PRODUCTION_ROOT
    ? path.resolve(process.env.PANCAKE_PRODUCTION_ROOT)
    : path.resolve(__dirname, "../../../../pancake-production");

if (!fs.existsSync(path.join(PANCAKE_PRODUCTION_ROOT, "lib/evidence-runner/runner.ts"))) {
  console.error(`pancake-production not found at ${PANCAKE_PRODUCTION_ROOT}`);
  process.exit(2);
}

const tsxBin = path.join(PANCAKE_PRODUCTION_ROOT, "node_modules", ".bin", "tsx");
if (!fs.existsSync(tsxBin)) {
  console.error(`tsx not found at ${tsxBin}; run \`npm install\` in pancake-production first.`);
  process.exit(2);
}

// -----------------------------------------------------------------------------
// Schema (matches runner.test.ts L84-92 + L48-56)
// -----------------------------------------------------------------------------

const SCHEMA_COLUMNS = [
  { name: "mkt",     type: "string", semantic_role: "market_link" },
  { name: "dec_ts",  type: "int",    semantic_role: "decision_time" },
  { name: "res_ts",  type: "int",    semantic_role: "resolution_time" },
  { name: "price",   type: "number", semantic_role: "entry_price", range: [0, 1] },
  { name: "outcome", type: "int",    semantic_role: "resolved_outcome_numeric" },
  { name: "alpha",   type: "number", semantic_role: "feature" },
  { name: "target",  type: "int",    semantic_role: "feature" },
];

function makeRawSpec({ side, sizing_value = 0.1, slip_bps = 0, fee_bps = 0,
                       entry_when = null, yes_payoff_when = null,
                       starting_capital = 1000 }) {
  return {
    spec_family: "pancake-evidence-spec",
    spec_version: "0.1",
    name: "ts-oracle-fixture",
    evidence_dataset_id: "ev_oracle",
    schema_requirements: { required_columns: SCHEMA_COLUMNS },
    strategy: {
      side,
      entry: { when: entry_when || { feature: "alpha", gte: 2.0 } },
      yes_payoff: {
        when: yes_payoff_when || { feature_equal: { a: "target", b: "outcome" } },
      },
      sizing: { mode: "fixed_fraction", value: sizing_value },
    },
    costs: { slippage_bps: slip_bps, fee_bps },
    starting_capital,
  };
}

function row({ mkt, dec_ts, res_ts, price, outcome, alpha = 3.0, target = 1 }) {
  return { mkt, dec_ts, res_ts, price, outcome, alpha, target };
}

// -----------------------------------------------------------------------------
// Five fixtures
// -----------------------------------------------------------------------------

const FIXTURES = [
  // 1. Simple 1-trade YES win, no fees, no slip. Expected: full match.
  {
    name: "single_yes_win_clean",
    raw_spec: makeRawSpec({ side: "YES", sizing_value: 0.1 }),
    rows: [row({ mkt: "m/A", dec_ts: 100, res_ts: 200, price: 0.5, outcome: 1 })],
    observation_now_sec: 300,
    expected: "match",
    divergence_notes: null,
  },
  // 2. NO side at 0.96 — the v1.3 regression case (the famous one).
  //    No overlap, no fees. Expected: full match.
  {
    name: "no_side_at_0_96",
    raw_spec: makeRawSpec({ side: "NO", sizing_value: 0.1 }),
    rows: [row({ mkt: "m/A", dec_ts: 100, res_ts: 200, price: 0.96, outcome: 0 })],
    observation_now_sec: 300,
    expected: "match",
    divergence_notes: null,
  },
  // 3. Three sequential trades, no overlap, no fees. Expected: full match.
  {
    name: "three_sequential_trades",
    raw_spec: makeRawSpec({ side: "YES", sizing_value: 0.1 }),
    rows: [
      row({ mkt: "m/A", dec_ts: 100,    res_ts: 200,    price: 0.5,  outcome: 1 }),
      row({ mkt: "m/A", dec_ts: 300,    res_ts: 400,    price: 0.42, outcome: 1 }),
      row({ mkt: "m/B", dec_ts: 500,    res_ts: 600,    price: 0.25, outcome: 0,
            alpha: 2.5, target: 1 }),  // loses
    ],
    observation_now_sec: 800,
    expected: "match",
    divergence_notes: null,
  },
  // 4. Cash-leak demonstration: A decides T=100 (sizing 0.9 of starting),
  //    B decides T=200 BEFORE A resolves at T=1000.
  //    Engine 0.3 prevents future cash leak; TS does not.
  //    Expected: documented_divergence (D-1).
  {
    name: "cash_leak_overlapping",
    raw_spec: makeRawSpec({ side: "YES", sizing_value: 0.9 }),
    rows: [
      row({ mkt: "m/A", dec_ts: 100, res_ts: 1000, price: 0.5, outcome: 1 }),
      row({ mkt: "m/B", dec_ts: 200, res_ts: 2000, price: 0.5, outcome: 1 }),
    ],
    observation_now_sec: 3000,
    expected: "documented_divergence",
    divergence_notes: "D-1 cash-leak: TS settles A at T=1000 before processing B's T=200 decision, "
                    + "so B sees post-settle cash. Engine 0.3 enforces event-time ledger.",
  },
  // 5. Fee realized at entry: non-zero fee_bps. Engine 0.3 marks at
  //    shares × entry_fill_price = notional − fee (fee realized at entry).
  //    TS implicitly defers fee to resolution.
  //    Expected: documented_divergence (D-11).
  {
    name: "fee_realized_at_entry",
    raw_spec: makeRawSpec({ side: "YES", sizing_value: 0.1, fee_bps: 100 }),  // 1% fee
    rows: [row({ mkt: "m/A", dec_ts: 100, res_ts: 200, price: 0.5, outcome: 1 })],
    observation_now_sec: 300,
    expected: "documented_divergence",
    divergence_notes: "D-11 fee realization: TS marks at cost; Engine 0.3 marks at "
                    + "shares × entry_fill_price (fee realized at entry).",
  },
];

// -----------------------------------------------------------------------------
// TS shim
// -----------------------------------------------------------------------------

const shimTs = `
import { compileEvidenceSpec } from "@/lib/spec/evidence-spec/compile";
import { runEvidenceBacktest } from "@/lib/evidence-runner/runner";

const fixtures = ${JSON.stringify(FIXTURES)};

const out = fixtures.map((f) => {
  const compileResult = compileEvidenceSpec(f.raw_spec);
  if (!compileResult.ok) {
    return { name: f.name, error: "compile_failed", errors: compileResult.errors };
  }
  const compiledSpec = compileResult.spec;

  const dataset = {
    id: "ds_oracle",
    owner_id: "00000000-0000-0000-0000-000000000001",
    created_at: "2026-05-22T00:00:00.000Z",
    schema: { columns: ${JSON.stringify(SCHEMA_COLUMNS)} },
    schema_sha256: "0".repeat(64),
    storage_mode: "inline",
    rows_inline: f.rows,
    rows_sha256: "0".repeat(64),
    row_count: f.rows.length,
    storage_bytes: 1024,
    ts_decision_min: f.rows.length > 0 ? Math.min(...f.rows.map((r) => r.dec_ts)) : null,
    ts_decision_max: f.rows.length > 0 ? Math.max(...f.rows.map((r) => r.dec_ts)) : null,
    ts_resolved_min: f.rows.length > 0 ? Math.min(...f.rows.map((r) => r.res_ts)) : null,
    ts_resolved_max: f.rows.length > 0 ? Math.max(...f.rows.map((r) => r.res_ts)) : null,
    provenance: { source_urls: ["https://example.test/oracle"] },
    transformations: [],
    derived_from: null,
    visibility: "private",
  };

  let result;
  try {
    result = runEvidenceBacktest(compiledSpec, dataset, {
      observation_now_ms: f.observation_now_sec * 1000,
      now_ms: () => 0,
    });
  } catch (e) {
    return { name: f.name, error: "runtime_failed", message: e.message };
  }

  return {
    name: f.name,
    raw_spec: f.raw_spec,
    rows: f.rows,
    observation_now_sec: f.observation_now_sec,
    expected: f.expected,
    divergence_notes: f.divergence_notes,
    ts_result: {
      compiled_spec_hash: result.compiled_spec_hash,
      metrics: result.metrics,
      equity_curve: result.equity_curve,
      drawdown_curve: result.drawdown_curve,
      monthly_returns: result.monthly_returns,
      trades: result.trades,
      validation: result.validation,
      evidence_runner_version: result.evidence_runner_version,
    },
  };
});

process.stdout.write(JSON.stringify(out, null, 2));
`;

const tmpFile = path.join(os.tmpdir(), `pancake_engine_py_ts_runner_oracle_${process.pid}.ts`);
fs.writeFileSync(tmpFile, shimTs, "utf8");

let stdout;
try {
  stdout = execFileSync(tsxBin, [tmpFile], {
    cwd: PANCAKE_PRODUCTION_ROOT,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "inherit"],
    maxBuffer: 16 * 1024 * 1024,
  });
} finally {
  try { fs.unlinkSync(tmpFile); } catch { /* best-effort */ }
}

const results = JSON.parse(stdout);
const outPath = path.join(__dirname, "ts_runner_expected.json");
fs.writeFileSync(outPath, JSON.stringify(results, null, 2) + "\n", "utf8");

console.log(`wrote ${results.length} fixtures to ${outPath}`);
console.log(`  pancake-production root: ${PANCAKE_PRODUCTION_ROOT}`);
for (const r of results) {
  if (r.error) {
    console.log(`  ${r.name}: ERROR ${r.error}: ${JSON.stringify(r.errors || r.message)}`);
  } else {
    const m = r.ts_result.metrics;
    console.log(`  ${r.name} [${r.expected}]: trades=${m.num_trades} total_return=${m.total_return.toFixed(4)} `
              + `sharpe=${m.sharpe?.toFixed(2)} max_dd=${m.max_drawdown.toFixed(4)}`);
  }
}
