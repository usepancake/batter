#!/usr/bin/env node
// Pancake Engine 0.3 — TS walk-forward parity oracle.
//
// TS has no walk-forward. This oracle proves PER-FOLD runner parity: slice a
// single fixture into 3 hand-computed test windows, invoke the real TS
// `runEvidenceBacktest` on each slice, and emit the per-fold expected outputs.
//
// The Python test `tests/test_walkforward_ts_parity.py` runs the same fixture
// through `run_walkforward` and asserts each fold's metrics match TS within 1e-9.
// Aggregate WF metrics have no TS counterpart — those are Engine-0.3 native and
// pinned via the committed `aggregate_result_hash` in `examples/btc_pred_hedge/expected_result.json`.
//
// Usage:
//   node tests/fixtures/runner/ts_walkforward_oracle.mjs

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
  console.error(`tsx not found at ${tsxBin}`);
  process.exit(2);
}

const SCHEMA_COLUMNS = [
  { name: "mkt",     type: "string", semantic_role: "market_link" },
  { name: "dec_ts",  type: "int",    semantic_role: "decision_time" },
  { name: "res_ts",  type: "int",    semantic_role: "resolution_time" },
  { name: "price",   type: "number", semantic_role: "entry_price", range: [0, 1] },
  { name: "outcome", type: "int",    semantic_role: "resolved_outcome_numeric" },
  { name: "alpha",   type: "number", semantic_role: "feature" },
  { name: "target",  type: "int",    semantic_role: "feature" },
];

// 30-day-fold fixture: 90 rows, one per day, all trades resolve inside their fold.
const DAY = 86_400;
const ROWS = [];
for (let i = 0; i < 90; i++) {
  ROWS.push({
    mkt: `m/W${i}`,
    dec_ts: i * DAY,
    res_ts: i * DAY + 12 * 3600,
    price: 0.5,
    outcome: 1,
    alpha: 3.0,
    target: 1,
  });
}

const RAW_SPEC = {
  spec_family: "pancake-evidence-spec",
  spec_version: "0.1",
  name: "ts-wf-fixture",
  evidence_dataset_id: "ev_wf",
  schema_requirements: { required_columns: SCHEMA_COLUMNS },
  strategy: {
    side: "YES",
    entry: { when: { feature: "alpha", gte: 2.0 } },
    yes_payoff: { when: { feature_equal: { a: "target", b: "outcome" } } },
    sizing: { mode: "fixed_fraction", value: 0.05 },
  },
  costs: { slippage_bps: 0, fee_bps: 0 },
  starting_capital: 10000,
};

// 3 folds: [0, 30d), [30d, 60d), [60d, 90d) — by decision_time.
const FOLD_WINDOWS = [
  [0, 30 * DAY],
  [30 * DAY, 60 * DAY],
  [60 * DAY, 90 * DAY],
];

const shimTs = `
import { compileEvidenceSpec } from "@/lib/spec/evidence-spec/compile";
import { runEvidenceBacktest } from "@/lib/evidence-runner/runner";

const rawSpec = ${JSON.stringify(RAW_SPEC)};
const allRows = ${JSON.stringify(ROWS)};
const foldWindows = ${JSON.stringify(FOLD_WINDOWS)};

const compileResult = compileEvidenceSpec(rawSpec);
if (!compileResult.ok) {
  process.stderr.write("compile failed: " + JSON.stringify(compileResult.errors));
  process.exit(1);
}
const compiledSpec = compileResult.spec;

const folds = foldWindows.map(([start, end], idx) => {
  // Slice by decision_time (matches Engine 0.3 schedule semantics)
  const rows = allRows.filter((r) => r.dec_ts >= start && r.dec_ts < end);
  const dataset = {
    id: "ds_wf_oracle",
    owner_id: "00000000-0000-0000-0000-000000000001",
    created_at: "2026-05-22T00:00:00.000Z",
    schema: { columns: ${JSON.stringify(SCHEMA_COLUMNS)} },
    schema_sha256: "0".repeat(64),
    storage_mode: "inline",
    rows_inline: rows,
    rows_sha256: "0".repeat(64),
    row_count: rows.length,
    storage_bytes: 1024,
    ts_decision_min: rows.length > 0 ? Math.min(...rows.map((r) => r.dec_ts)) : null,
    ts_decision_max: rows.length > 0 ? Math.max(...rows.map((r) => r.dec_ts)) : null,
    ts_resolved_min: rows.length > 0 ? Math.min(...rows.map((r) => r.res_ts)) : null,
    ts_resolved_max: rows.length > 0 ? Math.max(...rows.map((r) => r.res_ts)) : null,
    provenance: { source_urls: ["https://example.test/wf"] },
    transformations: [],
    derived_from: null,
    visibility: "private",
  };
  const result = runEvidenceBacktest(compiledSpec, dataset, {
    observation_now_ms: 200 * ${DAY} * 1000,
    now_ms: () => 0,
  });
  return {
    fold_index: idx,
    test_window: [start, end],
    ts_result: {
      metrics: result.metrics,
      trades: result.trades,
      equity_curve: result.equity_curve,
    },
  };
});

process.stdout.write(JSON.stringify({ raw_spec: rawSpec, rows: allRows, fold_windows: foldWindows, folds }, null, 2));
`;

const tmpFile = path.join(os.tmpdir(), `ts_wf_oracle_${process.pid}.ts`);
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
  try { fs.unlinkSync(tmpFile); } catch { /* */ }
}

const data = JSON.parse(stdout);
const outPath = path.join(__dirname, "ts_walkforward_expected.json");
fs.writeFileSync(outPath, JSON.stringify(data, null, 2) + "\n", "utf8");

console.log(`wrote ${data.folds.length} folds to ${outPath}`);
for (const f of data.folds) {
  const m = f.ts_result.metrics;
  console.log(`  fold ${f.fold_index} window=${JSON.stringify(f.test_window)}: `
            + `trades=${m.num_trades} total_return=${m.total_return.toFixed(4)} `
            + `max_dd=${m.max_drawdown.toFixed(4)}`);
}
