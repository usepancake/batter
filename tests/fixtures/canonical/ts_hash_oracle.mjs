#!/usr/bin/env node
// Pancake Engine 0.3 — TS hash oracle.
//
// Engine 0.3 is correctness-first, not TS parity. But for the substrate
// (canonicalization + SHA-256), Python `sha256_canonical` must byte-equal
// the TS evidence-runner's `hashSchema()` / `hashRows()` on equivalent
// EvidenceDataset content. This oracle grounds the parity test in the
// real TS code from pancake-production, not a hand-port.
//
// Usage (run from inside pancake-engine-py):
//   node tests/fixtures/canonical/ts_hash_oracle.mjs
//   PANCAKE_PRODUCTION_ROOT=/abs/path/to/pancake-production \
//     node tests/fixtures/canonical/ts_hash_oracle.mjs
//
// Mechanics:
//   1. Defines 5 small EvidenceDataset fixtures inline (schema + rows lifted
//      from pancake-production/tests/evidence-runner/runner.test.ts shape).
//   2. Writes a temp .ts shim that imports `@/lib/data/evidence/hash` from
//      pancake-production and calls `hashSchema` / `hashRows` on each.
//   3. Runs the shim via `tsx` with cwd = pancake-production so the `@/`
//      path alias resolves through pancake-production's tsconfig.
//   4. Captures the JSON output and writes `ts_hashes.json` next to this file.
//
// The committed `ts_hashes.json` is what `tests/test_ts_hash_parity.py` asserts
// Python's `sha256_canonical` matches. Regenerate this file only when fixtures
// change or when pancake-production's canonicalize/hash code changes —
// drift will be visible as different bytes in ts_hashes.json.

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

if (!fs.existsSync(path.join(PANCAKE_PRODUCTION_ROOT, "lib/data/evidence/hash.ts"))) {
  console.error(
    `pancake-production not found at ${PANCAKE_PRODUCTION_ROOT}\n` +
    `Set PANCAKE_PRODUCTION_ROOT to the absolute path of your pancake-production checkout.`
  );
  process.exit(2);
}

const tsxBin = path.join(PANCAKE_PRODUCTION_ROOT, "node_modules", ".bin", "tsx");
if (!fs.existsSync(tsxBin)) {
  console.error(
    `tsx not found at ${tsxBin}\n` +
    `Run \`npm install\` inside pancake-production first.`
  );
  process.exit(2);
}

// 5 EvidenceDataset fixtures.
//
// Schema is the canonical TS runner-test schema (mkt / dec_ts / res_ts /
// price / outcome / alpha / target). Row sets vary across fixtures to
// exercise different content shapes.

const SCHEMA = {
  columns: [
    { name: "mkt",     type: "string", semantic_role: "market_link" },
    { name: "dec_ts",  type: "int",    semantic_role: "decision_time" },
    { name: "res_ts",  type: "int",    semantic_role: "resolution_time" },
    { name: "price",   type: "number", semantic_role: "entry_price", range: [0, 1] },
    { name: "outcome", type: "int",    semantic_role: "resolved_outcome_numeric" },
    { name: "alpha",   type: "number", semantic_role: "feature" },
    { name: "target",  type: "int",    semantic_role: "feature" },
  ],
};

const FIXTURES = [
  {
    name: "single_row_yes_resolved",
    schema: SCHEMA,
    rows: [
      { mkt: "m/A", dec_ts: 1_700_000_000, res_ts: 1_700_100_000,
        price: 0.5, outcome: 1, alpha: 2.5, target: 1 },
    ],
  },
  {
    name: "single_row_no_side_at_0_96",
    schema: SCHEMA,
    rows: [
      { mkt: "m/B", dec_ts: 1_700_200_000, res_ts: 1_700_300_000,
        price: 0.96, outcome: 0, alpha: 3.0, target: 0 },
    ],
  },
  {
    name: "three_rows_mixed_outcomes",
    schema: SCHEMA,
    rows: [
      { mkt: "m/C", dec_ts: 1_700_400_000, res_ts: 1_700_500_000,
        price: 0.42, outcome: 1, alpha: 2.1, target: 1 },
      { mkt: "m/C", dec_ts: 1_700_600_000, res_ts: 1_700_700_000,
        price: 0.58, outcome: 0, alpha: 2.7, target: 0 },
      { mkt: "m/D", dec_ts: 1_700_800_000, res_ts: 1_700_900_000,
        price: 0.20, outcome: 1, alpha: 4.0, target: 1 },
    ],
  },
  {
    name: "fractional_alpha_values",
    schema: SCHEMA,
    rows: [
      { mkt: "m/E", dec_ts: 1_701_000_000, res_ts: 1_701_100_000,
        price: 0.125, outcome: 0, alpha: 0.1, target: 0 },
      { mkt: "m/E", dec_ts: 1_701_200_000, res_ts: 1_701_300_000,
        price: 0.875, outcome: 1, alpha: 0.30000000000000004, target: 1 },
    ],
  },
  {
    name: "five_rows_chronological",
    schema: SCHEMA,
    rows: [
      { mkt: "m/F", dec_ts: 1_702_000_000, res_ts: 1_702_086_400,
        price: 0.10, outcome: 0, alpha: 1.0, target: 0 },
      { mkt: "m/F", dec_ts: 1_702_172_800, res_ts: 1_702_259_200,
        price: 0.25, outcome: 1, alpha: 2.0, target: 1 },
      { mkt: "m/G", dec_ts: 1_702_345_600, res_ts: 1_702_432_000,
        price: 0.50, outcome: 1, alpha: 3.0, target: 1 },
      { mkt: "m/G", dec_ts: 1_702_518_400, res_ts: 1_702_604_800,
        price: 0.75, outcome: 0, alpha: 4.0, target: 0 },
      { mkt: "m/H", dec_ts: 1_702_691_200, res_ts: 1_702_777_600,
        price: 0.90, outcome: 1, alpha: 5.0, target: 1 },
    ],
  },
];

// Build the temp TS shim. The shim runs with cwd = PANCAKE_PRODUCTION_ROOT,
// so the `@/lib/...` path alias resolves through pancake-production's tsconfig.

const shimTs = `
import { hashSchema, hashRows } from "@/lib/data/evidence/hash";

const fixtures = ${JSON.stringify(FIXTURES)};
const out = fixtures.map((f) => ({
  name: f.name,
  schema_sha256: hashSchema(f.schema),
  rows_sha256: hashRows(f.rows),
}));
process.stdout.write(JSON.stringify(out));
`;

const tmpFile = path.join(os.tmpdir(), `pancake_engine_py_ts_oracle_${process.pid}.ts`);
fs.writeFileSync(tmpFile, shimTs, "utf8");

let stdout;
try {
  stdout = execFileSync(tsxBin, [tmpFile], {
    cwd: PANCAKE_PRODUCTION_ROOT,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "inherit"],
  });
} finally {
  try { fs.unlinkSync(tmpFile); } catch { /* best-effort */ }
}

const hashes = JSON.parse(stdout);

// Round-trip the canonical JSON form so the committed file is itself canonical-ish.
const outPath = path.join(__dirname, "ts_hashes.json");
const payload = hashes.map((h, i) => ({
  name: FIXTURES[i].name,
  schema: FIXTURES[i].schema,
  rows: FIXTURES[i].rows,
  schema_sha256: h.schema_sha256,
  rows_sha256: h.rows_sha256,
}));
fs.writeFileSync(outPath, JSON.stringify(payload, null, 2) + "\n", "utf8");

console.log(`wrote ${payload.length} fixtures + TS-computed hashes to ${outPath}`);
console.log(`  pancake-production root: ${PANCAKE_PRODUCTION_ROOT}`);
for (const p of payload) {
  console.log(`  ${p.name}: schema=${p.schema_sha256.slice(0, 12)}…  rows=${p.rows_sha256.slice(0, 12)}…`);
}
