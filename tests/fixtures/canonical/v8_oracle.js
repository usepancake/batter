#!/usr/bin/env node
// Pancake Engine 0.3 — V8 oracle for canonical bytes.
//
// Reads tests/fixtures/canonical/cases.json and writes expected_bytes.json
// containing the canonical UTF-8 string produced by V8's JSON.stringify for
// each case. Python canonicalize() output must byte-equal this oracle.
//
// Engine 0.3 is correctness-first, not TS parity; the V8 oracle is the
// canonical-form baseline.
//
// Usage:
//   node tests/fixtures/canonical/v8_oracle.js

"use strict";

const fs = require("fs");
const path = require("path");

const casesPath = path.join(__dirname, "cases.json");
const outPath = path.join(__dirname, "expected_bytes.json");

const cases = JSON.parse(fs.readFileSync(casesPath, "utf8"));
const expected = cases.map((c) => {
  let canonical;
  try {
    canonical = JSON.stringify(c.value);
    // JSON.stringify returns undefined for NaN/Infinity; we never expect those
    // in the fixture (Python rejects them), but defend anyway.
    if (canonical === undefined) {
      canonical = null;
    }
  } catch (e) {
    canonical = null;
  }
  return { name: c.name, value: c.value, canonical };
});

fs.writeFileSync(outPath, JSON.stringify(expected, null, 2) + "\n");
console.log(`wrote ${expected.length} cases to ${outPath}`);
