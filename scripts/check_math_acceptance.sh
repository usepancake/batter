#!/bin/sh
# Pancake Engine 0.3 — math acceptance determinism loop.
#
# Runs tests/test_math_acceptance.py in a loop to catch any nondeterminism
# that wouldn't surface in a single pytest run (clock-coupled values, hash-
# order dependence, race conditions, etc.). Engine 0.3 is correctness-first
# and deterministic by contract — any iteration producing a different
# result_hash is a regression.
#
# Usage:
#   scripts/check_math_acceptance.sh
#   RUNS=100 scripts/check_math_acceptance.sh
#   PYTEST=python3 -m pytest scripts/check_math_acceptance.sh

set -eu

cd "$(dirname "$0")/.."

RUNS="${RUNS:-25}"

# Default to the project venv pytest if no explicit binary requested.
if [ -z "${PYTEST:-}" ]; then
    if [ -x ".venv/bin/pytest" ]; then
        PYTEST=".venv/bin/pytest"
    else
        PYTEST="pytest"
    fi
fi

echo "Running tests/test_math_acceptance.py × $RUNS iterations with $PYTEST"

i=1
while [ "$i" -le "$RUNS" ]; do
    out="$($PYTEST -q tests/test_math_acceptance.py 2>&1)" || {
        printf "\n"
        echo "Iteration $i FAILED:"
        printf '%s\n' "$out" | tail -20
        exit 1
    }
    # Confirm the summary line shows passed (catches "no tests collected" etc.)
    if ! printf '%s\n' "$out" | tail -3 | grep -q "passed"; then
        printf "\n"
        echo "Iteration $i did not report passed:"
        printf '%s\n' "$out" | tail -10
        exit 1
    fi
    printf "."
    i=$((i + 1))
done
printf "\n"
echo "OK: $RUNS iterations all passed deterministically"
