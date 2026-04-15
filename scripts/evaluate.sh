#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
# Calls code in eval/ and writes scoring/scores.json.

set -e

cd /home/user

mkdir -p scoring results

RESULTS_DIR="${1:-/home/user/results}"

echo "=== GeoCoT Evaluation ==="
echo "Evaluating predictions from: $RESULTS_DIR"

python3 -m eval.evaluate_results \
    --results-dir "$RESULTS_DIR" \
    --reference /home/user/scoring/reference.json \
    --scores /home/user/scoring/scores.json

echo ""
echo "=== Evaluation Complete ==="
