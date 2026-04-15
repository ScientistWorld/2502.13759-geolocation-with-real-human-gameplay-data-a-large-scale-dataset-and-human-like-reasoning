#!/bin/bash
# Evaluation script for the held-out test slice.
# Calls code in eval/test/ and writes scoring/scores_test.json.

set -e

cd /home/user

mkdir -p scoring results

RESULTS_DIR="${1:-/home/user/results}"

echo "=== GeoCoT Test-Slice Evaluation ==="
echo "Evaluating predictions from: $RESULTS_DIR"

python3 -m eval.test.evaluate_results \
    --results-dir "$RESULTS_DIR" \
    --reference /home/user/scoring/reference.json \
    --scores /home/user/scoring/scores_test.json

echo ""
echo "=== Test-Slice Evaluation Complete ==="
