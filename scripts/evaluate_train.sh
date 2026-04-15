#!/bin/bash
# Evaluation script for the visible train slice.
# Calls code in eval/train/ and writes scoring/scores_train.json.

set -e

cd /home/user

mkdir -p scoring results

RESULTS_DIR="${1:-/home/user/results}"

echo "=== GeoCoT Train-Slice Evaluation ==="
echo "Evaluating predictions from: $RESULTS_DIR"

python3 -m eval.train.evaluate_results \
    --results-dir "$RESULTS_DIR" \
    --reference /home/user/scoring/reference.json \
    --scores /home/user/scoring/scores_train.json

echo ""
echo "=== Train-Slice Evaluation Complete ==="
