#!/bin/bash
# Baseline script — runs CoT baseline for geolocation
#
# This script runs the baseline method (standard chain-of-thought without
# GeoCoT-structured geographical reasoning) on the geolocation dataset.

set -e

cd /home/user

RESULTS_DIR="/home/user/results"
mkdir -p "$RESULTS_DIR"

# Use the same VLM type as method.sh (LLaVA-1.5-7B)
VLM_TYPE="llava"
MAX_IMAGES=10

echo "=========================================="
echo "Baseline: Standard CoT on GeoCLIP dataset"
echo "=========================================="

DATASET="/home/user/data/geoclip/geoclip.csv"
OUTPUT_PATH="$RESULTS_DIR/cot_baseline_predictions.json"

echo "Dataset: $DATASET"
echo "VLM Type: $VLM_TYPE"
echo "Max Images: $MAX_IMAGES"
echo "Output: $OUTPUT_PATH"

python3 << 'EOF'
import sys
sys.path.insert(0, '/home/user')
from baseline.llm_cot_baseline import run_baseline

results = run_baseline(
    dataset_path="/home/user/data/geoclip/geoclip.csv",
    output_path="/home/user/results/cot_baseline_predictions.json",
    vlm_type="llava",
    max_images=10,
)

print(f"\nGenerated {len(results)} predictions")
print(f"Saved to /home/user/results/cot_baseline_predictions.json")
EOF

# Generate scores
echo ""
echo "=== Evaluating baseline results ==="

python3 << 'EOF'
import sys
sys.path.insert(0, '/home/user')
import json
import os
from eval.metrics import evaluate_predictions, compute_all_metrics

pred_path = "/home/user/results/cot_baseline_predictions.json"
if os.path.exists(pred_path):
    with open(pred_path, "r") as f:
        predictions = json.load(f)

    metrics = compute_all_metrics(predictions)
    print("\nBaseline Metrics:")
    for key in ["city_accuracy", "country_accuracy", "continent_accuracy"]:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")

    # Save as separate file for evaluate.sh to pick up
    scores_path = "/home/user/results/cot_baseline_metrics.json"
    with open(scores_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {scores_path}")
else
    echo "No predictions found - baseline may have failed"
EOF

echo ""
echo "=========================================="
echo "Baseline Complete - $OUTPUT_PATH generated"
echo "=========================================="
