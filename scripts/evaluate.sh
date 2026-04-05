#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script evaluates predictions from GeoCoT or baseline methods
# against the reference metrics from the paper.
#
# OUTPUT CONTRACT: This script writes scoring/scores.json containing
# a JSON dict nested by experiment, following the structure of reference.json.
#
# Usage:
#   ./evaluate.sh                              # evaluate results/ directory
#   ./evaluate.sh results/geocot_predictions.json   # evaluate specific file

set -e

cd /home/user

mkdir -p scoring results

# Default: look for predictions in results directory
PREDICTIONS_DIR="${1:-/home/user/results}"

echo "=== GeoCoT Evaluation ==="
echo "Evaluating predictions from: $PREDICTIONS_DIR"

# Find prediction files
if [ -f "$PREDICTIONS_DIR" ]; then
    PRED_FILES=("$PREDICTIONS_DIR")
elif [ -d "$PREDICTIONS_DIR" ]; then
    PRED_FILES=("$PREDICTIONS_DIR"/*_predictions.json)
else
    echo "Error: $PREDICTIONS_DIR not found"
    exit 1
fi

# Run evaluation for each predictions file
python3 << EOPY
import json
import os
import sys

sys.path.insert(0, "/home/user")
from eval.metrics import evaluate_predictions

pred_dir = os.environ.get("PREDICTIONS_DIR", "/home/user/results")
pred_files = [f for f in os.listdir(pred_dir) if f.endswith("_predictions.json")]

all_results = {}

for pf in sorted(pred_files):
    name = pf.replace("_predictions.json", "")
    pf_path = os.path.join(pred_dir, pf)
    print(f"\nEvaluating: {name}")

    try:
        metrics = evaluate_predictions(pf_path)
        all_results[name] = metrics

        # Print key metrics
        for key in ["city_accuracy", "country_accuracy", "continent_accuracy",
                    "street_1km", "city_25km", "country_750km"]:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

# Build scores.json structure
scores = {"experiments": {}}

# Primary experiment: Im2GPS3K geolocation
if any("geocot" in name for name in all_results.keys()):
    scores["experiments"]["im2gps3k_geolocation"] = {
        "description": "Im2GPS3K geolocation prediction with GeoCoT",
        "results": {}
    }

    for method_name, metrics in all_results.items():
        entry = {}
        for key, value in metrics.items():
            if isinstance(value, float) and not key.startswith("_"):
                entry[key] = round(value, 4)

        # Normalize method name
        if "geocot" in method_name:
            label = "geocot_local"
        elif "cot" in method_name:
            label = "cot_local"
        else:
            label = method_name

        scores["experiments"]["im2gps3k_geolocation"]["results"][label] = entry

# Save scores
scores_path = "/home/user/scoring/scores.json"
with open(scores_path, "w") as f:
    json.dump(scores, f, indent=2)

print(f"\nScores saved to {scores_path}")
print(json.dumps(scores, indent=2))
EOPY

echo ""
echo "=== Evaluation Complete ==="
