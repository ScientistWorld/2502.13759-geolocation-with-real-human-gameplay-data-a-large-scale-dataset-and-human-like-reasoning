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

RESULTS_DIR="${1:-/home/user/results}"

echo "=== GeoCoT Evaluation ==="
echo "Evaluating predictions from: $RESULTS_DIR"

python3 << 'EOPY'
import json
import os
import sys

sys.path.insert(0, '/home/user/pylib')
sys.path.insert(0, '/home/user')
from eval.metrics import compute_all_metrics

results_dir = os.environ.get("PREDICTIONS_DIR", "/home/user/results")

pred_files = {}
for f in os.listdir(results_dir):
    if f.endswith("_predictions.json"):
        name = f.replace("_predictions.json", "")
        pred_files[name] = os.path.join(results_dir, f)

print(f"Found prediction files: {list(pred_files.keys())}")

all_results = {}

for name, filepath in sorted(pred_files.items()):
    print(f"\nEvaluating: {name}")
    try:
        with open(filepath) as f:
            predictions = json.load(f)

        if not isinstance(predictions, list):
            print(f"  Warning: unexpected format in {filepath}")
            continue

        valid_preds = [p for p in predictions if p.get('predicted_country') is not None]
        valid_gt = [p for p in valid_preds if p.get('ground_truth_country') is not None]
        print(f"  {len(valid_preds)}/{len(predictions)} valid, {len(valid_gt)} with ground truth")

        if valid_gt:
            metrics = compute_all_metrics(valid_gt)
            all_results[name] = metrics
            for key in ['city_accuracy', 'country_accuracy', 'continent_accuracy',
                        'street_1km', 'city_25km', 'country_750km']:
                if key in metrics:
                    print(f"  {key}: {metrics[key]:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

# Load reference to get the experiment structure
with open('/home/user/scoring/reference.json') as f:
    reference = json.load(f)

# Build scores.json from reference, filling in reproduction results
scores = {"experiments": {}}

for exp_name, exp_data in reference.get("experiments", {}).items():
    exp_copy = {
        "description": exp_data.get("description", ""),
        "weight": exp_data.get("weight", 0.5),
        "primary_metric": exp_data.get("primary_metric", ""),
        "metrics": exp_data.get("metrics", {}),
        "results": {}
    }

    ref_results = exp_data.get("results", {})
    ref_metrics = list(exp_data.get("metrics", {}).keys())

    # Copy reference results (paper values) — preserve placeholders for unrun methods
    for method_name, method_data in ref_results.items():
        entry = {}
        for key, val in method_data.items():
            if key == "type":
                continue  # Don't include type in scores.json
            entry[key] = val
        exp_copy["results"][method_name] = entry

    # Overwrite with reproduction results where available
    for pred_name, metrics in all_results.items():
        # Map prediction file name to method name
        if pred_name == "geocot":
            method_name = "qwen_geocot"
        elif pred_name == "cot":
            method_name = "qwen_cot"
        else:
            method_name = pred_name

        if method_name not in ref_results:
            continue  # Only add methods that exist in reference

        entry = exp_copy["results"][method_name]  # Start with placeholder
        for key in ref_metrics:
            if key in metrics:
                entry[key] = round(metrics[key], 4)
        exp_copy["results"][method_name] = entry

    scores["experiments"][exp_name] = exp_copy

scores_path = '/home/user/scoring/scores.json'
with open(scores_path, 'w') as f:
    json.dump(scores, f, indent=2)
print(f"\nScores saved to {scores_path}")
print(json.dumps(scores, indent=2))
EOPY

echo ""
echo "=== Evaluation Complete ==="
