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

# Ensure pylib packages (pandas, numpy) are available
export PYTHONPATH="/home/user/pylib:${PYTHONPATH:-}"

RESULTS_DIR="${1:-/home/user/results}"

echo "=== GeoCoT Evaluation ==="
echo "Evaluating predictions from: $RESULTS_DIR"

python3 << 'EOPY'
import json
import os
import sys

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

# Define the mapping from prediction file names to method names
METHOD_MAPPING = {
    "geocot": "qwen_geocot",
    "cot": "qwen_cot",
}

# Define which experiments each prediction type belongs to
# (since prediction files don't know which experiment they belong to)
PRED_TO_EXPERIMENTS = {
    "geocot": ["geocomp_classification", "geocomp_distance"],
    "cot": ["geocomp_classification", "geocomp_distance"],
}

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

    # Copy reference results (paper values)
    for method_name, method_data in ref_results.items():
        entry = {}
        for key, val in method_data.items():
            if key == "type":
                continue
            entry[key] = val
        exp_copy["results"][method_name] = entry

    # Add reproduced results for methods in reference
    for pred_name, metrics in all_results.items():
        method_name = METHOD_MAPPING.get(pred_name, pred_name)
        if method_name in ref_results:
            entry = exp_copy["results"][method_name]
            for key in ref_metrics:
                if key in metrics:
                    entry[key] = round(metrics[key], 4)

    # Add reproduced results for methods NOT in reference (e.g. qwen_geocot vs gpt4o_cot)
    # These methods belong to geocomp_classification and geocomp_distance experiments
    if exp_name in ["geocomp_classification", "geocomp_distance"]:
        for pred_name, metrics in all_results.items():
            method_name = METHOD_MAPPING.get(pred_name, pred_name)
            if method_name not in ref_results:
                # Add as a new method with the experiment's metrics
                entry = {}
                for key in ref_metrics:
                    if key in metrics:
                        entry[key] = round(metrics[key], 4)
                if entry:  # Only add if we have some metrics
                    exp_copy["results"][method_name] = entry

    scores["experiments"][exp_name] = exp_copy

scores_path = '/home/user/scoring/scores.json'
with open(scores_path, 'w') as f:
    json.dump(scores, f, indent=2)
print(f"\nScores saved to {scores_path}")

# Print summary
print("\n=== Summary ===")
for exp_name, exp_data in scores.get("experiments", {}).items():
    print(f"\n{exp_name}:")
    for method, result in exp_data.get("results", {}).items():
        pm = exp_data.get("primary_metric", "")
        val = result.get(pm, "N/A")
        print(f"  {method}: {pm}={val}")
EOPY

echo ""
echo "=== Evaluation Complete ==="
