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
                        'country_recall', 'country_f1',
                        'street_1km', 'city_25km', 'country_750km']:
                if key in metrics:
                    print(f"  {key}: {metrics[key]:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

# Build scores.json
scores = {"experiments": {}}

if not all_results:
    print("No valid results to evaluate.")
    sys.exit(0)

# Classification experiment
scores["experiments"]["geocomp_classification"] = {
    "description": "GeoCoT vs CoT on GeoCLIP dataset (4 countries: Kenya, Ecuador, Chile, Madagascar) with Qwen2.5-VL-7B-Instruct. Balanced sample.",
    "weight": 0.5,
    "primary_metric": "country_accuracy",
    "metrics": {
        "country_accuracy": {"higher_is_better": True, "coefficient": 1.0},
        "continent_accuracy": {"higher_is_better": True, "coefficient": 0.8},
        "city_accuracy": {"higher_is_better": True, "coefficient": 1.2}
    },
    "results": {}
}

# Paper reference values
scores["experiments"]["geocomp_classification"]["results"]["gpt4o_cot_paper"] = {
    "type": "baseline",
    "city_accuracy": 0.094,
    "country_accuracy": 0.623,
    "continent_accuracy": 0.819
}
scores["experiments"]["geocomp_classification"]["results"]["geocot_paper"] = {
    "type": "proposed",
    "city_accuracy": 0.118,
    "country_accuracy": 0.640,
    "continent_accuracy": 0.862
}

# Add reproduction results
for method_name, metrics in all_results.items():
    if not isinstance(metrics, dict):
        continue

    label = method_name
    if 'geocot' in method_name:
        method_type = 'proposed'
    else:
        method_type = 'baseline'

    entry = {'type': method_type}
    for key in ['country_accuracy', 'continent_accuracy', 'city_accuracy',
                'country_recall', 'country_f1', 'continent_recall', 'continent_f1',
                'city_recall', 'city_f1']:
        if key in metrics:
            entry[key] = round(metrics[key], 4)

    scores["experiments"]["geocomp_classification"]["results"][label] = entry

# Distance experiment
scores["experiments"]["geocomp_distance"] = {
    "description": "Distance-based accuracy on GeoCLIP (Qwen2.5-VL)",
    "weight": 0.5,
    "primary_metric": "city_25km",
    "metrics": {
        "street_1km": {"higher_is_better": True, "coefficient": 1.0},
        "city_25km": {"higher_is_better": True, "coefficient": 1.0},
        "country_750km": {"higher_is_better": True, "coefficient": 0.8}
    },
    "results": {}
}

scores["experiments"]["geocomp_distance"]["results"]["gpt4o_cot_paper"] = {
    "type": "baseline",
    "street_1km": 0.047,
    "city_25km": 0.151,
    "country_750km": 0.701
}
scores["experiments"]["geocomp_distance"]["results"]["geocot_paper"] = {
    "type": "proposed",
    "street_1km": 0.073,
    "city_25km": 0.157,
    "country_750km": 0.711
}

for method_name, metrics in all_results.items():
    if not isinstance(metrics, dict):
        continue

    label = method_name
    if 'geocot' in method_name:
        method_type = 'proposed'
    else:
        method_type = 'baseline'

    entry = {'type': method_type}
    for key in ['street_1km', 'city_25km', 'country_750km']:
        if key in metrics:
            entry[key] = round(metrics[key], 4)

    scores["experiments"]["geocomp_distance"]["results"][label] = entry

scores_path = '/home/user/scoring/scores.json'
with open(scores_path, 'w') as f:
    json.dump(scores, f, indent=2)
print(f"\nScores saved to {scores_path}")
print(json.dumps(scores, indent=2))
EOPY

echo ""
echo "=== Evaluation Complete ==="
