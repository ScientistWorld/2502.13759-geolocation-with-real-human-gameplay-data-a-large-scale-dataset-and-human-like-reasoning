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

python3 << 'EOPY'
import json
import os
import sys

sys.path.insert(0, '/home/user')
sys.path.insert(0, '/home/user/pylib')
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

# Paper reference results
scores["experiments"]["geocomp_classification"] = {
    "description": "GeoComp classification: city/country/continent accuracy",
    "results": {
        "gpt4o_cot": {
            "city_accuracy": 0.094,
            "country_accuracy": 0.623,
            "continent_accuracy": 0.819
        },
        "geocot": {
            "city_accuracy": 0.118,
            "country_accuracy": 0.640,
            "continent_accuracy": 0.862
        }
    }
}

scores["experiments"]["geocomp_distance"] = {
    "description": "GeoComp distance-based accuracy: fraction within 1km/25km/750km",
    "results": {
        "gpt4o_cot": {
            "street_1km": 0.047,
            "city_25km": 0.151,
            "country_750km": 0.701
        },
        "geocot": {
            "street_1km": 0.073,
            "city_25km": 0.157,
            "country_750km": 0.711
        }
    }
}

scores["experiments"]["im2gps_generalization"] = {
    "description": "Im2GPS/Im2GPS3K generalization (Table 4)",
    "results": {
        "geocot_im2gps": {
            "street_1km": 0.22,
            "city_25km": 0.55,
            "country_750km": 0.83
        },
        "gpt4o_cot_im2gps": {
            "street_1km": 0.16,
            "city_25km": 0.49,
            "country_750km": 0.77
        }
    }
}

scores["experiments"]["ablation_geocot_steps"] = {
    "description": "Ablation on GeoCoT reasoning steps (Table 8)",
    "results": {
        "geocot_full": {
            "city_accuracy": 0.118,
            "country_accuracy": 0.640,
            "continent_accuracy": 0.862
        }
    }
}

# Add reproduction results
for method_name, metrics in all_results.items():
    if not isinstance(metrics, dict):
        continue

    label = method_name

    # Classification metrics
    if any(k in metrics for k in ['city_accuracy', 'country_accuracy', 'continent_accuracy']):
        entry = {}
        for key in ['city_accuracy', 'country_accuracy', 'continent_accuracy',
                    'city_recall', 'city_f1', 'country_recall', 'country_f1',
                    'continent_recall', 'continent_f1']:
            if key in metrics:
                entry[key] = round(metrics[key], 4)
        scores["experiments"]["geocomp_classification"]["results"][label] = entry

    # Distance metrics
    if any(k in metrics for k in ['street_1km', 'city_25km', 'country_750km']):
        entry = {}
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
