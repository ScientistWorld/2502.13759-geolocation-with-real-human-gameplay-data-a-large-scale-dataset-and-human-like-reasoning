#!/usr/bin/env python3
"""Evaluate the held-out test slice of saved geolocation prediction files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

# Reuse the top-level evaluator's metric functions — they handle the correct
# denominator semantics (missing predicted coords count as failures).
from eval.evaluate_results import (
    compute_metrics as _compute_metrics,
    method_name as _method_name,
    route_experiment as _route_experiment,
)

SPLIT_NAME = "test"
# 10 images (50% of 20-image sample). Different geographic cluster than train.
SPLIT_IMAGE_BASENAMES = {
    "-1.3268797397613523_36.85028839111328_kenya_3.png",
    "-1.056867003440857_37.20905685424805_kenya_3.png",
    "-3.983148097991944_-80.02982330322266_ecuador_1.png",
    "-1.5168824195861816_-78.71324920654297_ecuador_3.png",
    "-0.3338196575641632_-78.4405746459961_ecuador_1.png",
    "-33.045433044433594_-71.58700561523438_chile_1.png",
    "-26.88873863220215_-69.0302734375_chile_3.png",
    "-22.235746383666992_43.23691940307617_madagascar_1.png",
    "-20.292774200439453_44.2698860168457_madagascar_1.png",
    "-18.555164337158203_43.86051177978516_madagascar_1.png",
}


def in_split(prediction: dict[str, Any]) -> bool:
    image_path = prediction.get("image_path")
    return bool(image_path) and Path(str(image_path)).name in SPLIT_IMAGE_BASENAMES


def evaluate(results_dir: Path, reference_path: Path, scores_path: Path) -> dict[str, Any]:
    pred_files = sorted(results_dir.glob("*_predictions.json"))
    print(f"Found prediction files: {[path.stem.replace('_predictions', '') for path in pred_files]}")
    print(f"Using {SPLIT_NAME} slice with {len(SPLIT_IMAGE_BASENAMES)} image basenames")

    all_results: dict[str, dict[str, float]] = {}
    for path in pred_files:
        pred_name = path.stem.replace("_predictions", "")
        print(f"\nEvaluating {SPLIT_NAME}: {pred_name}")
        predictions = json.loads(path.read_text())
        if not isinstance(predictions, list):
            print(f"  Warning: unexpected format in {path}")
            continue
        split_predictions = [pred for pred in predictions if in_split(pred)]
        valid = [pred for pred in split_predictions if pred.get("predicted_country") is not None]
        valid_gt = [pred for pred in split_predictions if pred.get("ground_truth_country") is not None]
        print(f"  {len(split_predictions)} in split, {len(valid)} valid, {len(valid_gt)} with ground truth")
        if valid_gt:
            metrics = _compute_metrics(valid_gt)
            # Mark slice for traceability
            metrics["_slice"] = SPLIT_NAME
            all_results[pred_name] = metrics
            for key in [
                "city_accuracy",
                "country_accuracy",
                "continent_accuracy",
                "street_1km",
                "city_25km",
                "country_750km",
                "avg_inference_time_s",
            ]:
                if key in metrics:
                    print(f"  {key}: {metrics[key]:.4f}")

    reference = json.loads(reference_path.read_text())
    scores = {"experiments": {}}

    for exp_name, exp_data in reference.get("experiments", {}).items():
        exp_copy = {
            "description": exp_data.get("description", ""),
            "weight": exp_data.get("weight", 0.0),
            "primary_metric": exp_data.get("primary_metric", ""),
            "metrics": exp_data.get("metrics", {}),
            "results": {},
        }
        ref_metrics = list(exp_data.get("metrics", {}).keys())
        for pred_name, metrics in all_results.items():
            if not _route_experiment(exp_name, pred_name):
                continue
            entry = {key: metrics[key] for key in ref_metrics if key in metrics and not key.startswith("_")}
            if entry:
                exp_copy["results"][_method_name(pred_name)] = entry
        if exp_copy["results"]:
            measured_metrics = set()
            for entry in exp_copy["results"].values():
                measured_metrics.update(entry)
            exp_copy["metrics"] = {
                key: value for key, value in exp_data.get("metrics", {}).items() if key in measured_metrics
            }
            scores["experiments"][exp_name] = exp_copy

    scores_path.write_text(json.dumps(scores, indent=2) + "\n")
    print(f"\n{SPLIT_NAME.capitalize()} scores saved to {scores_path}")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/home/user/results")
    parser.add_argument("--reference", default="/home/user/scoring/reference.json")
    parser.add_argument("--scores", default="/home/user/scoring/scores_test.json")
    args = parser.parse_args()
    evaluate(Path(args.results_dir), Path(args.reference), Path(args.scores))


if __name__ == "__main__":
    main()