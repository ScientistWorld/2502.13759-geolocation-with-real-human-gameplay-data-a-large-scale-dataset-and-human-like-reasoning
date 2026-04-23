#!/usr/bin/env python3
"""Evaluate the visible train slice of saved geolocation prediction files."""

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

SPLIT_NAME = "train"
# 10 images (50% of 20-image sample). Split by geographic clustering to
# ensure both slices cover all four countries.
SPLIT_IMAGE_BASENAMES = {
    "-3.990187883377075_39.57994842529297_kenya_1.png",
    "-1.223993897438049_35.262813568115234_kenya_1.png",
    "-0.5329132676124573_34.72760391235352_kenya_1.png",
    "-2.156399965286255_-79.83031463623047_ecuador_4.png",
    "-1.058655023574829_-80.45985412597656_ecuador_2.png",
    "-33.37306213378906_-70.69071960449219_chile_1.png",
    "-30.159818649291992_-71.37037658691406_chile_4.png",
    "-22.95543098449707_-70.28689575195312_chile_2.png",
    "-20.299379348754883_44.26924514770508_madagascar_3.png",
    "-18.925765991210938_47.53049087524414_madagascar_3.png",
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
    parser.add_argument("--scores", default="/home/user/scoring/scores_train.json")
    args = parser.parse_args()
    evaluate(Path(args.results_dir), Path(args.reference), Path(args.scores))


if __name__ == "__main__":
    main()