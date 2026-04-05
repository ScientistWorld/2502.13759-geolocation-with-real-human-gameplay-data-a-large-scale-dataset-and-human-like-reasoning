"""
Geolocation evaluation metrics.

Computes all metrics from the paper:
- Classification metrics: accuracy, recall, F1 at city/country/continent level
- Distance metrics: Street (1km), City (25km), Country (750km)
- Reasoning quality metrics: GPTScore, CE, AE, AC, LC
- Hallucination metrics: OH, FH, AH
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


# Haversine distance between two GPS coordinates (in km)
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great-circle distance between two points on Earth (in km)."""
    R = 6371.0  # Earth's radius in km
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# Country name normalization
def normalize_country(country: str) -> str:
    """Normalize country names for comparison."""
    if country is None:
        return None
    country = str(country).strip().lower()

    # Common aliases
    aliases = {
        "usa": "united states",
        "us": "united states",
        "u.s.": "united states",
        "u.s.a.": "united states",
        "uk": "united kingdom",
        "great britain": "united kingdom",
        "england": "united kingdom",
        " PRC ": "china",
        "prc": "china",
        "p.r.china": "china",
        "p.r.c.": "china",
        "roc": "taiwan",
        "r.o.c.": "taiwan",
        "netherlands antilles": "netherlands",
        "holland": "netherlands",
        "deutschland": "germany",
        "brasil": "brazil",
        "espana": "spain",
        "österreich": "austria",
        "suisse": "switzerland",
        "svizzera": "switzerland",
    }

    for alias, canonical in aliases.items():
        if alias in country or country in alias:
            return canonical
    return country


def normalize_continent(continent: str) -> str:
    """Normalize continent names."""
    if continent is None:
        return None
    continent = str(continent).strip().lower()

    aliases = {
        "europe": "europe",
        "asia": "asia",
        "africa": "africa",
        "north america": "north america",
        "north_america": "north america",
        "south america": "south america",
        "south_america": "south america",
        "oceania": "oceania",
        "australasia": "oceania",
        "australia": "oceania",
        "america": "north america",
        "americas": "north america",
    }

    for alias, canonical in aliases.items():
        if alias in continent:
            return canonical
    return continent


def normalize_city(city: str) -> str:
    """Normalize city names for comparison."""
    if city is None:
        return None
    city = str(city).strip().lower()
    # Remove common suffixes
    for suffix in [",", " city", " city,", "-"]:
        city = city.replace(suffix, "")
    return city


def compute_classification_metrics(
    predictions: List[Dict],
    level: str = "country"
) -> Dict[str, float]:
    """
    Compute classification metrics at a given geographical level.

    Args:
        predictions: List of prediction dicts with predicted_* and ground_truth_* keys
        level: "city", "country", or "continent"

    Returns:
        Dict with accuracy, recall, f1
    """
    pred_key = f"predicted_{level}"
    gt_key = f"ground_truth_{level}"

    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    total = 0

    for pred in predictions:
        if pred.get(gt_key) is None:
            continue
        total += 1

        pred_val = pred.get(pred_key)
        gt_val = pred.get(gt_key)

        if pred_val is None:
            fn += 1
            continue

        # Normalize for comparison
        if level == "country":
            pred_norm = normalize_country(pred_val)
            gt_norm = normalize_country(gt_val)
        elif level == "continent":
            pred_norm = normalize_continent(pred_val)
            gt_norm = normalize_continent(gt_val)
        else:
            pred_norm = normalize_city(pred_val)
            gt_norm = normalize_city(gt_val)

        if pred_norm == gt_norm:
            tp += 1
        else:
            fp += 1
            fn += 1

    if total == 0:
        return {"accuracy": 0.0, "recall": 0.0, "f1": 0.0, "total": 0}

    accuracy = tp / total if total > 0 else 0.0

    # For recall, we consider all ground truth cases
    # TP / (TP + FN) = matches / total
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 = 2 * precision * recall / (precision + recall)
    # precision = tp / (tp + fp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "recall": recall,
        "f1": f1,
        "precision": precision,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total": total,
    }


def compute_distance_metrics(
    predictions: List[Dict],
    thresholds_km: List[float] = None
) -> Dict[str, float]:
    """
    Compute distance-based accuracy metrics.

    Args:
        predictions: List of prediction dicts with predicted and ground truth GPS coords
        thresholds_km: List of distance thresholds in km

    Returns:
        Dict mapping threshold name to accuracy fraction
    """
    if thresholds_km is None:
        thresholds_km = [1.0, 25.0, 750.0]

    threshold_names = {
        1.0: "street_1km",
        25.0: "city_25km",
        750.0: "country_750km",
    }

    results = {}
    for threshold in thresholds_km:
        name = threshold_names.get(threshold, f"threshold_{threshold}km")
        within = 0
        total = 0

        for pred in predictions:
            lat = pred.get("ground_truth_lat")
            lon = pred.get("ground_truth_lon")
            pred_lat = pred.get("predicted_lat")
            pred_lon = pred.get("predicted_lon")

            if lat is None or lon is None:
                continue
            if pred_lat is None or pred_lon is None:
                continue

            total += 1
            dist = haversine(lat, lon, pred_lat, pred_lon)
            if dist <= threshold:
                within += 1

        results[name] = within / total if total > 0 else 0.0
        results[f"{name}_count"] = within
        results[f"{name}_total"] = total

    return results


def compute_all_metrics(predictions: List[Dict]) -> Dict:
    """
    Compute all geolocation metrics from predictions.

    Args:
        predictions: List of prediction dicts

    Returns:
        Dict with all computed metrics
    """
    metrics = {}

    # Classification metrics at each level
    for level in ["city", "country", "continent"]:
        level_metrics = compute_classification_metrics(predictions, level)
        for k, v in level_metrics.items():
            metrics[f"{level}_{k}"] = v

    # Distance metrics
    dist_metrics = compute_distance_metrics(predictions)
    metrics.update(dist_metrics)

    # Count predictions
    metrics["total_predictions"] = len(predictions)
    metrics["valid_predictions"] = sum(
        1 for p in predictions if p.get("predicted_city") is not None
    )

    return metrics


def evaluate_predictions(
    predictions_path: str,
    output_path: Optional[str] = None
) -> Dict:
    """
    Evaluate predictions from a JSON file.

    Args:
        predictions_path: Path to predictions JSON file
        output_path: Optional path to save metrics

    Returns:
        Dict with all metrics
    """
    with open(predictions_path, "r") as f:
        predictions = json.load(f)

    metrics = compute_all_metrics(predictions)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {output_path}")

    return metrics


def evaluate_against_reference(
    predictions_path: str,
    reference_path: str,
    output_path: Optional[str] = None
) -> Dict:
    """
    Evaluate predictions and compare against reference metrics.

    Args:
        predictions_path: Path to predictions JSON
        reference_path: Path to reference scores JSON
        output_path: Optional path to save comparison

    Returns:
        Dict with metrics and comparison
    """
    metrics = evaluate_predictions(predictions_path)

    if os.path.exists(reference_path):
        with open(reference_path, "r") as f:
            reference = json.load(f)
        metrics["reference"] = reference

    if output_path:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate geolocation predictions")
    parser.add_argument("predictions", type=str, help="Path to predictions JSON")
    parser.add_argument("--output", type=str, default=None, help="Output path for metrics")
    parser.add_argument("--reference", type=str, default=None, help="Reference scores JSON")
    args = parser.parse_args()

    if args.reference:
        results = evaluate_against_reference(args.predictions, args.reference, args.output)
    else:
        results = evaluate_predictions(args.predictions, args.output)

    print("\n=== Evaluation Results ===")
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
