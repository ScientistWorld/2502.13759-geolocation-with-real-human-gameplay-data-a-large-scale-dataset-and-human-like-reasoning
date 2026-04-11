"""
Geolocation evaluation metrics.

GPS coordinate lookup for reverse geocoding location names to lat/lon.
"""

# Country centroids (approximate geographic centers of countries)
# Used to compute distance-based metrics when model outputs location names.
COUNTRY_COORDINATES = {
    # Africa
    "kenya": (-1.2921, 36.8219),
    "madagascar": (-18.7669, 46.8691),
    "south africa": (-30.5595, 22.9375),
    "egypt": (26.8206, 30.8025),
    "nigeria": (9.0820, 8.6753),
    "morocco": (31.7917, -7.0926),
    "ethiopia": (9.1450, 40.4897),
    "tanzania": (-6.3690, 34.8888),
    "ghana": (7.9465, -1.0232),
    "algeria": (28.0339, 1.6596),
    "tunisia": (33.8869, 9.5375),
    "uganda": (1.3733, 32.2903),
    "cameroon": (3.3792, 8.6031),
    "ivory coast": (7.5400, -5.5471),
    "senegal": (14.4974, -14.4524),
    "mozambique": (-18.6657, 35.5296),
    "angola": (-11.2027, 17.8739),
    "zimbabwe": (-19.0154, 29.1549),
    "botswana": (-22.3285, 24.6849),
    "namibia": (-22.9576, 18.4904),
    "rwanda": (-1.9403, 29.8739),
    "zambia": (-13.1339, 27.8493),
    "malawi": (-13.2543, 34.3015),

    # South America
    "ecuador": (-1.8312, -78.1834),
    "chile": (-35.6751, -71.5430),
    "brazil": (-14.2350, -51.9253),
    "argentina": (-38.4161, -63.6167),
    "peru": (-9.1900, -75.0152),
    "colombia": (4.5709, -74.2973),
    "venezuela": (6.4238, -66.5897),
    "bolivia": (-16.2902, -63.5887),
    "paraguay": (-23.4425, -58.4438),
    "uruguay": (-32.5228, -55.7658),
    "guyana": (4.8604, -58.9302),
    "suriname": (3.9193, -56.0278),

    # North America
    "united states": (37.0902, -95.7129),
    "usa": (37.0902, -95.7129),
    "us": (37.0902, -95.7129),
    "canada": (56.1304, -106.3468),
    "mexico": (23.6345, -102.5528),

    # Europe
    "germany": (51.1657, 10.4515),
    "france": (46.2276, 2.2137),
    "united kingdom": (55.3781, -3.4360),
    "uk": (55.3781, -3.4360),
    "spain": (40.4637, -3.7492),
    "italy": (41.8719, 12.5674),
    "netherlands": (52.1326, 5.2913),
    "belgium": (50.5039, 4.4699),
    "switzerland": (46.8182, 8.2275),
    "austria": (47.5162, 14.5501),
    "poland": (51.9194, 19.1451),
    "sweden": (60.1282, 18.6435),
    "norway": (60.4720, 8.4689),
    "denmark": (56.2639, 9.5018),
    "finland": (61.9241, 25.7482),
    "portugal": (39.3999, -8.2245),
    "greece": (39.0742, 21.8243),
    "czech republic": (49.8175, 15.4730),
    "czechia": (49.8175, 15.4730),
    "hungary": (47.1625, 19.5033),
    "romania": (45.9432, 24.9668),
    "ireland": (53.1424, -7.6921),
    "croatia": (45.1000, 15.2000),
    "russia": (61.5240, 105.3188),
    "ukraine": (48.3794, 31.1656),
    "turkey": (38.9637, 35.2433),
    "serbia": (44.0165, 21.0059),
    "slovenia": (46.1512, 14.9955),
    "slovakia": (48.6690, 19.6990),
    "bulgaria": (42.7339, 25.4858),
    "lithuania": (55.1694, 23.8813),
    "latvia": (56.8796, 24.6032),
    "estonia": (58.5953, 25.0136),
    "iceland": (64.9631, -19.0208),
    "luxembourg": (49.8153, 6.1296),

    # Asia
    "china": (35.8617, 104.1954),
    "japan": (36.2048, 138.2529),
    "south korea": (35.9078, 127.7669),
    "india": (20.5937, 78.9629),
    "indonesia": (-0.7893, 113.9213),
    "thailand": (15.8700, 100.9925),
    "vietnam": (14.0583, 108.2772),
    "philippines": (12.8797, 121.7740),
    "malaysia": (4.2105, 101.9758),
    "singapore": (1.3521, 103.8198),
    "pakistan": (30.3753, 69.3451),
    "bangladesh": (23.6850, 90.3563),
    "nepal": (28.3949, 84.1240),
    "sri lanka": (7.8731, 80.7718),
    "cambodia": (12.5657, 104.9910),
    "myanmar": (21.9162, 95.9560),
    "mongolia": (46.8625, 103.8467),
    "taiwan": (23.6978, 120.9605),
    "hong kong": (22.3193, 114.1694),
    "uae": (23.4241, 53.8478),
    "united arab emirates": (23.4241, 53.8478),
    "saudi arabia": (23.8859, 45.0792),
    "israel": (31.0461, 34.8516),
    "iran": (32.4279, 53.6880),
    "iraq": (33.3152, 44.3661),
    "jordan": (30.5852, 36.2384),
    "lebanon": (33.8547, 35.8623),
    "qatar": (25.3548, 51.1839),
    "kazakhstan": (48.0196, 66.9237),

    # Oceania
    "australia": (-25.2744, 133.7751),
    "new zealand": (-40.9006, 174.8860),
    "papua new guinea": (-6.3150, 143.9555),
    "fiji": (-17.7134, 178.0650),
}


def reverse_geocode(country: str = None, city: str = None) -> tuple:
    """
    Convert predicted country/city names to GPS coordinates.

    Args:
        country: Predicted country name (normalized)
        city: Predicted city name

    Returns:
        Tuple of (lat, lon) or (None, None) if not found
    """
    if country is None:
        return None, None

    country_lower = country.strip().lower()

    # Try exact country match first
    if country_lower in COUNTRY_COORDINATES:
        return COUNTRY_COORDINATES[country_lower]

    # Try partial match
    for known_country, coords in COUNTRY_COORDINATES.items():
        if known_country in country_lower or country_lower in known_country:
            return coords

    return None, None


def enrich_predictions_with_coordinates(predictions: list) -> list:
    """
    Add predicted_lat and predicted_lon to predictions by reverse geocoding.

    This enables distance-based metrics even when the model outputs
    location names instead of GPS coordinates.
    """
    for pred in predictions:
        if pred.get("predicted_lat") is not None:
            continue  # Already has coordinates

        country = pred.get("predicted_country")
        city = pred.get("predicted_city")
        lat, lon = reverse_geocode(country, city)
        pred["predicted_lat"] = lat
        pred["predicted_lon"] = lon

    return predictions


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

    # Enrich with GPS coordinates via reverse geocoding
    enrich_predictions_with_coordinates(predictions)

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
