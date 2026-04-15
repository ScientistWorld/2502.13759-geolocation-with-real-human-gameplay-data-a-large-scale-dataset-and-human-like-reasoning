#!/usr/bin/env python3
"""Evaluate the visible train slice of saved geolocation prediction files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SPLIT_NAME = "train"
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

COUNTRY_COORDS = {
    "kenya": (-1.2921, 36.8219),
    "madagascar": (-18.7669, 46.8691),
    "ecuador": (-1.8312, -78.1834),
    "chile": (-35.6751, -71.5430),
    "brazil": (-14.2350, -51.9253),
    "argentina": (-38.4161, -63.6167),
    "peru": (-9.1900, -75.0152),
    "colombia": (4.5709, -74.2973),
    "united states": (37.0902, -95.7129),
    "canada": (56.1304, -106.3468),
    "mexico": (23.6345, -102.5528),
    "germany": (51.1657, 10.4515),
    "france": (46.2276, 2.2137),
    "united kingdom": (55.3781, -3.4360),
    "china": (35.8617, 104.1954),
    "japan": (36.2048, 138.2529),
    "india": (20.5937, 78.9629),
    "russia": (61.5240, 105.3188),
    "south africa": (-30.5595, 22.9375),
    "egypt": (26.8206, 30.8025),
    "nigeria": (9.0820, 8.6753),
    "australia": (-25.2744, 133.7751),
    "thailand": (15.8700, 100.9925),
    "indonesia": (-0.7893, 113.9213),
    "saudi arabia": (24.7136, 46.6753),
    "turkey": (38.9637, 35.2433),
    "south korea": (35.9078, 127.7669),
    "italy": (41.8719, 12.5674),
    "spain": (40.4637, -3.7492),
    "finland": (61.9241, 25.7482),
    "new zealand": (-40.9006, 174.8860),
    "uganda": (1.3733, 32.2903),
    "tanzania": (-6.3690, 34.8888),
    "ethiopia": (9.1450, 40.4897),
    "ghana": (7.9465, -1.0232),
    "morocco": (31.7917, -7.0926),
    "algeria": (28.0339, 1.6596),
}


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def normalize_country(value: Any) -> str | None:
    if not value:
        return None
    country = str(value).strip().lower()
    aliases = {
        "usa": "united states",
        "us": "united states",
        "u.s.": "united states",
        "uk": "united kingdom",
        "britain": "united kingdom",
        "british": "united kingdom",
        "england": "united kingdom",
        "scotland": "united kingdom",
        "wales": "united kingdom",
        "holland": "netherlands",
        "deutschland": "germany",
        "brasil": "brazil",
        "california": "united states",
        "nevada": "united states",
        "florida": "united states",
        "texas": "united states",
        "new york": "united states",
        "washington": "united states",
    }
    return aliases.get(country, country)


def normalize_continent(value: Any) -> str | None:
    if not value:
        return None
    continent = str(value).strip().lower()
    if "australia" in continent or "oceania" in continent:
        return "oceania"
    return continent


def reverse_geocode(country: Any) -> tuple[float | None, float | None]:
    normalized = normalize_country(country)
    if not normalized:
        return None, None
    for key, coords in COUNTRY_COORDS.items():
        if key in normalized or normalized in key:
            return coords
    return None, None


def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, float]:
    for pred in predictions:
        if pred.get("predicted_lat") is None:
            lat, lon = reverse_geocode(pred.get("predicted_country"))
            pred["predicted_lat"] = lat
            pred["predicted_lon"] = lon

    metrics: dict[str, float] = {}
    total = sum(1 for pred in predictions if pred.get("ground_truth_country"))

    for level in ["city", "country", "continent"]:
        tp = fp = fn = 0
        pred_key = f"predicted_{level}"
        gt_key = f"ground_truth_{level}"
        for pred in predictions:
            gt_val = pred.get(gt_key)
            if not gt_val:
                continue
            pred_val = pred.get(pred_key)
            if not pred_val:
                fn += 1
                continue
            if level == "country":
                pred_norm = normalize_country(pred_val)
                gt_norm = normalize_country(gt_val)
            elif level == "continent":
                pred_norm = normalize_continent(pred_val)
                gt_norm = normalize_continent(gt_val)
            else:
                pred_norm = str(pred_val).strip().lower()
                gt_norm = str(gt_val).strip().lower()
            if pred_norm == gt_norm:
                tp += 1
            else:
                fp += 1
                fn += 1
        accuracy = tp / total if total else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics[f"{level}_accuracy"] = round(accuracy, 4)
        metrics[f"{level}_recall"] = round(recall, 4)
        metrics[f"{level}_f1"] = round(f1, 4)

    for threshold, name in [(1.0, "street_1km"), (25.0, "city_25km"), (750.0, "country_750km")]:
        within = total_with_coords = 0
        for pred in predictions:
            lat = pred.get("ground_truth_lat")
            lon = pred.get("ground_truth_lon")
            pred_lat = pred.get("predicted_lat")
            pred_lon = pred.get("predicted_lon")
            if lat is None or lon is None or pred_lat is None or pred_lon is None:
                continue
            total_with_coords += 1
            try:
                if haversine(float(lat), float(lon), float(pred_lat), float(pred_lon)) <= threshold:
                    within += 1
            except (TypeError, ValueError):
                continue
        metrics[name] = round(within / total_with_coords, 4) if total_with_coords else 0.0

    times = [float(pred["inference_time"]) for pred in predictions if pred.get("inference_time") is not None]
    if times:
        metrics["avg_inference_time_s"] = round(sum(times) / len(times), 4)
    responses = [str(pred.get("model_response", "")) for pred in predictions if pred.get("model_response")]
    if responses:
        metrics["avg_generated_tokens"] = round(sum(len(response.split()) for response in responses) / len(responses), 4)
    metrics["total"] = float(total)
    return metrics


def method_name(prediction_stem: str) -> str:
    name = prediction_stem.lower()
    if "_step1" in name and "_step2" not in name:
        return "geocot_step1"
    if "_step2" in name and "_step3" not in name:
        return "geocot_step1_2"
    if "_step3" in name and "_step4" not in name:
        return "geocot_step1_2_3"
    if "_step4" in name and "_step5" not in name and "_full" not in name:
        return "geocot_step1_2_3_4"
    if "_step5" in name or "_full" in name:
        return "geocot_full"
    if "geocot" in name:
        return "qwen_geocot"
    if "cot" in name and "geocot" not in name:
        return "qwen_cot"
    return prediction_stem


def route_experiment(experiment: str, prediction_stem: str) -> bool:
    pred = prediction_stem.lower()
    is_ablation = pred.startswith("abl_")
    is_main = not is_ablation
    if experiment == "ablation_explanation_scaffolds":
        return is_ablation and method_name(prediction_stem) != "qwen_cot"
    if experiment in {"geocomp_classification", "geocomp_distance", "geocomp_efficiency"}:
        return is_main
    return False


from eval.evaluate_results import compute_metrics, method_name, route_experiment


def in_split(prediction: dict[str, Any]) -> bool:
    image_path = prediction.get("image_path")
    return bool(image_path) and Path(str(image_path)).name in SPLIT_IMAGE_BASENAMES


def evaluate(results_dir: Path, reference_path: Path, scores_path: Path) -> dict[str, Any]:
    pred_files = sorted(results_dir.glob("*_predictions.json"))
    print(f"Found prediction files: {[path.stem.replace('_predictions', '') for path in pred_files]}")
    print(f"Using {SPLIT_NAME} slice with {len(SPLIT_IMAGE_BASENAMES)} image ids")

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
            metrics = compute_metrics(valid_gt)
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
            if not route_experiment(exp_name, pred_name):
                continue
            entry = {key: metrics[key] for key in ref_metrics if key in metrics}
            if entry:
                exp_copy["results"][method_name(pred_name)] = entry
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
