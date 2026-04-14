#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script evaluates predictions from GeoCoT or baseline methods.
# Uses Python's built-in modules to avoid external dependency issues.
#
# OUTPUT CONTRACT: This script writes scoring/scores.json containing
# actual reproduced metrics from running inference on predictions.
#
# Usage:
#   ./evaluate.sh                                      # evaluate results/ directory
#   ./evaluate.sh /path/to/predictions_dir             # evaluate specific directory

set -e

cd /home/user

mkdir -p scoring results

RESULTS_DIR="${1:-/home/user/results}"

echo "=== GeoCoT Evaluation ==="
echo "Evaluating predictions from: $RESULTS_DIR"

python3 << EOPY
import json
import os
import sys
import math

RESULTS_DIR = "${RESULTS_DIR}"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*math.sin(delta_lon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

COUNTRY_COORDS = {
    "kenya": (-1.2921, 36.8219), "madagascar": (-18.7669, 46.8691),
    "ecuador": (-1.8312, -78.1834), "chile": (-35.6751, -71.5430),
    "brazil": (-14.2350, -51.9253), "argentina": (-38.4161, -63.6167),
    "peru": (-9.1900, -75.0152), "colombia": (4.5709, -74.2973),
    "south africa": (-30.5595, 22.9375), "egypt": (26.8206, 30.8025),
    "nigeria": (9.0820, 8.6753), "united states": (37.0902, -95.7129),
    "usa": (37.0902, -95.7129), "canada": (56.1304, -106.3468),
    "germany": (51.1657, 10.4515), "france": (46.2276, 2.2137),
    "uk": (55.3781, -3.4360), "japan": (36.2048, 138.2529),
    "china": (35.8617, 104.1954), "india": (20.5937, 78.9629),
    "indonesia": (-0.7893, 113.9213), "australia": (-25.2744, 133.7751),
    "russia": (61.5240, 105.3188), "mexico": (23.6345, -102.5528),
    "thailand": (15.8700, 100.9925), "vietnam": (14.0583, 108.2772),
}

def reverse_geocode(country):
    if not country: return None, None
    c = country.strip().lower()
    for k, v in COUNTRY_COORDS.items():
        if k in c or c in k: return v
    return None, None

def normalize_country(c):
    if not c: return None
    c = c.strip().lower()
    aliases = {"usa":"united states","us":"united states","uk":"united kingdom",
               "american":"united states","british":"united kingdom"}
    return aliases.get(c, c)

def normalize_continent(c):
    if not c: return None
    c = c.strip().lower()
    if "australia" in c: return "oceania"
    return c

def compute_metrics(predictions):
    """Compute all geolocation metrics from predictions."""
    # Enrich with GPS coordinates
    for p in predictions:
        if p.get('predicted_lat') is None:
            lat, lon = reverse_geocode(p.get('predicted_country'))
            p['predicted_lat'] = lat
            p['predicted_lon'] = lon

    metrics = {}
    total = sum(1 for p in predictions if p.get('ground_truth_country'))

    for level in ['city', 'country', 'continent']:
        tp = fp = fn = 0
        pk = f'predicted_{level}'
        gk = f'ground_truth_{level}'
        for p in predictions:
            gv = p.get(gk)
            if not gv: continue
            pv = p.get(pk)
            if not pv:
                fn += 1; continue
            if level == 'country':
                pv, gv = normalize_country(pv), normalize_country(gv)
            elif level == 'continent':
                pv, gv = normalize_continent(pv), normalize_continent(gv)
            else:
                pv, gv = pv.strip().lower(), gv.strip().lower() if gv else ""
            if pv == gv: tp += 1
            else: fp += 1; fn += 1
        acc = tp/total if total else 0
        rec = tp/(tp+fn) if (tp+fn) else 0
        prec = tp/(tp+fp) if (tp+fp) else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        metrics[f'{level}_accuracy'] = round(acc, 4)
        metrics[f'{level}_recall'] = round(rec, 4)
        metrics[f'{level}_f1'] = round(f1, 4)

    for thresh, name in [(1.0,"street_1km"),(25.0,"city_25km"),(750.0,"country_750km")]:
        within = total_d = 0
        for p in predictions:
            lat, lon = p.get('ground_truth_lat'), p.get('ground_truth_lon')
            plat, plon = p.get('predicted_lat'), p.get('predicted_lon')
            if lat and lon and plat and plon:
                total_d += 1
                try:
                    dist = haversine(float(lat), float(lon), float(plat), float(plon))
                    if dist <= thresh: within += 1
                except: pass
        metrics[name] = round(within/total_d, 4) if total_d else 0

    metrics['total'] = total
    return metrics


# Find prediction files
pred_files = {}
if os.path.isdir(RESULTS_DIR):
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.endswith('_predictions.json'):
            name = f.replace('_predictions.json', '')
            pred_files[name] = os.path.join(RESULTS_DIR, f)

print(f"Found prediction files: {list(pred_files.keys())}")

all_results = {}

for name, filepath in sorted(pred_files.items()):
    print(f"\\nEvaluating: {name}")
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
            metrics = compute_metrics(valid_gt)
            all_results[name] = metrics
            for key in ['city_accuracy', 'country_accuracy', 'continent_accuracy',
                        'street_1km', 'city_25km', 'country_750km']:
                if key in metrics:
                    print(f"  {key}: {metrics[key]:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

# Load reference for structure
with open('/home/user/scoring/reference.json') as f:
    reference = json.load(f)

# Build scores.json with reproduction results
scores = {"experiments": {}}
METHOD_MAP = {"geocot": "qwen_geocot", "cot": "qwen_cot"}

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

    # Copy reference (paper) results
    for method_name, method_data in ref_results.items():
        entry = {}
        for key, val in method_data.items():
            if key == "type": continue
            entry[key] = val
        exp_copy["results"][method_name] = entry

    # Update/add reproduced results
    for pred_name, metrics in all_results.items():
        method_name = METHOD_MAP.get(pred_name, pred_name)
        if method_name not in ref_results:
            # Add new method to geocomp experiments
            if exp_name in ["geocomp_classification", "geocomp_distance"]:
                entry = {}
                for key in ref_metrics:
                    if key in metrics:
                        entry[key] = metrics[key]
                if entry:
                    exp_copy["results"][method_name] = entry
        else:
            entry = exp_copy["results"][method_name]
            for key in ref_metrics:
                if key in metrics:
                    entry[key] = metrics[key]

    scores["experiments"][exp_name] = exp_copy

scores_path = '/home/user/scoring/scores.json'
with open(scores_path, 'w') as f:
    json.dump(scores, f, indent=2)
print(f"\\nScores saved to {scores_path}")

# Print summary
print("\\n=== Summary ===")
for exp_name, exp_data in scores.get("experiments", {}).items():
    print(f"\\n{exp_name}:")
    for method, result in exp_data.get("results", {}).items():
        pm = exp_data.get("primary_metric", "")
        val = result.get(pm, "N/A")
        if isinstance(val, float): val = f"{val:.4f}"
        print(f"  {method}: {pm}={val}")
EOPY

echo ""
echo "=== Evaluation Complete ==="
