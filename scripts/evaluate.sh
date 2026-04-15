#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# Evaluates predictions from GeoCoT or baseline methods.
# Computes classification metrics (accuracy/recall/F1 at city/country/continent level)
# and distance metrics (1km/25km/750km thresholds).
#
# OUTPUT CONTRACT: writes scoring/scores.json with actual reproduced metrics.
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
    "united states": (37.0902, -95.7129), "canada": (56.1304, -106.3468),
    "south africa": (-30.5595, 22.9375), "egypt": (26.8206, 30.8025),
    "nigeria": (9.0820, 8.6753), "australia": (-25.2744, 133.7751),
    "germany": (51.1657, 10.4515), "france": (46.2276, 2.2137),
    "united kingdom": (55.3781, -3.4360), "china": (35.8617, 104.1954),
    "japan": (36.2048, 138.2529), "india": (20.5937, 78.9629),
    "russia": (61.5240, 105.3188), "mexico": (23.6345, -102.5528),
    "thailand": (15.8700, 100.9925), "indonesia": (-0.7893, 113.9213),
    "saudi arabia": (24.7136, 46.6753), "turkey": (38.9637, 35.2433),
    "south korea": (35.9078, 127.7669), "italy": (41.8719, 12.5674),
    "spain": (40.4637, -3.7492), "finland": (61.9241, 25.7482),
    "new zealand": (-40.9006, 174.8860), "uganda": (1.3733, 32.2903),
    "portugal": (39.3999, -8.2245), "greece": (39.0742, 21.8243),
    "netherlands": (52.1326, 5.2913), "belgium": (50.5039, 4.4699),
    "switzerland": (46.8182, 8.2275), "austria": (47.5162, 14.5501),
    "poland": (51.9194, 19.1451), "sweden": (60.1282, 18.6435),
    "norway": (60.4720, 8.4689), "denmark": (56.2639, 9.5018),
    "ireland": (53.1424, -7.6921), "ukraine": (48.3794, 31.1656),
    "romania": (45.9432, 24.9668), "hungary": (47.1625, 19.5033),
    "czech republic": (49.8175, 15.4730), "croatia": (45.1000, 15.2000),
    "tanzania": (-6.3690, 34.8888), "ethiopia": (9.1450, 40.4897),
    "ghana": (7.9465, -1.0232), "morocco": (31.7917, -7.0926),
    "algeria": (28.0339, 1.6596), "bolivia": (-16.2902, -63.5887),
    "paraguay": (-23.4425, -58.4438), "uruguay": (-32.5228, -55.7658),
    "venezuela": (6.4238, -66.5897), "guyana": (4.8604, -58.9302),
    "panama": (8.5380, -80.7821), "costa rica": (9.7489, -83.7534),
    "guatemala": (15.7835, -90.2308), "cuba": (21.5218, -77.7812),
    "jamaica": (18.1096, -77.2975), "philippines": (12.8797, 121.7740),
    "malaysia": (4.2105, 101.9758), "vietnam": (14.0583, 108.2772),
    "pakistan": (30.3753, 69.3451), "iran": (32.4279, 53.6880),
    "iraq": (33.3152, 44.3661), "israel": (31.0461, 34.8516),
    "jordan": (30.5852, 36.2384), "sri lanka": (7.8731, 80.7718),
    "bangladesh": (23.6850, 90.3563), "nepal": (28.3949, 84.1240),
    "cambodia": (12.5657, 104.9910), "myanmar": (21.9162, 95.9560),
    "singapore": (1.3521, 103.8198), "taiwan": (23.6978, 120.9605),
    "hong kong": (22.3193, 114.1694),
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
    aliases = {
        "usa": "united states", "us": "united states", "u.s.": "united states",
        "uk": "united kingdom", "british": "united kingdom",
        "england": "united kingdom", "scotland": "united kingdom",
        "wales": "united kingdom", "holland": "netherlands",
        "deutschland": "germany", "brasil": "brazil",
        "california": "united states", "nevada": "united states",
        "florida": "united states", "texas": "united states",
        "new york": "united states", "washington": "united states",
    }
    return aliases.get(c, c)

def normalize_continent(c):
    if not c: return None
    c = c.strip().lower()
    if "australia" in c: return "oceania"
    if "oceania" in c: return "oceania"
    return c

def compute_metrics(predictions):
    """Compute all geolocation metrics from predictions."""
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
            m = compute_metrics(valid_gt)
            all_results[name] = m
            for key in ['city_accuracy', 'country_accuracy', 'continent_accuracy',
                        'street_1km', 'city_25km', 'country_750km']:
                if key in m:
                    print(f"  {key}: {m[key]:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

# Build scores.json from reference structure + reproduced results
with open('/home/user/scoring/reference.json') as f:
    reference = json.load(f)

scores = {"experiments": {}}

# Map prediction file names to score method names
def get_method_name(pred_name):
    n = pred_name.lower()
    if "geocot" in n:
        return "qwen_geocot"  # Our reproduction of GeoCoT
    elif "cot" in n:
        return "qwen_cot"     # Our reproduction of CoT baseline
    return pred_name

for exp_name, exp_data in reference.get("experiments", {}).items():
    exp_copy = {
        "description": exp_data.get("description", ""),
        "weight": exp_data.get("weight", 0.5),
        "primary_metric": exp_data.get("primary_metric", ""),
        "metrics": exp_data.get("metrics", {}),
        "results": {}
    }

    ref_metrics = list(exp_data.get("metrics", {}).keys())

    # Add reproduced results (NOT reference/paper results - those stay in reference.json)
    for pred_name, metrics in all_results.items():
        method_name = get_method_name(pred_name)
        entry = {}
        for key in ref_metrics:
            if key in metrics:
                entry[key] = metrics[key]
        if entry:
            # Determine type based on method
            entry["type"] = "proposed" if "geocot" in method_name else "baseline"
            exp_copy["results"][method_name] = entry

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
