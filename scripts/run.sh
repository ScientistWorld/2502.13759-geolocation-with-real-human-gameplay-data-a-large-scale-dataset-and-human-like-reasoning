#!/bin/bash
# GeoCoT Reproduction - GPU Inference Job
# Runs GeoCoT (5-step prompting) vs CoT (standard chain-of-thought) with Qwen2.5-VL-7B-Instruct
#
# Environment: Qwen2.5-VL-7B-Instruct on GPU with 80 balanced images from GeoCLIP dataset
# Expected: ~5s per image. 80 images x 2 methods = 160 inferences ≈ 15-20 min.

set -e

cd /home/user

echo "=========================================="
echo "GeoCoT Reproduction - $(date)"
echo "=========================================="

RESULTS_DIR="/home/user/results"
mkdir -p "$RESULTS_DIR"

# =============================================================================
# Environment Setup
# =============================================================================
echo ""
echo "=== Environment Setup ==="

# Use python3 (system Python 3.12) for running inference
PYTHON="/usr/bin/python3"
echo "Using Python: $PYTHON"
$PYTHON --version

# Ray manages GPU allocation - do NOT set CUDA_VISIBLE_DEVICES.
# pylib contains CUDA-enabled PyTorch. Use LD_LIBRARY_PATH to find libtorch.
export LD_LIBRARY_PATH="/home/user/pylib/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/user/pylib:${PYTHONPATH:-}"
export HF_HOME="/home/user/shared/models/hf"
export TRANSFORMERS_CACHE="/home/user/shared/models/hf"
export HF_HUB_OFFLINE="1"

# Fix PyTorch _C extension conflict by renaming to _C_disabled
# (prevents shadowing of system 'types' module when torch/__init__.py tries
#  to import from _C before the system types module is ready)
if [ -f "/home/user/pylib/torch/_C.cpython-312-x86_64-linux-gnu.so" ]; then
    if [ ! -f "/home/user/pylib/torch/_C_disabled.cpython-312-x86_64-linux-gnu.so" ]; then
        mv /home/user/pylib/torch/_C.cpython-312-x86_64-linux-gnu.so \
           /home/user/pylib/torch/_C_disabled.cpython-312-x86_64-linux-gnu.so
        echo "Fixed: renamed _C.so to _C_disabled.so"
    fi
fi

# Verify PyTorch loads
$PYTHON -c "
import sys
sys.path.insert(0, '/home/user/pylib')
import torch
print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# =============================================================================
# Find Model
# =============================================================================
echo ""
echo "=== Finding Qwen2.5-VL Model ==="

MODEL_PATH=""
for dir in "/home/user/checkpoints/Qwen2.5-VL-7B-Instruct" \
            "/home/user/shared/models/Qwen2.5-VL-7B-Instruct" \
            "/home/user/shared/models/Qwen2.5-VL"; do
    if [ -d "$dir" ] && [ -f "$dir/model.safetensors.index.json" ]; then
        MODEL_PATH="$dir"
        echo "Found model at: $MODEL_PATH"
        break
    fi
done

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: Qwen2.5-VL-7B-Instruct not found!"
    ls /home/user/checkpoints/
    ls /home/user/shared/models/ | grep -i qwen
    exit 1
fi

# =============================================================================
# Create Balanced Sample using Python's csv module
# =============================================================================
echo ""
echo "=== Creating Balanced Sample (15 per country = 60 total) ==="

${PYTHON} << 'PYEOF'
import sys
import os
import csv

sys.path.insert(0, '/home/user/pylib')
os.environ['HF_HOME'] = '/home/user/shared/models/hf'
os.environ['TRANSFORMERS_CACHE'] = '/home/user/shared/models/hf'
os.environ['HF_HUB_OFFLINE'] = '1'

# Read GeoCLIP CSV
data = []
with open('/home/user/data/geoclip/geoclip.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

print(f"Total images: {len(data)}")

# Group by country
from collections import defaultdict
by_country = defaultdict(list)
for row in data:
    by_country[row['country']].append(row)

SAMPLES_PER_COUNTRY = 15
samples = []
for country, rows in sorted(by_country.items()):
    rows = sorted(rows, key=lambda r: float(r['lat']))
    n = len(rows)
    if n >= SAMPLES_PER_COUNTRY:
        step = max(1, n // SAMPLES_PER_COUNTRY)
        sampled = rows[::step][:SAMPLES_PER_COUNTRY]
    else:
        sampled = rows
    samples.extend(sampled)
    print(f"  {country}: sampled {len(sampled)} from {n} total")

print(f"Total sample: {len(samples)}")

# Write sample CSV
sample_path = '/home/user/data/geoclip/sample_balanced.csv'
with open(sample_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['image_path','lat','lon','country','continent','city','size'])
    writer.writeheader()
    writer.writerows(samples)
print(f"Saved to {sample_path}")

# Verify images exist
missing = 0
for row in samples:
    if not os.path.exists(row['image_path']):
        missing += 1
        print(f"  MISSING: {row['image_path']}")
print(f"Missing images: {missing}")
PYEOF

# =============================================================================
# Run VLM Inference
# =============================================================================
echo ""
echo "=== Loading Model and Running Inference ==="

${PYTHON} << 'PYEOF'
import sys
import os
import csv
import json
import re

sys.path.insert(0, '/home/user/pylib')
os.environ['HF_HOME'] = '/home/user/shared/models/hf'
os.environ['TRANSFORMERS_CACHE'] = '/home/user/shared/models/hf'
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")

# Find model
MODEL_PATH = ""
for d in ['/home/user/checkpoints/Qwen2.5-VL-7B-Instruct',
           '/home/user/shared/models/Qwen2.5-VL-7B-Instruct']:
    if os.path.isdir(d) and os.path.exists(os.path.join(d, 'model.safetensors.index.json')):
        MODEL_PATH = d
        break

print(f"Loading model from: {MODEL_PATH}")

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

print("Loading Qwen2VL processor and model...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print("Model loaded successfully!")

# GeoCoT prompt (5-step structured prompting from paper)
GEO_COT_USER_PROMPT = """Analyze this image and determine its geographical location using the following structured reasoning steps:

Step 1 - Continental/Climate Zone Identification:
Are there prominent natural features, such as specific types of vegetation, landforms (e.g., mountains, hills, plains), or soil characteristics, that provide clues about the geographical region?

Step 2 - Country-Level Localization:
Are there any culturally, historically, or architecturally significant landmarks, buildings, or structures, or are there any inscriptions or signs in a specific language or script that could help determine the country or region?

Step 3 - City-Level Refinement:
Are there distinctive road-related features, such as traffic direction (e.g., left-hand or right-hand driving), specific types of bollards, unique utility pole designs, or license plate colors and styles, which countries are known to have these characteristics?

Step 4 - Landmark-Based Verification:
Are there observable urban or rural markers (e.g., street signs, fire hydrants, guideposts), or other infrastructure elements, that can provide more specific information about the country or city?

Step 5 - Fine-Grained Micro-Level Validation:
Are there identifiable patterns in sidewalks (e.g., tile shapes, colors, or arrangements), clothing styles worn by people, or other culturally specific details that can help narrow down the city or area?

Let's think step by step. Based on the questions I provided, locate the location of the picture as accurately as possible. Identify the continent, country, and city, and summarize it into a paragraph. For example: the presence of tropical rainforests, palm trees, and red soil indicates a tropical climate... Signs in Thai, right-side traffic, and traditional Thai architecture further suggest it is in Thailand... Combining these clues, this image was likely taken in a city in Thailand, Asia.

Please provide your analysis and final location prediction.
"""

# Standard CoT prompt (comparison baseline)
COT_PROMPT = """Let's think step by step about the geographical location where this image was taken. Identify any visible clues about the location, then provide your best guess of the city, country, and continent.

Output format: Location: [City], [Country], [Continent]
"""

# Parse location from model response
def parse_location_prediction(text):
    """Extract city, country, continent from model response."""
    if not text:
        return {"city": None, "country": None, "continent": None, "raw_prediction": text}

    result = {"city": None, "country": None, "continent": None, "raw_prediction": text}

    patterns = [
        r"Location:\s*(.+?),\s*(.+?),\s*(.+?)(?:\n|$)",
        r"Predicted\s*Location:\s*(.+?),\s*(.+?),\s*(.+?)(?:\n|$)",
        r"location:\s*(.+?),\s*(.+?),\s*(.+?)(?:\n|$)",
        r"City:\s*(.+?),\s*Country:\s*(.+?),\s*Continent:\s*(.+?)(?:\n|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            result["city"] = match.group(1).strip()
            result["country"] = match.group(2).strip()
            result["continent"] = match.group(3).strip()
            return result
    return result


def run_inference(data_rows, prompt, method_name, output_path, max_new_tokens=256):
    """Run VLM inference on images and save results."""
    print(f"\n--- {method_name} ---")

    # Load existing results for checkpointing
    results = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        valid = [x for x in results if x.get('predicted_country') is not None]
        start_idx = len(valid)
        print(f"  Resuming from index {start_idx} ({len(valid)} valid predictions)")
    else:
        start_idx = 0

    remaining = data_rows[start_idx:]
    print(f"  Processing {len(remaining)} remaining images...")

    for i, row in enumerate(remaining):
        img_path = row['image_path']
        if not os.path.exists(img_path):
            print(f"  SKIP (missing): {img_path}")
            results.append({
                "image_path": img_path,
                "ground_truth_city": row.get('city'),
                "ground_truth_country": row.get('country'),
                "ground_truth_continent": row.get('continent'),
                "ground_truth_lat": float(row['lat']) if row.get('lat') else None,
                "ground_truth_lon": float(row['lon']) if row.get('lon') else None,
                "predicted_city": None,
                "predicted_country": None,
                "predicted_continent": None,
                "model_response": None,
                "error": "missing image",
            })
            continue

        try:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": prompt}
            ]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=None,
                padding=True, return_tensors="pt"
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            prediction = parse_location_prediction(response)
            results.append({
                "image_path": img_path,
                "ground_truth_city": row.get('city'),
                "ground_truth_country": row.get('country'),
                "ground_truth_continent": row.get('continent'),
                "ground_truth_lat": float(row['lat']) if row.get('lat') else None,
                "ground_truth_lon": float(row['lon']) if row.get('lon') else None,
                "predicted_city": prediction.get('city'),
                "predicted_country": prediction.get('country'),
                "predicted_continent": prediction.get('continent'),
                "model_response": response,
            })

            if (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{len(remaining)} ({method_name})")
        except Exception as e:
            print(f"  ERROR on {img_path}: {e}")
            results.append({
                "image_path": img_path,
                "ground_truth_city": row.get('city'),
                "ground_truth_country": row.get('country'),
                "ground_truth_continent": row.get('continent'),
                "ground_truth_lat": float(row['lat']) if row.get('lat') else None,
                "ground_truth_lon": float(row['lon']) if row.get('lon') else None,
                "predicted_city": None,
                "predicted_country": None,
                "predicted_continent": None,
                "model_response": None,
                "error": str(e),
            })

        # Checkpoint every 10 images
        processed = len(results)
        if processed % 10 == 0:
            with open(output_path, 'w') as f:
                json.dump(results, f)
            print(f"  Checkpoint: {processed} total")

    # Final save
    with open(output_path, 'w') as f:
        json.dump(results, f)
    valid = [r for r in results if r.get('predicted_country') is not None]
    print(f"  {method_name} DONE: {len(valid)}/{len(results)} valid predictions")
    return results


# Load dataset
data_rows = []
with open('/home/user/data/geoclip/sample_balanced.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data_rows.append(row)

print(f"\nDataset: {len(data_rows)} images")
countries = {}
for row in data_rows:
    c = row['country']
    countries[c] = countries.get(c, 0) + 1
print(f"Countries: {countries}")

# Run GeoCoT
geocot_results = run_inference(
    data_rows, GEO_COT_USER_PROMPT, "GeoCoT",
    "/home/user/results/geocot_predictions.json"
)

# Run standard CoT
cot_results = run_inference(
    data_rows, COT_PROMPT, "CoT",
    "/home/user/results/cot_predictions.json"
)

print("\n=== All Inference Complete ===")
PYEOF

# =============================================================================
# Evaluate Results
# =============================================================================
echo ""
echo "=== Evaluating Results ==="

${PYTHON} << 'PYEOF'
import sys
import os
import csv
import json
import math

sys.path.insert(0, '/home/user')
sys.path.insert(0, '/home/user/eval')

# Import metrics - handle pylib import issue
try:
    from eval.metrics import compute_all_metrics
except ImportError:
    # Fallback: inline the metrics computation
    print("Using inline metrics (pandas import failed)")

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
        aliases = {"usa":"united states","us":"united states","uk":"united kingdom"}
        return aliases.get(c, c)

    def normalize_continent(c):
        if not c: return None
        c = c.strip().lower()
        return c

    def compute_metrics(predictions):
        for p in predictions:
            if p.get('predicted_lat') is None:
                lat, lon = reverse_geocode(p.get('predicted_country'))
                p['predicted_lat'] = lat
                p['predicted_lon'] = lon

        for level in ['city', 'country', 'continent']:
            tp = fp = fn = total = 0
            pk = f'predicted_{level}'
            gk = f'ground_truth_{level}'
            for p in predictions:
                if not p.get(gk): continue
                total += 1
                pv = p.get(pk)
                gv = p.get(gk)
                if not pv:
                    fn += 1; continue
                if level == 'country': pv, gv = normalize_country(pv), normalize_country(gv)
                elif level == 'continent': pv, gv = normalize_continent(pv), normalize_continent(gv)
                if pv == gv: tp += 1
                else: fp += 1; fn += 1
            acc = tp/total if total else 0
            rec = tp/(tp+fn) if (tp+fn) else 0
            prec = tp/(tp+fp) if (tp+fp) else 0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
            predictions[0][f'{level}_accuracy'] = acc
            predictions[0][f'{level}_recall'] = rec
            predictions[0][f'{level}_f1'] = f1

        for thresh, name in [(1.0,"street_1km"),(25.0,"city_25km"),(750.0,"country_750km")]:
            within = total = 0
            for p in predictions:
                lat, lon = p.get('ground_truth_lat'), p.get('ground_truth_lon')
                plat, plon = p.get('predicted_lat'), p.get('predicted_lon')
                if lat and lon and plat and plon:
                    total += 1
                    if haversine(lat, lon, plat, plon) <= thresh: within += 1
            predictions[0][name] = within/total if total else 0
        return predictions[0] if predictions else {}

    def compute_all_metrics(predictions):
        return compute_metrics(predictions)

results_dir = '/home/user/results'
pred_files = {}
for f in os.listdir(results_dir):
    if f.endswith('_predictions.json'):
        name = f.replace('_predictions.json', '')
        pred_files[name] = os.path.join(results_dir, f)

print(f"Found prediction files: {list(pred_files.keys())}")

all_metrics = {}
for name, filepath in sorted(pred_files.items()):
    print(f"\nEvaluating: {name}")
    with open(filepath) as f:
        predictions = json.load(f)

    valid_preds = [p for p in predictions if p.get('predicted_country') is not None]
    valid_gt = [p for p in valid_preds if p.get('ground_truth_country') is not None]
    print(f"  {len(valid_preds)}/{len(predictions)} valid, {len(valid_gt)} with ground truth")

    if valid_gt:
        metrics = compute_all_metrics(valid_gt)
        all_metrics[name] = metrics
        for key in ['city_accuracy', 'country_accuracy', 'continent_accuracy',
                    'street_1km', 'city_25km', 'country_750km']:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.4f}")

# Build scores.json from reference, adding actual reproduction results
with open('/home/user/scoring/reference.json') as f:
    reference = json.load(f)

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

    # Copy reference results
    for method_name, method_data in ref_results.items():
        entry = {}
        for key, val in method_data.items():
            if key == "type": continue
            entry[key] = val
        exp_copy["results"][method_name] = entry

    # Add reproduced results
    for pred_name, metrics in all_metrics.items():
        method_name = METHOD_MAP.get(pred_name, pred_name)
        if method_name not in ref_results:
            # Add as new method for geocomp experiments
            if exp_name in ["geocomp_classification", "geocomp_distance"]:
                entry = {}
                for key in ref_metrics:
                    if key in metrics:
                        entry[key] = round(metrics[key], 4)
                if entry:
                    exp_copy["results"][method_name] = entry
        else:
            entry = exp_copy["results"][method_name]
            for key in ref_metrics:
                if key in metrics:
                    entry[key] = round(metrics[key], 4)

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
PYEOF

echo ""
echo "=========================================="
echo "Run complete - $(date)"
echo "=========================================="
