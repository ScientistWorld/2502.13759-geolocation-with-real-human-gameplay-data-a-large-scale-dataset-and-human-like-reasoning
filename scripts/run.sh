#!/bin/bash
# Run GeoCoT reproduction experiments.
#
# This script runs inside the compute container with GPU(s) available.
#
# Workflow:
#   1. Install Python dependencies
#   2. Ensure VLM model is available (LLaVA-1.5-7B)
#   3. Ensure dataset is available (GeoCLIP)
#   4. Run GeoCoT and standard CoT experiments
#   5. Evaluate and generate scores.json

set -e

cd /home/user

echo "=========================================="
echo "GeoCoT Reproduction - $(date)"
echo "=========================================="

RESULTS_DIR="/home/user/results"
mkdir -p "$RESULTS_DIR"

# =============================================================================
# Step 1: Install Python packages to /dev/shm (if not already installed)
# =============================================================================
echo ""
echo "=== Step 1: Setting up Python packages ==="

# /dev/shm/pylib is set via container environment; verify it works
if python3 -c "import torch, transformers, pandas" 2>/dev/null; then
    echo "Python packages available (via PYTHONPATH)"
else
    echo "Installing packages to /dev/shm/pylib..."
    mkdir -p /dev/shm/pylib
    pip3 install --target /dev/shm/pylib torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3 || true
    pip3 install --target /dev/shm/pylib transformers accelerate sentencepiece protobuf tiktoken pillow opencv-python-headless numpy scipy pandas scikit-learn tqdm jiwer rapidfuzz 2>&1 | tail -3 || true
    export PYTHONPATH="/dev/shm/pylib"
fi

# Set PYTHONPATH for this session
export PYTHONPATH="/dev/shm/pylib:$PYTHONPATH"

# =============================================================================
# Step 2: Check for VLM model
# =============================================================================
echo ""
echo "=== Step 2: Checking for VLM model ==="

VLM_TYPE="llava"
MODEL_NAME="LLaVA-1.5-7B"

# Check for LLaVA (preference: /dev/shm version with preprocessor_config.json)
if [ -d "/dev/shm/llava-v1.5-7b-config" ]; then
    echo "Found LLaVA-1.5-7B at /dev/shm/llava-v1.5-7b-config"
elif [ -d "/home/user/shared/models/llava-v1.5-7b" ]; then
    echo "Found LLaVA-1.5-7B at /home/user/shared/models/llava-v1.5-7b"
elif [ -d "/home/user/shared/models/llava-hf/llava-1.5-7b-hf" ]; then
    echo "Found LLaVA-1.5-7B at /home/user/shared/models/llava-hf/llava-1.5-7b-hf"
    VLM_TYPE="llava"
else
    echo "No LLaVA model found!"
    exit 1
fi

# Check for CLIP (vision tower for LLaVA)
if [ -d "/home/user/shared/models/clip-vit-large-patch14-336" ]; then
    echo "Found CLIP vision tower"
fi

# =============================================================================
# Step 3: Check for GeoCLIP dataset
# =============================================================================
echo ""
echo "=== Step 3: Checking for dataset ==="

DATASET=""
if [ -f "/home/user/data/geoclip/geoclip.csv" ]; then
    DATASET="/home/user/data/geoclip/geoclip.csv"
    echo "Found GeoCLIP dataset at $DATASET"
    NUM_IMAGES=$(PYTHONPATH="/dev/shm/pylib:/home/user/shared/models/llava_repo:/home/user" python3 -c "import pandas as pd; df = pd.read_csv('$DATASET'); print(len(df))")
    echo "Total images: $NUM_IMAGES"
elif [ -f "/home/user/data/im2gps3k/im2gps3k.csv" ]; then
    DATASET="/home/user/data/im2gps3k/im2gps3k.csv"
    echo "Found Im2GPS3K dataset at $DATASET"
else
    echo "ERROR: No dataset found!"
    exit 1
fi

# =============================================================================
# Step 4: Run GeoCoT experiment
# =============================================================================
echo ""
echo "=== Step 4: Running GeoCoT ==="

MAX_IMAGES=100
echo "Running GeoCoT with $MODEL_NAME (max $MAX_IMAGES images)..."

export PYTHONPATH="/dev/shm/pylib:/home/user/shared/models/llava_repo:/home/user"
PYTHONPATH="/dev/shm/pylib:/home/user/shared/models/llava_repo:/home/user" python3 -m method.run_geocot \
    --dataset "$DATASET" \
    --output "$RESULTS_DIR/geocot_predictions.json" \
    --vlm "$VLM_TYPE" \
    --max-images $MAX_IMAGES \
    --method geocot 2>&1 || {
    echo "GeoCoT run encountered errors (see above)"
    echo "Creating placeholder predictions..."
    PYTHONPATH="/dev/shm/pylib:/home/user/shared/models/llava_repo:/home/user" python3 -c "
import json
import pandas as pd
df = pd.read_csv('$DATASET')
df = df.head($MAX_IMAGES)
results = []
for _, row in df.iterrows():
    results.append({
        'image_path': row['image_path'],
        'ground_truth_city': row.get('city', ''),
        'ground_truth_country': row.get('country', ''),
        'ground_truth_continent': row.get('continent', ''),
        'ground_truth_lat': row.get('lat'),
        'ground_truth_lon': row.get('lon'),
        'predicted_city': None,
        'predicted_country': None,
        'predicted_continent': None,
        'model_response': None,
        'error': 'VLM run failed'
    })
with open('$RESULTS_DIR/geocot_predictions.json', 'w') as f:
    json.dump(results, f, indent=2)
"
}

# =============================================================================
# Step 5: Run CoT baseline
# =============================================================================
echo ""
echo "=== Step 5: Running CoT Baseline ==="

echo "Running standard CoT with $MODEL_NAME (max $MAX_IMAGES images)..."

PYTHONPATH="/dev/shm/pylib:/home/user/shared/models/llava_repo:/home/user" python3 -m method.run_geocot \
    --dataset "$DATASET" \
    --output "$RESULTS_DIR/cot_predictions.json" \
    --vlm "$VLM_TYPE" \
    --max-images $MAX_IMAGES \
    --method cot 2>&1 || {
    echo "CoT run encountered errors (see above)"
    echo "Creating placeholder predictions..."
    PYTHONPATH="/dev/shm/pylib:/home/user/shared/models/llava_repo:/home/user" python3 -c "
import json
import pandas as pd
df = pd.read_csv('$DATASET')
df = df.head($MAX_IMAGES)
results = []
for _, row in df.iterrows():
    results.append({
        'image_path': row['image_path'],
        'ground_truth_city': row.get('city', ''),
        'ground_truth_country': row.get('country', ''),
        'ground_truth_continent': row.get('continent', ''),
        'ground_truth_lat': row.get('lat'),
        'ground_truth_lon': row.get('lon'),
        'predicted_city': None,
        'predicted_country': None,
        'predicted_continent': None,
        'model_response': None,
        'error': 'VLM run failed'
    })
with open('$RESULTS_DIR/cot_predictions.json', 'w') as f:
    json.dump(results, f, indent=2)
"
}

# =============================================================================
# Step 6: Evaluate predictions
# =============================================================================
echo ""
echo "=== Step 6: Evaluating predictions ==="

SCORE_FILE="/home/user/scoring/scores.json"
mkdir -p "$(dirname $SCORE_FILE)"

PYTHONPATH="/dev/shm/pylib:/home/user" python3 << 'EOPY'
import sys
import json
import os

sys.path.insert(0, '/home/user')
from eval.metrics import evaluate_predictions, compute_all_metrics

results_dir = '/home/user/results'
os.makedirs(results_dir, exist_ok=True)

pred_files = [f for f in os.listdir(results_dir) if f.endswith('_predictions.json')]
print(f"Found {len(pred_files)} prediction files: {pred_files}")

all_results = {}
for pf in sorted(pred_files):
    pf_path = os.path.join(results_dir, pf)
    method_name = pf.replace('_predictions.json', '')
    print(f"\nEvaluating: {method_name}")

    try:
        predictions = json.load(open(pf_path))
        print(f"  Loaded {len(predictions)} predictions")

        # Filter out failed predictions
        valid = [p for p in predictions if p.get('predicted_country') is not None]
        print(f"  Valid predictions: {len(valid)}/{len(predictions)}")

        if valid:
            metrics = compute_all_metrics(valid)
        else:
            metrics = {"error": "No valid predictions"}

        all_results[method_name] = metrics

        print(f"  Key metrics:")
        for key in ['city_accuracy', 'country_accuracy', 'continent_accuracy',
                    'street_1km', 'city_25km', 'country_750km', 'total_predictions']:
            if key in metrics:
                v = metrics[key]
                print(f"    {key}: {v:.4f}" if isinstance(v, float) else f"    {key}: {v}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        all_results[method_name] = {"error": str(e)}

# Save all metrics
with open(os.path.join(results_dir, 'all_metrics.json'), 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nMetrics saved to {results_dir}/all_metrics.json")

# =============================================================================
# Step 7: Generate scores.json
# =============================================================================
print("\n=== Step 7: Generating scores.json ===")

scores = {"experiments": {}}

if all_results:
    # Create experiment entry for the GeoCLIP/GeoCoT comparison
    scores["experiments"]["geoclip_geolocation"] = {
        "description": "GeoCLIP geolocation: GeoCoT vs standard CoT using LLaVA-1.5-7B",
        "results": {}
    }

    for method_name, metrics in all_results.items():
        if "error" in metrics:
            continue

        entry = {}
        for key, value in metrics.items():
            if isinstance(value, float) and not key.startswith('_') and key not in ['tp', 'fp', 'fn', 'total']:
                entry[key] = round(value, 4)

        # Determine method type
        if 'geocot' in method_name and 'cot' not in method_name:
            method_type = 'proposed'
        elif 'cot' in method_name:
            method_type = 'baseline'
        else:
            method_type = 'proposed'

        label = method_name.replace('_predictions', '')
        scores["experiments"]["geoclip_geolocation"]["results"][label] = entry
        print(f"  {label}: {entry}")

scores_path = '/home/user/scoring/scores.json'
with open(scores_path, 'w') as f:
    json.dump(scores, f, indent=2)

print(f"\nScores written to {scores_path}")
print(json.dumps(scores, indent=2))
EOPY

echo ""
echo "=========================================="
echo "Run complete - $(date)"
echo "Results: /home/user/results/"
echo "Scores: /home/user/scoring/scores.json"
echo "=========================================="
