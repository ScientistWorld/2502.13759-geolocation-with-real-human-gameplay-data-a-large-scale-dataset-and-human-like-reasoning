#!/bin/bash
# Run GeoCoT reproduction experiments.
#
# This script runs inside the compute container with GPU(s) available.
#
# Workflow:
#   1. Install Python dependencies
#   2. Ensure VLM model is available
#   3. Ensure dataset is available
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
# Step 1: Verify Python environment
# =============================================================================
echo ""
echo "=== Step 1: Verifying Python environment ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || \
    echo "Warning: PyTorch not available"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || \
    echo "Warning: Transformers not available"

# =============================================================================
# Step 2: Check for VLM model
# =============================================================================
echo ""
echo "=== Step 2: Checking for VLM model ==="

VLM_TYPE="none"
MODEL_NAME="none"

# Check for Qwen2.5-VL
if [ -d "/home/user/shared/models/Qwen2.5-VL-7B-Instruct" ]; then
    VLM_TYPE="qwen_vl"
    MODEL_NAME="Qwen2.5-VL"
    echo "Found Qwen2.5-VL"
elif [ -d "/home/user/shared/models/llava-hf/llava-1.5-7b-hf" ]; then
    VLM_TYPE="llava"
    MODEL_NAME="LLaVA"
    echo "Found LLaVA"
else
    echo "No local VLM found in shared/models/"
    ls /home/user/shared/models/ | head -20
fi

# =============================================================================
# Step 3: Check for dataset
# =============================================================================
echo ""
echo "=== Step 3: Checking for dataset ==="

DATASET=""
if [ -f "/home/user/data/im2gps3k/im2gps3k.csv" ]; then
    DATASET="/home/user/data/im2gps3k/im2gps3k.csv"
    echo "Found Im2GPS3K at $DATASET"
elif [ -f "/home/user/data/sample_geolocation.csv" ]; then
    DATASET="/home/user/data/sample_geolocation.csv"
    echo "Found sample dataset at $DATASET"
else
    echo "No dataset found. Creating synthetic dataset..."
    python3 << 'EOPY'
import pandas as pd
import numpy as np
np.random.seed(42)

locations = [
    {'city': 'Berlin', 'country': 'Germany', 'continent': 'Europe', 'lat': 52.52, 'lon': 13.40},
    {'city': 'Paris', 'country': 'France', 'continent': 'Europe', 'lat': 48.86, 'lon': 2.35},
    {'city': 'London', 'country': 'United Kingdom', 'continent': 'Europe', 'lat': 51.51, 'lon': -0.13},
    {'city': 'Rome', 'country': 'Italy', 'continent': 'Europe', 'lat': 41.90, 'lon': 12.50},
    {'city': 'Barcelona', 'country': 'Spain', 'continent': 'Europe', 'lat': 41.39, 'lon': 2.17},
    {'city': 'Amsterdam', 'country': 'Netherlands', 'continent': 'Europe', 'lat': 52.37, 'lon': 4.90},
    {'city': 'New York', 'country': 'United States', 'continent': 'North America', 'lat': 40.71, 'lon': -74.01},
    {'city': 'Los Angeles', 'country': 'United States', 'continent': 'North America', 'lat': 34.05, 'lon': -118.24},
    {'city': 'San Francisco', 'country': 'United States', 'continent': 'North America', 'lat': 37.77, 'lon': -122.42},
    {'city': 'Chicago', 'country': 'United States', 'continent': 'North America', 'lat': 41.88, 'lon': -87.63},
    {'city': 'Toronto', 'country': 'Canada', 'continent': 'North America', 'lat': 43.65, 'lon': -79.38},
    {'city': 'Mexico City', 'country': 'Mexico', 'continent': 'North America', 'lat': 19.43, 'lon': -99.13},
    {'city': 'Tokyo', 'country': 'Japan', 'continent': 'Asia', 'lat': 35.68, 'lon': 139.69},
    {'city': 'Beijing', 'country': 'China', 'continent': 'Asia', 'lat': 39.90, 'lon': 116.41},
    {'city': 'Seoul', 'country': 'South Korea', 'continent': 'Asia', 'lat': 37.57, 'lon': 126.98},
    {'city': 'Singapore', 'country': 'Singapore', 'continent': 'Asia', 'lat': 1.35, 'lon': 103.82},
    {'city': 'Bangkok', 'country': 'Thailand', 'continent': 'Asia', 'lat': 13.76, 'lon': 100.50},
    {'city': 'Mumbai', 'country': 'India', 'continent': 'Asia', 'lat': 19.08, 'lon': 72.88},
    {'city': 'Dubai', 'country': 'United Arab Emirates', 'continent': 'Asia', 'lat': 25.20, 'lon': 55.27},
    {'city': 'Rio de Janeiro', 'country': 'Brazil', 'continent': 'South America', 'lat': -22.91, 'lon': -43.17},
    {'city': 'Buenos Aires', 'country': 'Argentina', 'continent': 'South America', 'lat': -34.60, 'lon': -58.38},
    {'city': 'Lima', 'country': 'Peru', 'continent': 'South America', 'lat': -12.05, 'lon': -77.04},
    {'city': 'Cape Town', 'country': 'South Africa', 'continent': 'Africa', 'lat': -33.93, 'lon': 18.42},
    {'city': 'Cairo', 'country': 'Egypt', 'continent': 'Africa', 'lat': 30.04, 'lon': 31.24},
    {'city': 'Sydney', 'country': 'Australia', 'continent': 'Oceania', 'lat': -33.87, 'lon': 151.21},
    {'city': 'Melbourne', 'country': 'Australia', 'continent': 'Oceania', 'lat': -37.81, 'lon': 144.96},
    {'city': 'Auckland', 'country': 'New Zealand', 'continent': 'Oceania', 'lat': -36.85, 'lon': 174.76},
    {'city': 'Vienna', 'country': 'Austria', 'continent': 'Europe', 'lat': 48.21, 'lon': 16.37},
    {'city': 'Prague', 'country': 'Czech Republic', 'continent': 'Europe', 'lat': 50.08, 'lon': 14.44},
    {'city': 'Stockholm', 'country': 'Sweden', 'continent': 'Europe', 'lat': 59.33, 'lon': 18.07},
    {'city': 'Moscow', 'country': 'Russia', 'continent': 'Europe', 'lat': 55.75, 'lon': 37.62},
]

samples = []
for i in range(50):
    loc = locations[i % len(locations)]
    samples.append({'id': i, 'image_path': f'synthetic_{i:04d}.jpg', **loc})

df = pd.DataFrame(samples)
df.to_csv('/home/user/data/sample_geolocation.csv', index=False)
print(f'Created {len(samples)} synthetic samples')
EOPY
    DATASET="/home/user/data/sample_geolocation.csv"
fi

# =============================================================================
# Step 4: Run GeoCoT experiment
# =============================================================================
echo ""
echo "=== Step 4: Running GeoCoT ==="

if [ "$VLM_TYPE" != "none" ]; then
    echo "Running GeoCoT with $MODEL_NAME (max 50 images)..."
    python3 -m method.run_geocot \
        --dataset "$DATASET" \
        --output "$RESULTS_DIR/geocot_predictions.json" \
        --vlm "$VLM_TYPE" \
        --max-images 50 \
        --method geocot 2>&1 || echo "GeoCoT run encountered errors (see above)"
else
    echo "No VLM available. Creating placeholder GeoCoT predictions..."
    python3 << 'EOPY'
import json
import pandas as pd

df = pd.read_csv('/home/user/data/sample_geolocation.csv')
df = df.head(50)

results = []
for _, row in df.iterrows():
    results.append({
        'image_path': row['image_path'],
        'ground_truth_city': row.get('city'),
        'ground_truth_country': row.get('country'),
        'ground_truth_continent': row.get('continent'),
        'ground_truth_lat': row.get('lat'),
        'ground_truth_lon': row.get('lon'),
        'predicted_city': None,
        'predicted_country': None,
        'predicted_continent': None,
        'model_response': None,
        'error': 'No VLM available'
    })

with open('/home/user/results/geocot_predictions.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Created {len(results)} placeholder predictions')
EOPY
fi

# =============================================================================
# Step 5: Run CoT baseline
# =============================================================================
echo ""
echo "=== Step 5: Running CoT Baseline ==="

if [ "$VLM_TYPE" != "none" ]; then
    echo "Running standard CoT with $MODEL_NAME (max 50 images)..."
    python3 -m method.run_geocot \
        --dataset "$DATASET" \
        --output "$RESULTS_DIR/cot_predictions.json" \
        --vlm "$VLM_TYPE" \
        --max-images 50 \
        --method cot 2>&1 || echo "CoT run encountered errors (see above)"
else
    echo "No VLM available. Creating placeholder CoT predictions..."
    python3 << 'EOPY'
import json
import pandas as pd

df = pd.read_csv('/home/user/data/sample_geolocation.csv')
df = df.head(50)

results = []
for _, row in df.iterrows():
    results.append({
        'image_path': row['image_path'],
        'ground_truth_city': row.get('city'),
        'ground_truth_country': row.get('country'),
        'ground_truth_continent': row.get('continent'),
        'ground_truth_lat': row.get('lat'),
        'ground_truth_lon': row.get('lon'),
        'predicted_city': None,
        'predicted_country': None,
        'predicted_continent': None,
        'model_response': None,
        'error': 'No VLM available'
    })

with open('/home/user/results/cot_predictions.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Created {len(results)} placeholder predictions')
EOPY
fi

# =============================================================================
# Step 6: Evaluate predictions
# =============================================================================
echo ""
echo "=== Step 6: Evaluating predictions ==="

python3 << 'EOPY'
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

        metrics = evaluate_predictions(pf_path)
        all_results[method_name] = metrics

        print(f"  Key metrics:")
        for key in ['city_accuracy', 'country_accuracy', 'continent_accuracy',
                    'street_1km', 'city_25km', 'country_750km', 'total_predictions', 'valid_predictions']:
            if key in metrics:
                print(f"    {key}: {metrics[key]:.4f}" if isinstance(metrics[key], float)
                      else f"    {key}: {metrics[key]}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

# Save all metrics
with open(os.path.join(results_dir, 'all_metrics.json'), 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nMetrics saved to {results_dir}/all_metrics.json")
EOPY

# =============================================================================
# Step 7: Generate scores.json
# =============================================================================
echo ""
echo "=== Step 7: Generating scores.json ==="

python3 << 'EOPY'
import json
import os

results_dir = '/home/user/results'
metrics_file = os.path.join(results_dir, 'all_metrics.json')

scores = {"experiments": {}}

if os.path.exists(metrics_file):
    with open(metrics_file, 'r') as f:
        all_results = json.load(f)

    # Create experiment for Im2GPS3K geolocation
    scores["experiments"]["im2gps3k_geolocation"] = {
        "description": "Im2GPS3K geolocation: GeoCoT vs standard CoT using local VLM",
        "results": {}
    }

    for method_name, metrics in all_results.items():
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
        scores["experiments"]["im2gps3k_geolocation"]["results"][label] = entry
        print(f"  {label}: {entry}")
else:
    print("Warning: No metrics file found")

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
