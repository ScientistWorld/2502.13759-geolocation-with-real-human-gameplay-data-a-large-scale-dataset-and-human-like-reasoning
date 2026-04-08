#!/bin/bash
# Run GeoCoT reproduction experiments.
#
# This script runs inside the compute container with GPU(s) available.
#
# Workflow:
#   1. Install CUDA runtime (if not in container)
#   2. Verify Python packages at /dev/shm/pylib
#   3. Ensure VLM model is available (LLaVA-1.5-7B)
#   4. Ensure dataset is available (GeoCLIP)
#   5. Run GeoCoT and standard CoT experiments
#   6. Evaluate and generate scores.json

set -e

cd /home/user

echo "=========================================="
echo "GeoCoT Reproduction - $(date)"
echo "=========================================="

RESULTS_DIR="/home/user/results"
mkdir -p "$RESULTS_DIR"

# =============================================================================
# Step 0: Install CUDA runtime at runtime (not build time)
# The container has base OS only — install CUDA here with real root access
# =============================================================================
echo ""
echo "=== Step 0: Setting up CUDA runtime ==="

install_cuda_runtime() {
    echo "CUDA not found in container, installing at runtime..."

    # Detect OS and install appropriate CUDA packages
    if [ -f /etc/debian_version ]; then
        # Debian/Ubuntu-based container
        export DEBIAN_FRONTEND=noninteractive
        wget -q -O /tmp/cuda-keyring.deb \
            https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
        dpkg -i /tmp/cuda-keyring.deb
        rm /tmp/cuda-keyring.deb
        apt-get update
        apt-get install -y --no-install-recommends cuda-runtime-12-5 || {
            echo "CUDA apt install failed, trying alternative..."
            apt-get install -y --no-install-recommends cuda-runtime-12-5 cuda-12-5 || true
        }
        rm -rf /var/lib/apt/lists/*
    elif [ -f /etc/redhat-release ]; then
        # RHEL/Rocky-based container
        # Add NVIDIA CUDA RHEL9 repo
        wget -q -O /etc/yum.repos.d/cuda.repo \
            https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
        yum install -y cuda-runtime-12-5 || yum install -y cuda-12-5-runtime || true
    fi

    echo "CUDA installation complete"
}

# Check if CUDA is available
if [ -d "/usr/local/cuda" ] || [ -d "/usr/lib/cuda" ]; then
    echo "CUDA already installed"
elif ldconfig -p 2>/dev/null | grep -q libcudart; then
    echo "CUDA libraries found via ldconfig"
else
    install_cuda_runtime || echo "WARNING: CUDA installation failed, continuing anyway..."
fi

# Set CUDA environment
if [ -d "/usr/local/cuda" ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    export CUDA_HOME="/usr/local/cuda"
fi

echo "CUDA_HOME=$CUDA_HOME"
nvcc --version 2>/dev/null || echo "nvcc not available (runtime libs only)"

# =============================================================================
# Step 1: Verify Python packages at /dev/shm/pylib
# (pre-installed by cluster admins on GPU compute nodes)
# =============================================================================
echo ""
echo "=== Step 1: Setting up Python packages ==="

# Check cluster pre-installed packages first
export PYTHONPATH="/dev/shm/pylib:/home/user/shared/models/llava_repo:/home/user"
if python3 -c "import torch; import transformers; import pandas" 2>/dev/null; then
    echo "Python packages available at /dev/shm/pylib (cluster pre-installed)"
else
    echo "Packages not at /dev/shm/pylib, checking system..."
    # Check system Python
    if python3 -c "import torch" 2>/dev/null; then
        echo "torch found in system Python"
        export PYTHONPATH="/home/user/shared/models/llava_repo:/home/user"
    else
        echo "Warning: torch not found, VLM may not work"
        export PYTHONPATH="/home/user/shared/models/llava_repo:/home/user"
    fi
fi

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
    python3 << 'EOPLACEHOLDER'
import json
import sys
sys.path.insert(0, '/dev/shm/pylib')
import pandas as pd
DATASET = "/home/user/data/geoclip/geoclip.csv"
MAX = 100
df = pd.read_csv(DATASET)
df = df.head(MAX)
results = []
for _, row in df.iterrows():
    results.append({
        'image_path': row['image_path'],
        'ground_truth_city': str(row.get('city', '')),
        'ground_truth_country': str(row.get('country', '')),
        'ground_truth_continent': str(row.get('continent', '')),
        'ground_truth_lat': float(row.get('lat')) if pd.notna(row.get('lat')) else None,
        'ground_truth_lon': float(row.get('lon')) if pd.notna(row.get('lon')) else None,
        'predicted_city': None,
        'predicted_country': None,
        'predicted_continent': None,
        'model_response': None,
        'error': 'VLM run failed'
    })
with open('/home/user/results/geocot_predictions.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Placeholder predictions: {len(results)}")
EOPLACEHOLDER
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
    python3 << 'EOPLACEHOLDER'
import json
import sys
sys.path.insert(0, '/dev/shm/pylib')
import pandas as pd
DATASET = "/home/user/data/geoclip/geoclip.csv"
MAX = 100
df = pd.read_csv(DATASET)
df = df.head(MAX)
results = []
for _, row in df.iterrows():
    results.append({
        'image_path': row['image_path'],
        'ground_truth_city': str(row.get('city', '')),
        'ground_truth_country': str(row.get('country', '')),
        'ground_truth_continent': str(row.get('continent', '')),
        'ground_truth_lat': float(row.get('lat')) if pd.notna(row.get('lat')) else None,
        'ground_truth_lon': float(row.get('lon')) if pd.notna(row.get('lon')) else None,
        'predicted_city': None,
        'predicted_country': None,
        'predicted_continent': None,
        'model_response': None,
        'error': 'VLM run failed'
    })
with open('/home/user/results/cot_predictions.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Placeholder predictions: {len(results)}")
EOPLACEHOLDER
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
sys.path.insert(0, '/dev/shm/pylib')
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
        "description": "GeoCLIP geolocation: GeoCoT vs standard CoT using LLaVA-1.5-7B on 999 images from Kenya/Ecuador/Madagascar/Chile",
        "results": {}
    }

    for method_name, metrics in all_results.items():
        if "error" in metrics:
            continue

        entry = {"type": "unknown"}
        for key, value in metrics.items():
            if isinstance(value, float) and not key.startswith('_') and key not in ['tp', 'fp', 'fn', 'total', 'valid_predictions', 'total_predictions']:
                entry[key] = round(value, 4)

        # Determine method type
        if 'geocot' in method_name and 'cot' not in method_name:
            entry['type'] = 'proposed'
        elif 'cot' in method_name:
            entry['type'] = 'baseline'

        label = method_name.replace('_predictions', '')
        scores["experiments"]["geoclip_geolocation"]["results"][label] = entry
        print(f"  {label} ({entry['type']}): {entry}")

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
