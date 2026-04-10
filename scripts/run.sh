#!/bin/bash
# Run GeoCoT reproduction experiments.
#
# This script runs inside the compute container with GPU(s) available.
#
# Key environment:
#   - Python packages: /pylib (installed by container build)
#   - Or system Python with packages
#   - LLaVA-1.5-7B: /home/user/shared/models/llava-v1.5-7b
#   - CLIP vision: /home/user/shared/models/clip-vit-large-patch14-336
#   - llava_repo: /home/user/shared/models/llava_repo
#   - Workspace: /home/user

set -e

cd /home/user

echo "=========================================="
echo "GeoCoT Reproduction - $(date)"
echo "=========================================="

RESULTS_DIR="/home/user/results"
mkdir -p "$RESULTS_DIR"

# =============================================================================
# Step 0: Verify environment
# =============================================================================
echo ""
echo "=== Step 0: Verifying environment ==="

# Check Python and packages
PYTHON_PATH=""
for python_bin in python3 python; do
    if $python_bin -c "import sys; sys.path.insert(0, '/pylib'); import torch; import transformers; print('Packages OK')" 2>/dev/null; then
        PYTHON_PATH="/pylib"
        break
    elif $python_bin -c "import torch; import transformers; print('System packages OK')" 2>/dev/null; then
        PYTHON_PATH="system"
        break
    fi
done

if [ -z "$PYTHON_PATH" ]; then
    echo "ERROR: No working Python environment found!"
    exit 1
fi
echo "Using Python packages from: $PYTHON_PATH"

# Check GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); torch.cuda.is_available() and print(f'GPU: {torch.cuda.get_device_name(0)}')" 2>/dev/null || echo "Warning: CUDA not available"

# =============================================================================
# Step 1: Check for models and data
# =============================================================================
echo ""
echo "=== Step 1: Checking models and data ==="

# Check LLaVA model
LLAVA_MODEL=""
for path in \
    "/home/user/shared/models/llava-v1.5-7b" \
    "/home/user/shared/models/llava-hf/llava-1.5-7b-hf"; do
    if [ -d "$path" ]; then
        LLAVA_MODEL="$path"
        echo "Found LLaVA at: $LLAVA_MODEL"
        break
    fi
done

# Check dataset
DATASET=""
for path in \
    "/home/user/data/geoclip/geoclip.csv" \
    "/home/user/data/im2gps3k/im2gps3k.csv"; do
    if [ -f "$path" ]; then
        DATASET="$path"
        echo "Found dataset at: $DATASET"
        break
    fi
done

if [ -z "$DATASET" ]; then
    echo "ERROR: No dataset found!"
    exit 1
fi

# Count images
NUM_IMAGES=$(python3 -c "
import pandas as pd
df = pd.read_csv('$DATASET')
print(len(df))
" 2>/dev/null || echo "unknown")
echo "Dataset has $NUM_IMAGES images"

# =============================================================================
# Step 2: Run GeoCoT experiment
# =============================================================================
echo ""
echo "=== Step 2: Running GeoCoT ==="

export PYTHONPATH="/pylib:/home/user/shared/models/llava_repo:/home/user:/home/user/shared/models"
export HF_HOME="/home/user/shared/models/hf"
export TRANSFORMERS_CACHE="/home/user/shared/models/hf"
export HF_HUB_OFFLINE="1"
export PYTORCH_ENABLE_MPS_FALLBACK="1"
export CUDA_VISIBLE_DEVICES="0"

MAX_IMAGES=50
echo "Running GeoCoT on $MAX_IMAGES images..."

python3 -c "
import sys
sys.path.insert(0, '/pylib')
sys.path.insert(0, '/home/user/shared/models/llava_repo')
sys.path.insert(0, '/home/user')
import os
os.environ['HF_HOME'] = '/home/user/shared/models/hf'
os.environ['TRANSFORMERS_CACHE'] = '/home/user/shared/models/hf'
os.environ['HF_HUB_OFFLINE'] = '1'

from method.run_geocot import run_geocot
results = run_geocot(
    dataset_path='$DATASET',
    output_path='$RESULTS_DIR/geocot_predictions.json',
    vlm_type='llava',
    max_images=$MAX_IMAGES,
    method='geocot'
)
print(f'GeoCoT complete: {len(results)} results')
" 2>&1 | tee "$RESULTS_DIR/geocot_log.txt"

# =============================================================================
# Step 3: Run CoT baseline
# =============================================================================
echo ""
echo "=== Step 3: Running CoT Baseline ==="

echo "Running standard CoT on $MAX_IMAGES images..."

python3 -c "
import sys
sys.path.insert(0, '/pylib')
sys.path.insert(0, '/home/user/shared/models/llava_repo')
sys.path.insert(0, '/home/user')
import os
os.environ['HF_HOME'] = '/home/user/shared/models/hf'
os.environ['TRANSFORMERS_CACHE'] = '/home/user/shared/models/hf'
os.environ['HF_HUB_OFFLINE'] = '1'

from method.run_geocot import run_geocot
results = run_geocot(
    dataset_path='$DATASET',
    output_path='$RESULTS_DIR/cot_predictions.json',
    vlm_type='llava',
    max_images=$MAX_IMAGES,
    method='cot'
)
print(f'CoT complete: {len(results)} results')
" 2>&1 | tee "$RESULTS_DIR/cot_log.txt"

# =============================================================================
# Step 4: Evaluate and score
# =============================================================================
echo ""
echo "=== Step 4: Evaluating predictions ==="

python3 -c "
import sys
sys.path.insert(0, '/pylib')
sys.path.insert(0, '/home/user')
import json
import os

from eval.metrics import compute_all_metrics

results_dir = '$RESULTS_DIR'
scores = {'experiments': {}}

for method_name in ['geocot', 'cot']:
    pred_file = os.path.join(results_dir, f'{method_name}_predictions.json')
    if not os.path.exists(pred_file):
        print(f'Warning: {pred_file} not found')
        continue

    with open(pred_file) as f:
        predictions = json.load(f)

    valid = [p for p in predictions if p.get('predicted_country') is not None]
    print(f'{method_name}: {len(valid)}/{len(predictions)} valid predictions')

    if valid:
        metrics = compute_all_metrics(valid)
        print(f'  City acc: {metrics.get(\"city_accuracy\", 0):.4f}')
        print(f'  Country acc: {metrics.get(\"country_accuracy\", 0):.4f}')
        print(f'  Continent acc: {metrics.get(\"continent_accuracy\", 0):.4f}')

        # Add to scores
        scores['experiments']['geolocation_comparison'] = {
            'description': 'GeoCoT vs standard CoT on geolocation dataset using LLaVA-1.5-7B',
            'results': {}
        }
        entry = {}
        for k, v in metrics.items():
            if isinstance(v, float) and not k.startswith('_') and k not in ['tp', 'fp', 'fn', 'total', 'valid_predictions', 'total_predictions']:
                entry[k] = round(v, 4)
        entry['type'] = 'proposed' if method_name == 'geocot' else 'baseline'
        scores['experiments']['geolocation_comparison']['results'][f'llava_{method_name}'] = entry

# Save scores
scores_path = '/home/user/scoring/scores.json'
os.makedirs(os.path.dirname(scores_path), exist_ok=True)
with open(scores_path, 'w') as f:
    json.dump(scores, f, indent=2)
print(f'Scores saved to {scores_path}')
print(json.dumps(scores, indent=2))
"

echo ""
echo "=========================================="
echo "Run complete - $(date)"
echo "=========================================="
