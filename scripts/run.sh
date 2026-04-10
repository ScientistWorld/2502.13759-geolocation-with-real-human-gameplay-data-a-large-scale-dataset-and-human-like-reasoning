#!/bin/bash
# Run GeoCoT reproduction experiments with Qwen2.5-VL-7B-Instruct.
#
# This script runs inside the compute container with GPU(s) available.
#
# Key environment:
#   - Python packages: /home/user/pylib
#   - Qwen2.5-VL-7B-Instruct: /home/user/checkpoints/Qwen2.5-VL-7B-Instruct
#   - GeoCLIP dataset: /home/user/data/geoclip/
#   - Workspace: /home/user

set -e

cd /home/user

echo "=========================================="
echo "GeoCoT Reproduction with Qwen2.5-VL - $(date)"
echo "=========================================="

RESULTS_DIR="/home/user/results"
mkdir -p "$RESULTS_DIR"

# =============================================================================
# Step 0: Verify environment
# =============================================================================
echo ""
echo "=== Step 0: Verifying environment ==="

export PYTHONPATH="/home/user/pylib:/home/user"
python3 -c "import sys; sys.path.insert(0, '/home/user/pylib'); import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python3 -c "import sys; sys.path.insert(0, '/home/user/pylib'); import transformers; print(f'transformers: {transformers.__version__}')"

# Check GPU
python3 -c "import sys; sys.path.insert(0, '/home/user/pylib'); import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# =============================================================================
# Step 1: Check for models and data
# =============================================================================
echo ""
echo "=== Step 1: Checking models and data ==="

QwenVLModel="/home/user/checkpoints/Qwen2.5-VL-7B-Instruct"
if [ -d "$QwenVLModel" ]; then
    echo "Found Qwen2.5-VL at: $QwenVLModel"
else
    echo "ERROR: Qwen2.5-VL not found at $QwenVLModel"
    exit 1
fi

DATASET="/home/user/data/geoclip/geoclip.csv"
if [ -f "$DATASET" ]; then
    echo "Found GeoCLIP dataset at: $DATASET"
else
    echo "ERROR: GeoCLIP dataset not found at $DATASET"
    exit 1
fi

NUM_IMAGES=$(python3 -c "
import sys
sys.path.insert(0, '/home/user/pylib')
import pandas
print(len(pandas.read_csv('$DATASET')))
")
echo "Dataset has $NUM_IMAGES images"

# =============================================================================
# Step 2: Run GeoCoT experiment
# =============================================================================
echo ""
echo "=== Step 2: Running GeoCoT with Qwen2.5-VL ==="

export PYTHONPATH="/home/user/pylib:/home/user"
export HF_HOME="/home/user/shared/models/hf"
export TRANSFORMERS_CACHE="/home/user/shared/models/hf"
export HF_HUB_OFFLINE="1"

# Process 80 images (20 per country for diversity)
MAX_IMAGES=80
GEOCOT_OUTPUT="$RESULTS_DIR/geocot_predictions.json"

if [ ! -f "$GEOCOT_OUTPUT" ] || [ $(python3 -c "
import sys, json
sys.path.insert(0, '/home/user/pylib')
with open('$GEOCOT_OUTPUT') as f:
    data = json.load(f)
print(len(data))
" 2>/dev/null || echo "0") -lt 20 ]; then
    echo "Running GeoCoT on $MAX_IMAGES images..."
    python3 -c "
import sys
sys.path.insert(0, '/home/user/pylib')
sys.path.insert(0, '/home/user')
import os
os.environ['HF_HOME'] = '/home/user/shared/models/hf'
os.environ['TRANSFORMERS_CACHE'] = '/home/user/shared/models/hf'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PYTHONPATH'] = '/home/user/pylib:/home/user'

from method.run_geocot import run_geocot

results = run_geocot(
    dataset_path='/home/user/data/geoclip/geoclip.csv',
    output_path='/home/user/results/geocot_predictions.json',
    vlm_type='qwen_vl',
    max_images=$MAX_IMAGES,
    method='geocot'
)
valid = [r for r in results if r.get('predicted_country') is not None]
print(f'GeoCoT complete: {len(valid)}/{len(results)} valid predictions')
" 2>&1 | tee "$RESULTS_DIR/geocot_log.txt"
else
    echo "GeoCoT predictions already exist, skipping."
fi

# =============================================================================
# Step 3: Run CoT baseline
# =============================================================================
echo ""
echo "=== Step 3: Running CoT Baseline ==="

COT_OUTPUT="$RESULTS_DIR/cot_predictions.json"

if [ ! -f "$COT_OUTPUT" ] || [ $(python3 -c "
import sys, json
sys.path.insert(0, '/home/user/pylib')
with open('$COT_OUTPUT') as f:
    data = json.load(f)
print(len(data))
" 2>/dev/null || echo "0") -lt 20 ]; then
    echo "Running CoT on $MAX_IMAGES images..."
    python3 -c "
import sys
sys.path.insert(0, '/home/user/pylib')
sys.path.insert(0, '/home/user')
import os
os.environ['HF_HOME'] = '/home/user/shared/models/hf'
os.environ['TRANSFORMERS_CACHE'] = '/home/user/shared/models/hf'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PYTHONPATH'] = '/home/user/pylib:/home/user'

from method.run_geocot import run_geocot

results = run_geocot(
    dataset_path='/home/user/data/geoclip/geoclip.csv',
    output_path='/home/user/results/cot_predictions.json',
    vlm_type='qwen_vl',
    max_images=$MAX_IMAGES,
    method='cot'
)
valid = [r for r in results if r.get('predicted_country') is not None]
print(f'CoT complete: {len(valid)}/{len(results)} valid predictions')
" 2>&1 | tee "$RESULTS_DIR/cot_log.txt"
else
    echo "CoT predictions already exist, skipping."
fi

# =============================================================================
# Step 4: Evaluate and generate scores.json
# =============================================================================
echo ""
echo "=== Step 4: Evaluating and generating scores.json ==="

python3 -c "
import sys
sys.path.insert(0, '/home/user/pylib')
sys.path.insert(0, '/home/user')
import os
import json

from eval.metrics import compute_all_metrics

scores = {'experiments': {}}

pred_files = {
    'qwen_geocot': 'geocot_predictions.json',
    'qwen_cot': 'cot_predictions.json',
}

experiment_results = {}

for method_name, filename in pred_files.items():
    pred_path = f'/home/user/results/{filename}'
    if not os.path.exists(pred_path):
        print(f'Warning: {pred_path} not found')
        continue

    with open(pred_path) as f:
        predictions = json.load(f)

    # Filter to predictions with country labels
    valid_preds = [p for p in predictions if p.get('predicted_country') is not None]
    valid_gt = [p for p in valid_preds if p.get('ground_truth_country') is not None]
    print(f'{method_name}: {len(valid_preds)}/{len(predictions)} valid, {len(valid_gt)} with ground truth')

    if valid_gt:
        metrics = compute_all_metrics(valid_gt)
        experiment_results[method_name] = metrics
        print(f'  Country accuracy: {metrics.get(\"country_accuracy\", 0):.4f}')
        print(f'  Continent accuracy: {metrics.get(\"continent_accuracy\", 0):.4f}')
        if 'city_accuracy' in metrics:
            print(f'  City accuracy: {metrics.get(\"city_accuracy\", 0):.4f}')

# Build scores.json
# Create experiment matching the reference structure
if experiment_results:
    scores['experiments']['geocomp_classification'] = {
        'description': 'GeoCoT vs CoT on GeoCLIP dataset (4 countries: Kenya, Ecuador, Chile, Madagascar) with Qwen2.5-VL-7B-Instruct',
        'results': {}
    }

    for method_name, metrics in experiment_results.items():
        if 'geocot' in method_name:
            label = 'qwen_geocot'
            method_type = 'proposed'
        else:
            label = 'qwen_cot'
            method_type = 'baseline'

        entry = {'type': method_type}

        # Add all available metrics (rounded)
        for key in ['country_accuracy', 'continent_accuracy', 'country_recall', 'country_f1',
                    'continent_accuracy', 'continent_recall', 'continent_f1',
                    'country_precision', 'continent_precision',
                    'country_tp', 'country_fp', 'country_fn', 'country_total',
                    'continent_tp', 'continent_fp', 'continent_fn', 'continent_total']:
            if key in metrics:
                entry[key] = round(metrics[key], 4)

        scores['experiments']['geocomp_classification']['results'][label] = entry

# Save scores.json
scores_path = '/home/user/scoring/scores.json'
with open(scores_path, 'w') as f:
    json.dump(scores, f, indent=2)
print(f'\\nScores saved to {scores_path}')
print(json.dumps(scores, indent=2))
"

echo ""
echo "=========================================="
echo "Run complete - $(date)"
echo "=========================================="
