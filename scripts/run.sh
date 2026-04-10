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
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "No PyTorch"
python3 -c "import transformers; print(f'transformers: {transformers.__version__}')" 2>/dev/null || echo "No transformers"

# Check GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); torch.cuda.is_available() and print(f'GPU: {torch.cuda.get_device_name(0)}')" 2>/dev/null || echo "No CUDA"

# =============================================================================
# Step 1: Check for models and data
# =============================================================================
echo ""
echo "=== Step 1: Checking models and data ==="

QWEN_MODEL="/home/user/checkpoints/Qwen2.5-VL-7B-Instruct"
if [ -d "$QWEN_MODEL" ]; then
    echo "Found Qwen2.5-VL at: $QWEN_MODEL"
else
    echo "ERROR: Qwen2.5-VL not found at $QWEN_MODEL"
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
" 2>/dev/null || echo "unknown")
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

MAX_IMAGES=100
echo "Running GeoCoT on $MAX_IMAGES images..."

python3 << 'PYEOF' 2>&1 | tee "$RESULTS_DIR/geocot_log.txt"
import sys
sys.path.insert(0, '/home/user/pylib')
sys.path.insert(0, '/home/user')
import os
os.environ['HF_HOME'] = '/home/user/shared/models/hf'
os.environ['TRANSFORMERS_CACHE'] = '/home/user/shared/models/hf'
os.environ['HF_HUB_OFFLINE'] = '1'

from method.run_geocot import run_geocot

results = run_geocot(
    dataset_path='/home/user/data/geoclip/geoclip.csv',
    output_path='/home/user/results/qwen_geocot_predictions.json',
    vlm_type='qwen_vl',
    max_images=100,
    method='geocot'
)
valid = [r for r in results if r.get('predicted_country') is not None]
print(f'GeoCoT complete: {len(valid)}/{len(results)} valid predictions')
PYEOF

# =============================================================================
# Step 3: Run CoT baseline
# =============================================================================
echo ""
echo "=== Step 3: Running CoT Baseline ==="

python3 << 'PYEOF' 2>&1 | tee "$RESULTS_DIR/cot_log.txt"
import sys
sys.path.insert(0, '/home/user/pylib')
sys.path.insert(0, '/home/user')
import os
os.environ['HF_HOME'] = '/home/user/shared/models/hf'
os.environ['TRANSFORMERS_CACHE'] = '/home/user/shared/models/hf'
os.environ['HF_HUB_OFFLINE'] = '1'

from method.run_geocot import run_geocot

results = run_geocot(
    dataset_path='/home/user/data/geoclip/geoclip.csv',
    output_path='/home/user/results/qwen_cot_predictions.json',
    vlm_type='qwen_vl',
    max_images=100,
    method='cot'
)
valid = [r for r in results if r.get('predicted_country') is not None]
print(f'CoT complete: {len(valid)}/{len(results)} valid predictions')
PYEOF

# =============================================================================
# Step 4: Evaluate and generate scores
# =============================================================================
echo ""
echo "=== Step 4: Evaluating predictions ==="

python3 << 'PYEOF'
import sys
sys.path.insert(0, '/home/user/pylib')
sys.path.insert(0, '/home/user')
import os
import json

from eval.metrics import compute_all_metrics

scores = {'experiments': {}}

pred_files = {
    'qwen_geocot': 'qwen_geocot_predictions.json',
    'qwen_cot': 'qwen_cot_predictions.json',
}

for method_name, filename in pred_files.items():
    pred_path = f'/home/user/results/{filename}'
    if not os.path.exists(pred_path):
        print(f'Warning: {pred_path} not found')
        continue

    with open(pred_path) as f:
        predictions = json.load(f)

    valid_preds = [p for p in predictions if p.get('predicted_country') is not None]
    print(f'{method_name}: {len(valid_preds)}/{len(predictions)} valid predictions')

    if valid_preds:
        metrics = compute_all_metrics(valid_preds)

        scores['experiments']['geocomp_classification'] = {
            'description': 'GeoCoT vs CoT on GeoCLIP dataset with Qwen2.5-VL-7B-Instruct (reproduction)',
            'results': {}
        }

        entry = {
            'country_accuracy': round(metrics.get('country_accuracy', 0.0), 4),
            'continent_accuracy': round(metrics.get('continent_accuracy', 0.0), 4),
            'type': 'proposed' if 'geocot' in method_name else 'baseline',
        }
        scores['experiments']['geocomp_classification']['results'][method_name] = entry

        print(f'  Country accuracy: {metrics.get("country_accuracy", 0):.4f}')
        print(f'  Continent accuracy: {metrics.get("continent_accuracy", 0):.4f}')

# Save scores
scores_path = '/home/user/scoring/scores.json'
with open(scores_path, 'w') as f:
    json.dump(scores, f, indent=2)
print(f'Scores saved to {scores_path}')
print(json.dumps(scores, indent=2))
PYEOF

echo ""
echo "=========================================="
echo "Run complete - $(date)"
echo "=========================================="