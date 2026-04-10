#!/bin/bash
# Run GeoCoT reproduction with Qwen2.5-VL-7B-Instruct.
#
# This script runs inside the compute container with GPU(s) available.
# All models and data are pre-downloaded.

set -e

cd /home/user

echo "=========================================="
echo "GeoCoT Reproduction - $(date)"
echo "=========================================="

RESULTS_DIR="/home/user/results"
mkdir -p "$RESULTS_DIR"

# =============================================================================
# Step 0: Environment setup
# =============================================================================
echo ""
echo "=== Step 0: Environment Setup ==="

export PYTHONPATH="/home/user/pylib:/home/user"
export HF_HOME="/home/user/shared/models/hf"
export TRANSFORMERS_CACHE="/home/user/shared/models/hf"
export HF_HUB_OFFLINE="1"
export CUDA_VISIBLE_DEVICES=""

echo "Python: $(python3 --version)"
python3 -c "
import sys
sys.path.insert(0, '/home/user/pylib')
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# Verify packages
python3 -c "
import sys
sys.path.insert(0, '/home/user/pylib')
import transformers; print(f'transformers: {transformers.__version__}')
import qwen_vl_utils; print('qwen_vl_utils: OK')
import PIL; print('PIL: OK')
"

# =============================================================================
# Step 1: Verify models and data
# =============================================================================
echo ""
echo "=== Step 1: Verifying Models and Data ==="

# Check Qwen model
MODEL_DIR=""
for dir in "/home/user/checkpoints/Qwen2.5-VL-7B-Instruct" "/home/user/shared/models/Qwen2.5-VL-7B-Instruct"; do
    if [ -d "$dir" ] && [ -f "$dir/model.safetensors.index.json" ]; then
        MODEL_DIR="$dir"
        break
    fi
done

if [ -z "$MODEL_DIR" ]; then
    echo "ERROR: Qwen2.5-VL-7B-Instruct not found!"
    echo "Checked:"
    echo "  /home/user/checkpoints/Qwen2.5-VL-7B-Instruct"
    echo "  /home/user/shared/models/Qwen2.5-VL-7B-Instruct"
    exit 1
fi
echo "Found Qwen2.5-VL at: $MODEL_DIR"

# Check geoclip data
CSV_PATH="/home/user/data/geoclip/geoclip.csv"
if [ ! -f "$CSV_PATH" ]; then
    echo "ERROR: GeoCLIP CSV not found at $CSV_PATH"
    exit 1
fi
NUM_IMAGES=$(python3 -c "
import sys, csv
sys.path.insert(0, '/home/user/pylib')
with open('$CSV_PATH') as f:
    reader = csv.DictReader(f)
    print(sum(1 for _ in reader))
")
echo "Found $NUM_IMAGES images in GeoCLIP dataset"

# =============================================================================
# Step 2: Enrich dataset with city names (reverse geocoding)
# =============================================================================
echo ""
echo "=== Step 2: Enriching Dataset with City Names ==="

python3 -c "
import sys
sys.path.insert(0, '/home/user/pylib')
sys.path.insert(0, '/home/user')
import pandas as pd
import os
sys.path.insert(0, '/home/user/data')

from reverse_geocode import reverse_geocode

df = pd.read_csv('$CSV_PATH')
print(f'Total rows: {len(df)}')

# Check if already enriched
if 'city' in df.columns:
    non_null = df['city'].notna() & (df['city'] != '')
    if non_null.sum() > 0:
        print(f'Already enriched with {non_null.sum()} city names')
    else:
        # Re-enrich
        cities = []
        for _, row in df.iterrows():
            result = reverse_geocode(row['lat'], row['lon'], row.get('country', None))
            cities.append(result['name'])
        df['city'] = cities
        df.to_csv('$CSV_PATH', index=False)
        print(f'Enriched with city names: {df[\"city\"].value_counts().head(10).to_dict()}')
else:
    cities = []
    for _, row in df.iterrows():
        result = reverse_geocode(row['lat'], row['lon'], row.get('country', None))
        cities.append(result['name'])
    df['city'] = cities
    df.to_csv('$CSV_PATH', index=False)
    print(f'Enriched with city names: {df[\"city\"].value_counts().head(10).to_dict()}')
"

# =============================================================================
# Step 3: Create balanced sample (40 images per country = 160 total)
# =============================================================================
echo ""
echo "=== Step 3: Creating Balanced Sample ==="

python3 -c "
import sys
sys.path.insert(0, '/home/user/pylib')
import pandas as pd
import os

df = pd.read_csv('$CSV_PATH')
print(f'Total images: {len(df)}')
print(f'Countries: {df[\"country\"].value_counts().to_dict()}')

# Create balanced sample: 40 images per country (stratified by lat/lon buckets)
SAMPLES_PER_COUNTRY = 40
samples = []
for country in df['country'].unique():
    subset = df[df['country'] == country].copy()
    # Sort by lat to get spatial diversity
    subset = subset.sort_values('lat')
    # Take evenly spaced samples
    n = len(subset)
    if n >= SAMPLES_PER_COUNTRY:
        step = n // SAMPLES_PER_COUNTRY
        subset = subset.iloc[::step].head(SAMPLES_PER_COUNTRY)
    samples.append(subset)

sample_df = pd.concat(samples, ignore_index=True)
print(f'Sample size: {len(sample_df)}')
print(f'Sample countries: {sample_df[\"country\"].value_counts().to_dict()}')

# Verify all image files exist
missing = 0
for _, row in sample_df.iterrows():
    if not os.path.exists(row['image_path']):
        missing += 1
print(f'Missing images: {missing}')

# Save sample CSV
sample_csv = '/home/user/data/geoclip/sample_balanced.csv'
sample_df.to_csv(sample_csv, index=False)
print(f'Saved balanced sample to {sample_csv}')
"

# =============================================================================
# Step 4: Run VLM inference - Load and test model
# =============================================================================
echo ""
echo "=== Step 4: Loading Qwen2.5-VL Model ==="

# First, test loading the model on a single image to verify it works
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/home/user/pylib')
sys.path.insert(0, '/home/user')
import os

os.environ['HF_HOME'] = '/home/user/shared/models/hf'
os.environ['TRANSFORMERS_CACHE'] = '/home/user/shared/models/hf'
os.environ['HF_HUB_OFFLINE'] = '1'

# Find model
model_path = None
for d in ['/home/user/checkpoints/Qwen2.5-VL-7B-Instruct',
           '/home/user/shared/models/Qwen2.5-VL-7B-Instruct']:
    if os.path.isdir(d) and os.path.exists(os.path.join(d, 'model.safetensors.index.json')):
        model_path = d
        break

if not model_path:
    print("ERROR: Model not found!")
    sys.exit(1)

print(f"Loading Qwen2.5-VL from {model_path}...")

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print(f"Model loaded! Device map: {model.hf_device_map}")
print("Testing inference on one image...")

# Test on one image
test_csv = '/home/user/data/geoclip/sample_balanced.csv'
import pandas as pd
df_test = pd.read_csv(test_csv)
test_row = df_test.iloc[0]
test_img = test_row['image_path']
print(f"Test image: {test_img}")

from PIL import Image
image = Image.open(test_img).convert('RGB')
print(f"Image size: {image.size}")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": test_img},
            {"type": "text", "text": "What country is this image from? Answer just the country name."}
        ]
    }
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, _ = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=None, padding=True, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(f"Model response: {response}")
print("MODEL LOAD TEST PASSED!")
PYEOF

MODEL_LOADED=$?
if [ $MODEL_LOADED -ne 0 ]; then
    echo "ERROR: Model loading failed!"
    exit 1
fi

# =============================================================================
# Step 5: Run GeoCoT inference on full sample
# =============================================================================
echo ""
echo "=== Step 5: Running GeoCoT Inference ==="

GEOCOT_OUTPUT="$RESULTS_DIR/geocot_predictions.json"
MAX_IMAGES=160

# Check if already done
if [ -f "$GEOCOT_OUTPUT" ]; then
    EXISTING=$(python3 -c "
import sys, json
sys.path.insert(0, '/home/user/pylib')
try:
    with open('$GEOCOT_OUTPUT') as f:
        data = json.load(f)
    if isinstance(data, list):
        print(len(data))
    else:
        print(0)
except:
    print(0)
" 2>/dev/null)
    if [ "$EXISTING" -ge 100 ]; then
        echo "GeoCoT predictions already exist ($EXISTING images), skipping."
    else
        rm -f "$GEOCOT_OUTPUT"
    fi
fi

if [ ! -f "$GEOCOT_OUTPUT" ]; then
    python3 << PYEOF
import sys
sys.path.insert(0, '/home/user/pylib')
sys.path.insert(0, '/home/user')
import os
os.environ['HF_HOME'] = '/home/user/shared/models/hf'
os.environ['TRANSFORMERS_CACHE'] = '/home/user/shared/models/hf'
os.environ['HF_HUB_OFFLINE'] = '1'

import json
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from method.prompt_template import GEO_COT_USER_PROMPT
from method.run_geocot import parse_location_prediction

# Load model
model_path = None
for d in ['/home/user/checkpoints/Qwen2.5-VL-7B-Instruct',
           '/home/user/shared/models/Qwen2.5-VL-7B-Instruct']:
    if os.path.isdir(d) and os.path.exists(os.path.join(d, 'model.safetensors.index.json')):
        model_path = d
        break

print(f"Loading model from {model_path}...")
processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print("Model loaded.")

# Load dataset
df = pd.read_csv('/home/user/data/geoclip/sample_balanced.csv')
print(f"Processing {len(df)} images with GeoCoT...")

results = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="GeoCoT"):
    img_path = row['image_path']
    if not os.path.exists(img_path):
        print(f"Warning: Missing image {img_path}")
        continue

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": GEO_COT_USER_PROMPT}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=None, padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        prediction = parse_location_prediction(response)

        result = {
            "image_path": img_path,
            "ground_truth_city": row.get('city', None),
            "ground_truth_country": row.get('country', None),
            "ground_truth_continent": row.get('continent', None),
            "ground_truth_lat": row.get('lat', None),
            "ground_truth_lon": row.get('lon', None),
            "predicted_city": prediction.get('city'),
            "predicted_country": prediction.get('country'),
            "predicted_continent": prediction.get('continent'),
            "model_response": response,
        }
        results.append(result)
    except Exception as e:
        print(f"Error on {img_path}: {e}")
        results.append({
            "image_path": img_path,
            "ground_truth_city": row.get('city', None),
            "ground_truth_country": row.get('country', None),
            "ground_truth_continent": row.get('continent', None),
            "ground_truth_lat": row.get('lat', None),
            "ground_truth_lon": row.get('lon', None),
            "predicted_city": None,
            "predicted_country": None,
            "predicted_continent": None,
            "model_response": None,
            "error": str(e),
        })

    # Save every 20 images
    if len(results) % 20 == 0:
        with open('$GEOCOT_OUTPUT', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} results")

# Final save
with open('$GEOCOT_OUTPUT', 'w') as f:
    json.dump(results, f, indent=2)

valid = [r for r in results if r.get('predicted_country') is not None]
print(f"GeoCoT complete: {len(valid)}/{len(results)} valid predictions")
PYEOF
fi

# =============================================================================
# Step 6: Run CoT baseline inference
# =============================================================================
echo ""
echo "=== Step 6: Running CoT Baseline Inference ==="

COT_OUTPUT="$RESULTS_DIR/cot_predictions.json"

if [ -f "$COT_OUTPUT" ]; then
    EXISTING=$(python3 -c "
import sys, json
sys.path.insert(0, '/home/user/pylib')
try:
    with open('$COT_OUTPUT') as f:
        data = json.load(f)
    if isinstance(data, list):
        print(len(data))
    else:
        print(0)
except:
    print(0)
" 2>/dev/null)
    if [ "$EXISTING" -ge 100 ]; then
        echo "CoT predictions already exist ($EXISTING images), skipping."
    else
        rm -f "$COT_OUTPUT"
    fi
fi

if [ ! -f "$COT_OUTPUT" ]; then
    python3 << PYEOF
import sys
sys.path.insert(0, '/home/user/pylib')
sys.path.insert(0, '/home/user')
import os
os.environ['HF_HOME'] = '/home/user/shared/models/hf'
os.environ['TRANSFORMERS_CACHE'] = '/home/user/shared/models/hf'
os.environ['HF_HUB_OFFLINE'] = '1'

import json
import torch
import pandas as pd
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from method.prompt_template import build_baseline_prompt
from method.run_geocot import parse_location_prediction

# Load model
model_path = None
for d in ['/home/user/checkpoints/Qwen2.5-VL-7B-Instruct',
           '/home/user/shared/models/Qwen2.5-VL-7B-Instruct']:
    if os.path.isdir(d) and os.path.exists(os.path.join(d, 'model.safetensors.index.json')):
        model_path = d
        break

print(f"Loading model from {model_path}...")
processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print("Model loaded.")

# Load dataset
df = pd.read_csv('/home/user/data/geoclip/sample_balanced.csv')
print(f"Processing {len(df)} images with CoT...")

cot_prompt = build_baseline_prompt()
results = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="CoT"):
    img_path = row['image_path']
    if not os.path.exists(img_path):
        continue

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": cot_prompt}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=None, padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        prediction = parse_location_prediction(response)

        result = {
            "image_path": img_path,
            "ground_truth_city": row.get('city', None),
            "ground_truth_country": row.get('country', None),
            "ground_truth_continent": row.get('continent', None),
            "ground_truth_lat": row.get('lat', None),
            "ground_truth_lon": row.get('lon', None),
            "predicted_city": prediction.get('city'),
            "predicted_country": prediction.get('country'),
            "predicted_continent": prediction.get('continent'),
            "model_response": response,
        }
        results.append(result)
    except Exception as e:
        print(f"Error on {img_path}: {e}")
        results.append({
            "image_path": img_path,
            "ground_truth_city": row.get('city', None),
            "ground_truth_country": row.get('country', None),
            "ground_truth_continent": row.get('continent', None),
            "ground_truth_lat": row.get('lat', None),
            "ground_truth_lon": row.get('lon', None),
            "predicted_city": None,
            "predicted_country": None,
            "predicted_continent": None,
            "model_response": None,
            "error": str(e),
        })

    if len(results) % 20 == 0:
        with open('$COT_OUTPUT', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} results")

with open('$COT_OUTPUT', 'w') as f:
    json.dump(results, f, indent=2)

valid = [r for r in results if r.get('predicted_country') is not None]
print(f"CoT complete: {len(valid)}/{len(results)} valid predictions")
PYEOF
fi

# =============================================================================
# Step 7: Evaluate and generate scores.json
# =============================================================================
echo ""
echo "=== Step 7: Evaluating and Generating scores.json ==="

python3 << PYEOF
import sys
sys.path.insert(0, '/home/user/pylib')
sys.path.insert(0, '/home/user')
import json

from eval.metrics import compute_all_metrics

scores = {"experiments": {}}

pred_files = {
    "qwen_geocot": "$RESULTS_DIR/geocot_predictions.json",
    "qwen_cot": "$RESULTS_DIR/cot_predictions.json",
}

experiment_results = {}

for method_name, filepath in pred_files.items():
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        continue

    with open(filepath) as f:
        predictions = json.load(f)

    valid_preds = [p for p in predictions if p.get('predicted_country') is not None]
    valid_gt = [p for p in valid_preds if p.get('ground_truth_country') is not None]
    print(f"{method_name}: {len(valid_preds)}/{len(predictions)} valid, {len(valid_gt)} with ground truth")

    if valid_gt:
        metrics = compute_all_metrics(valid_gt)
        experiment_results[method_name] = metrics
        for key in ['city_accuracy', 'country_accuracy', 'continent_accuracy',
                    'country_recall', 'country_f1', 'continent_recall', 'continent_f1',
                    'city_recall', 'city_f1',
                    'street_1km', 'city_25km', 'country_750km']:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.4f}")

# Build scores.json matching reference structure
if experiment_results:
    # Primary experiment: classification
    scores["experiments"]["geocomp_classification"] = {
        "description": "GeoCoT vs CoT on GeoCLIP dataset (4 countries: Kenya, Ecuador, Chile, Madagascar) with Qwen2.5-VL-7B-Instruct. Balanced sample of 160 images.",
        "weight": 0.5,
        "primary_metric": "country_accuracy",
        "metrics": {
            "country_accuracy": {"higher_is_better": True, "coefficient": 1.0},
            "continent_accuracy": {"higher_is_better": True, "coefficient": 0.8},
            "city_accuracy": {"higher_is_better": True, "coefficient": 1.2},
        },
        "results": {}
    }

    for method_name, metrics in experiment_results.items():
        if 'geocot' in method_name:
            label = 'qwen_geocot'
            method_type = 'proposed'
        else:
            label = 'qwen_cot'
            method_type = 'baseline'

        entry = {'type': method_type}
        for key in ['country_accuracy', 'continent_accuracy', 'city_accuracy',
                    'country_recall', 'country_f1', 'continent_recall', 'continent_f1',
                    'city_recall', 'city_f1']:
            if key in metrics:
                entry[key] = round(metrics[key], 4)

        scores["experiments"]["geocomp_classification"]["results"][label] = entry

    # Distance experiment
    scores["experiments"]["geocomp_distance"] = {
        "description": "Distance-based accuracy on GeoCLIP (Qwen2.5-VL)",
        "weight": 0.5,
        "primary_metric": "city_25km",
        "metrics": {
            "street_1km": {"higher_is_better": True, "coefficient": 1.0},
            "city_25km": {"higher_is_better": True, "coefficient": 1.0},
            "country_750km": {"higher_is_better": True, "coefficient": 0.8},
        },
        "results": {}
    }

    for method_name, metrics in experiment_results.items():
        if 'geocot' in method_name:
            label = 'qwen_geocot'
            method_type = 'proposed'
        else:
            label = 'qwen_cot'
            method_type = 'baseline'

        entry = {'type': method_type}
        for key in ['street_1km', 'city_25km', 'country_750km']:
            if key in metrics:
                entry[key] = round(metrics[key], 4)

        scores["experiments"]["geocomp_distance"]["results"][label] = entry

scores_path = '/home/user/scoring/scores.json'
with open(scores_path, 'w') as f:
    json.dump(scores, f, indent=2)
print(f"\nScores saved to {scores_path}")
print(json.dumps(scores, indent=2))
PYEOF

echo ""
echo "=========================================="
echo "Run complete - $(date)"
echo "=========================================="
