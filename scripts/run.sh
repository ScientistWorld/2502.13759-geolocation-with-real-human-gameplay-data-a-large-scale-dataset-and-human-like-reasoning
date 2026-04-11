#!/bin/bash
# GeoCoT Reproduction - GPU Inference Job
# Runs GeoCoT (5-step prompting) vs CoT (standard chain-of-thought) with Qwen2.5-VL-7B-Instruct
#
# Expected: ~5s per image inference. 80 images x 2 methods = 160 inferences ≈ 15-20 min.

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

# Ray manages GPU allocation - do NOT set CUDA_VISIBLE_DEVICES.
# pylib contains CUDA-enabled PyTorch. Prepend to paths first.
export LD_LIBRARY_PATH="/home/user/pylib/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/user/pylib:${PYTHONPATH:-}"
export HF_HOME="/home/user/shared/models/hf"
export TRANSFORMERS_CACHE="/home/user/shared/models/hf"
export HF_HUB_OFFLINE="1"

# Fix: pylib/torch/_C directory contains stub .pyi files that conflict with
# the real _C.so extension module. Rename it to avoid Python package confusion.
if [ -d "/home/user/pylib/torch/_C" ] && [ -f "/home/user/pylib/torch/_C/__init__.pyi" ]; then
    echo "Fixing torch/_C directory conflict..."
    mv /home/user/pylib/torch/_C /home/user/pylib/torch/_C_stubs 2>/dev/null || true
fi

# Verify PyTorch can load
python3 -c "
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

# Verify model files
echo "Model files:"
ls "$MODEL_PATH/" | head -10

# =============================================================================
# Verify Data
# =============================================================================
echo ""
echo "=== Verifying Data ==="

CSV_PATH="/home/user/data/geoclip/geoclip.csv"
if [ ! -f "$CSV_PATH" ]; then
    echo "ERROR: GeoCLIP CSV not found at $CSV_PATH"
    exit 1
fi
N_LINES=$(wc -l < "$CSV_PATH")
echo "GeoCLIP CSV: $N_LINES lines"

# =============================================================================
# Create Balanced Sample
# =============================================================================
echo ""
echo "=== Creating Balanced Sample (20 per country = 80 total) ==="

python3 << 'PYEOF'
import sys
sys.path.insert(0, '/home/user/pylib')
import pandas as pd
import os

df = pd.read_csv('/home/user/data/geoclip/geoclip.csv')
print(f"Total images: {len(df)}")
print(f"Countries: {df['country'].value_counts().to_dict()}")

SAMPLES_PER_COUNTRY = 20
samples = []
for country in df['country'].unique():
    subset = df[df['country'] == country].copy()
    subset = subset.sort_values('lat')
    n = len(subset)
    if n >= SAMPLES_PER_COUNTRY:
        step = max(1, n // SAMPLES_PER_COUNTRY)
        subset = subset.iloc[::step].head(SAMPLES_PER_COUNTRY)
    samples.append(subset)
    print(f"  {country}: sampled {len(subset)} from {n} total")

sample_df = pd.concat(samples, ignore_index=True)
print(f"Total sample: {len(sample_df)}")

# Verify images exist
missing = 0
for _, row in sample_df.iterrows():
    if not os.path.exists(row['image_path']):
        missing += 1
        print(f"  MISSING: {row['image_path']}")
print(f"Missing images: {missing}")

sample_csv = '/home/user/data/geoclip/sample_balanced.csv'
sample_df.to_csv(sample_csv, index=False)
print(f"Saved to {sample_csv}")
PYEOF

# =============================================================================
# Run VLM Inference
# =============================================================================
echo ""
echo "=== Loading Model and Running Inference ==="

python3 << 'PYEOF'
import sys
sys.path.insert(0, '/home/user/pylib')
sys.path.insert(0, '/home/user/method')

import os
os.environ['HF_HOME'] = '/home/user/shared/models/hf'
os.environ['TRANSFORMERS_CACHE'] = '/home/user/shared/models/hf'
os.environ['HF_HUB_OFFLINE'] = '1'

import json
import re
import pandas as pd
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

# Import prompts
from prompt_template import GEO_COT_USER_PROMPT, COT_PROMPT

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

def run_inference(df, prompt, method_name, output_path, max_new_tokens=256, save_every=5):
    """Run VLM inference on images and save results with checkpointing."""
    print(f"\n--- {method_name} ---")

    # Load existing results for checkpointing
    results = []
    start_idx = 0
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        valid = [x for x in results if x.get('predicted_country') is not None]
        start_idx = len(valid)
        print(f"  Resuming from index {start_idx} ({len(valid)} valid predictions already)")

    # Filter df to remaining images
    remaining_df = df.iloc[start_idx:]
    print(f"  Processing {len(remaining_df)} remaining images...")

    for i, (_, row) in enumerate(remaining_df.iterrows()):
        img_path = row['image_path']
        if not os.path.exists(img_path):
            print(f"  SKIP (missing): {img_path}")
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
                "ground_truth_lat": row.get('lat'),
                "ground_truth_lon": row.get('lon'),
                "predicted_city": prediction.get('city'),
                "predicted_country": prediction.get('country'),
                "predicted_continent": prediction.get('continent'),
                "model_response": response,
            })
        except Exception as e:
            print(f"  ERROR on {img_path}: {e}")
            results.append({
                "image_path": img_path,
                "ground_truth_city": row.get('city'),
                "ground_truth_country": row.get('country'),
                "ground_truth_continent": row.get('continent'),
                "ground_truth_lat": row.get('lat'),
                "ground_truth_lon": row.get('lon'),
                "predicted_city": None,
                "predicted_country": None,
                "predicted_continent": None,
                "model_response": None,
                "error": str(e),
            })

        # Checkpoint every save_every images
        processed = len(results)
        if processed % save_every == 0:
            with open(output_path, 'w') as f:
                json.dump(results, f)
            print(f"  Checkpoint: {processed}/{len(df)} processed")

    # Final save
    with open(output_path, 'w') as f:
        json.dump(results, f)
    valid = [r for r in results if r.get('predicted_country') is not None]
    print(f"  {method_name} DONE: {len(valid)}/{len(results)} valid predictions")
    return results

# Load dataset
df = pd.read_csv('/home/user/data/geoclip/sample_balanced.csv')
print(f"\nDataset: {len(df)} images")
print(f"Countries: {df['country'].value_counts().to_dict()}")

# Run GeoCoT
geocot_results = run_inference(
    df, GEO_COT_USER_PROMPT, "GeoCoT",
    "/home/user/results/geocot_predictions.json",
    save_every=5
)

# Run standard CoT
cot_results = run_inference(
    df, COT_PROMPT, "CoT",
    "/home/user/results/cot_predictions.json",
    save_every=5
)

print("\n=== All Inference Complete ===")
PYEOF

# =============================================================================
# Evaluate Results
# =============================================================================
echo ""
echo "=== Evaluating Results ==="

python3 << 'PYEOF'
import sys
sys.path.insert(0, '/home/user/pylib')
import json
import os
sys.path.insert(0, '/home/user/eval')
from eval.metrics import compute_all_metrics

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

# Build scores.json
with open('/home/user/scoring/reference.json') as f:
    reference = json.load(f)

scores = {"experiments": {}}
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

    # Copy reference results first
    for method_name, method_data in ref_results.items():
        entry = {}
        for key, val in method_data.items():
            if key == "type":
                continue
            entry[key] = val
        exp_copy["results"][method_name] = entry

    # Update with reproduced results
    for pred_name, metrics in all_metrics.items():
        if pred_name == "geocot":
            method_name = "qwen_geocot"
        elif pred_name == "cot":
            method_name = "qwen_cot"
        else:
            method_name = pred_name

        if method_name not in ref_results:
            continue

        entry = exp_copy["results"][method_name]
        for key in ref_metrics:
            if key in metrics:
                entry[key] = round(metrics[key], 4)
        exp_copy["results"][method_name] = entry

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
