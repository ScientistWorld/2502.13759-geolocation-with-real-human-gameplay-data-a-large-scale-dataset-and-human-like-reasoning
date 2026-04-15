#!/bin/bash
# Baseline script — runs standard CoT baseline for geolocation
#
# Runs a standard chain-of-thought prompt (without GeoCoT-structured
# geographical reasoning) on the geolocation dataset for comparison.

set -e

cd /home/user

RESULTS_DIR="/home/user/results"
mkdir -p "$RESULTS_DIR"

echo "=== Baseline: Standard CoT on GeoCLIP dataset ==="

# The run.sh script already includes CoT baseline inference.
# This script re-runs just the CoT portion if needed.

# Environment setup
if [ -f "/home/user/environment/setup.sh" ]; then
    source /home/user/environment/setup.sh
fi

# Find model
MODEL_PATH=""
for dir in "/home/user/checkpoints/Qwen2.5-VL-7B-Instruct" \
            "/home/user/checkpoints/Qwen2.5-VL-32B-Instruct" \
            "/home/user/data/downloads/Qwen2.5-VL-7B-Instruct" \
            "/home/user/data/downloads/Qwen2.5-VL-32B-Instruct"; do
    if [ -d "$dir" ] && [ -f "$dir/model.safetensors.index.json" ]; then
        MODEL_PATH="$dir"
        break
    fi
done

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: No VLM model found"
    exit 1
fi

echo "Model: $MODEL_PATH"

# Run CoT inference
python3 << 'PYEOF'
import os, csv, json, re, time
import torch
os.environ['HF_HOME'] = '/home/user/data/downloads/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/home/user/data/downloads/hf_cache'
os.environ['HF_HUB_OFFLINE'] = '1'

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

# Find model
MODEL_PATH = ""
for d in ['/home/user/checkpoints/Qwen2.5-VL-7B-Instruct',
           '/home/user/checkpoints/Qwen2.5-VL-32B-Instruct',
           '/home/user/data/downloads/Qwen2.5-VL-7B-Instruct',
           '/home/user/data/downloads/Qwen2.5-VL-32B-Instruct']:
    if os.path.isdir(d) and os.path.exists(os.path.join(d, 'model.safetensors.index.json')):
        MODEL_PATH = d
        break

processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_PATH)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")

COT_PROMPT = """Let's think step by step about the geographical location where this image was taken. Identify any visible clues about the location, then provide your best guess of the city, country, and continent.

Output format: Location: [City], [Country], [Continent]
"""

# Same comprehensive country list as run.sh
ALL_COUNTRIES = {
    "kenya": ("Kenya", "Africa"), "madagascar": ("Madagascar", "Africa"),
    "ecuador": ("Ecuador", "South America"), "chile": ("Chile", "South America"),
    "brazil": ("Brazil", "South America"), "argentina": ("Argentina", "South America"),
    "peru": ("Peru", "South America"), "colombia": ("Colombia", "South America"),
    "united states": ("United States", "North America"), "canada": ("Canada", "North America"),
    "mexico": ("Mexico", "North America"), "germany": ("Germany", "Europe"),
    "france": ("France", "Europe"), "china": ("China", "Asia"), "japan": ("Japan", "Asia"),
    "india": ("India", "Asia"), "australia": ("Australia", "Oceania"),
    "russia": ("Russia", "Europe"),
}

def parse_location(text):
    if not text: return {"city": None, "country": None, "continent": None}
    result = {"city": None, "country": None, "continent": None}
    text_lower = text.lower()
    for name, (canonical, continent) in ALL_COUNTRIES.items():
        if name in text_lower:
            result["country"] = canonical
            result["continent"] = continent
            break
    return result

# Load dataset
data = []
with open('/home/user/data/geoclip/sample_balanced.csv') as f:
    for row in csv.DictReader(f):
        data.append(row)

results = []
for i, row in enumerate(data):
    img_path = row['image_path']
    if not os.path.exists(img_path): continue
    try:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": COT_PROMPT}
        ]}]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text_input], images=image_inputs, videos=None, padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], ids)]
        response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        pred = parse_location(response)
        results.append({
            "image_path": img_path,
            "ground_truth_city": row.get('city'), "ground_truth_country": row.get('country'),
            "ground_truth_continent": row.get('continent'),
            "ground_truth_lat": float(row['lat']) if row.get('lat') else None,
            "ground_truth_lon": float(row['lon']) if row.get('lon') else None,
            "predicted_city": pred.get('city'), "predicted_country": pred.get('country'),
            "predicted_continent": pred.get('continent'), "model_response": response,
        })
        if (i+1) % 10 == 0:
            print(f"  Processed {i+1}/{len(data)}")
    except Exception as e:
        print(f"  Error on {img_path}: {e}")

with open('/home/user/results/cot_predictions.json', 'w') as f:
    json.dump(results, f)
print(f"CoT baseline: {sum(1 for r in results if r.get('predicted_country'))}/{len(results)} valid")
PYEOF

echo ""
echo "=== Evaluating baseline ==="
bash scripts/evaluate.sh
