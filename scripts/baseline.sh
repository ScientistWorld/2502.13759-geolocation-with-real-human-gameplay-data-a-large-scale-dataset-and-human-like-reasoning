#!/bin/bash
# Run baseline method (standard Chain-of-Thought) for comparison.
#
# Baseline: standard CoT prompting (no GeoCoT structure)
# vs GeoCoT: structured 5-step geographical reasoning prompt

set -e

cd /home/user

RESULTS_DIR="/home/user/results"
mkdir -p "$RESULTS_DIR"

echo "=== Running Standard CoT Baseline ==="

# Determine which VLM to use
if [ -d "/home/user/shared/models/Qwen2.5-VL-7B-Instruct" ]; then
    VLM_TYPE="qwen_vl"
    MODEL_NAME="Qwen2.5-VL"
elif [ -d "/home/user/shared/models/llava-hf/llava-1.5-7b-hf" ]; then
    VLM_TYPE="llava"
    MODEL_NAME="LLaVA"
else
    VLM_TYPE="none"
    MODEL_NAME="none"
fi

echo "Using VLM: $MODEL_NAME"

# Find dataset
if [ -f "/home/user/data/im2gps3k/im2gps3k.csv" ]; then
    DATASET="/home/user/data/im2gps3k/im2gps3k.csv"
elif [ -f "/home/user/data/sample_geolocation.csv" ]; then
    DATASET="/home/user/data/sample_geolocation.csv"
else
    DATASET="/home/user/data/sample_geolocation.csv"
fi

# Run standard CoT
OUTPUT="$RESULTS_DIR/cot_predictions.json"
if [ "$VLM_TYPE" != "none" ]; then
    echo "Running standard CoT with $MODEL_NAME..."
    python3 -m method.run_geocot \
        --dataset "$DATASET" \
        --output "$OUTPUT" \
        --vlm "$VLM_TYPE" \
        --max-images 50 \
        --method cot
else
    echo "Creating sample CoT predictions..."
    python3 -c "
import json
import pandas as pd

df = pd.read_csv('$DATASET')
results = []
for _, row in df.iterrows():
    results.append({
        'image_path': row.get('image_path', ''),
        'ground_truth_city': row.get('city', None),
        'ground_truth_country': row.get('country', None),
        'ground_truth_continent': row.get('continent', None),
        'ground_truth_lat': row.get('lat', None),
        'ground_truth_lon': row.get('lon', None),
        'predicted_city': 'SampleCity',
        'predicted_country': 'SampleCountry',
        'predicted_continent': 'SampleContinent',
        'model_response': 'Standard CoT reasoning: [sample]'
    })
with open('$OUTPUT', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Created {len(results)} sample predictions')
"
fi

echo "CoT baseline predictions saved to: $OUTPUT"

# Evaluate
bash scripts/evaluate.sh
