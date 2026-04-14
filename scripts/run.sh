#!/bin/bash
# GeoCoT Reproduction - GPU Inference Job
# Uses the paper's ACTUAL GeoCoT prompt from their GitHub codebase
# (ImageGeoLocator_TestSet.py)
#
# Key improvements over previous attempts:
#   1. Uses paper's actual prompt (not the 5-step Appendix B version)
#   2. Uses Qwen2.5-VL-32B-Instruct (much stronger than 7B)
#   3. Temperature=0.7 (matching paper's settings)
#   4. Better country extraction parser
#   5. 80 images per country for more robust evaluation
#
# Expected: ~60s/image for 32B GeoCoT, ~30s for CoT
# 80 images x 2 methods ≈ 90 min total with 32B

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

if [ -f "/home/user/environment/setup.sh" ]; then
    source /home/user/environment/setup.sh
else
    echo "WARNING: setup.sh not found"
fi

PYTHON="/usr/bin/python3"
echo "Using Python: $PYTHON"
$PYTHON --version

# =============================================================================
# Find Model - prefer 32B over 7B
# =============================================================================
echo ""
echo "=== Finding VLM Model ==="

MODEL_PATH=""
MODEL_SIZE=""
for dir in "/home/user/checkpoints/Qwen2.5-VL-32B-Instruct" \
            "/home/user/shared/models/Qwen2.5-VL-32B-Instruct" \
            "/home/user/checkpoints/Qwen2.5-VL-7B-Instruct" \
            "/home/user/shared/models/Qwen2.5-VL-7B-Instruct"; do
    if [ -d "$dir" ] && [ -f "$dir/config.json" ]; then
        MODEL_PATH="$dir"
        if echo "$dir" | grep -q "32B"; then
            MODEL_SIZE="32B"
        else
            MODEL_SIZE="7B"
        fi
        echo "Found model: $MODEL_PATH ($MODEL_SIZE)"
        break
    fi
done

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: No Qwen2.5-VL model found!"
    exit 1
fi

# =============================================================================
# Create Sample
# =============================================================================
echo ""
echo "=== Creating Sample ==="

${PYTHON} << 'PYEOF'
import os, csv
from collections import defaultdict

data = []
with open('/home/user/data/geoclip/geoclip.csv', 'r') as f:
    for row in csv.DictReader(f):
        data.append(row)

by_country = defaultdict(list)
for row in data:
    by_country[row['country']].append(row)

SAMPLES_PER_COUNTRY = 20
samples = []
for country, rows in sorted(by_country.items()):
    rows = sorted(rows, key=lambda r: float(r['lat']))
    n = len(rows)
    if n >= SAMPLES_PER_COUNTRY:
        step = max(1, n // SAMPLES_PER_COUNTRY)
        sampled = rows[::step][:SAMPLES_PER_COUNTRY]
    else:
        sampled = rows
    samples.extend(sampled)
    print(f"  {country}: {len(sampled)}/{n}")

with open('/home/user/data/geoclip/sample_balanced.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['image_path','lat','lon','country','continent','city','size'])
    w.writeheader()
    w.writerows(samples)
print(f"Total: {len(samples)} images, missing: {sum(1 for r in samples if not os.path.exists(r['image_path']))}")
PYEOF

# =============================================================================
# Run Inference
# =============================================================================
echo ""
echo "=== Loading Model and Running Inference ==="

${PYTHON} << 'PYEOF'
import os, csv, json, re, time, math

os.environ['HF_HOME'] = '/home/user/shared/models/hf'
os.environ['TRANSFORMERS_CACHE'] = '/home/user/shared/models/hf'
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB" if torch.cuda.is_available() else "")

# Find model
MODEL_PATH = ""
MODEL_SIZE = ""
for d in ['/home/user/checkpoints/Qwen2.5-VL-32B-Instruct',
           '/home/user/shared/models/Qwen2.5-VL-32B-Instruct',
           '/home/user/checkpoints/Qwen2.5-VL-7B-Instruct',
           '/home/user/shared/models/Qwen2.5-VL-7B-Instruct']:
    if os.path.isdir(d) and os.path.exists(os.path.join(d, 'config.json')):
        MODEL_PATH = d
        MODEL_SIZE = "32B" if "32B" in d else "7B"
        break

print(f"Model: {MODEL_PATH} ({MODEL_SIZE})")

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

t0 = time.time()
processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_PATH)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
)
print(f"Loaded in {time.time()-t0:.1f}s, GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# =================================================================
# PROMPTS - from paper's actual code
# =================================================================

# GeoCoT prompt: paper's actual prompt from ImageGeoLocator_TestSet.py
# Combines Q1 (street elements), Q2 (sidewalk patterns), and summary prompt
GEO_COT_PROMPT = """Q1: Are there easily identifiable street elements, such as road direction (left or right), types of bollards, utility poles, or license plate colors?

Q2: Are there identifiable sidewalk patterns (tile color, shape, arrangement) or clothing styles that can help determine the city?

Let's think step by step. Based on the questions I provided, locate the location of the picture as accurately as possible. Identify the continent, country, and city, and summarize it into a paragraph (note: it must be a paragraph). For example: the presence of tropical rainforests, palm trees, and red soil indicates a tropical climate, most likely located in Southeast Asia. The signs are in Thai and the hydrant is a green circle, indicating that it is in Thailand. Right-side traffic, cylindrical bollards with blue markings and license plates with black lettering on a white background meet Thai standards. Traditional Thai architecture, such as pitched roofs and wooden structures, further points to a specific location in Thailand. Square gray sidewalk tiles and non-motorized lanes marked with red asphalt are specific urban design features that help narrow down the location. Combining tropical vegetation, Thai-language road signs, traditional architecture, and specific urban design features, this image was most likely taken in a city in Bangkok, Thailand, Asia."""

# Standard CoT prompt: from paper's inference_example.py baseline
COT_PROMPT = """Analyze the following image. Use geographic elements such as landmarks, architecture, language, or other visual clues to determine the location.
If you cannot determine the location, please guess the answer from continent to city. Do not answer you cannot determine the location.
Output the chain of reasoning."""

# =================================================================
# PARSER
# =================================================================
ALL_COUNTRIES = {
    "kenya": "Kenya", "madagascar": "Madagascar", "ecuador": "Ecuador",
    "chile": "Chile", "brazil": "Brazil", "argentina": "Argentina",
    "peru": "Peru", "colombia": "Colombia", "united states": "United States",
    "usa": "United States", "canada": "Canada", "mexico": "Mexico",
    "germany": "Germany", "france": "France", "united kingdom": "United Kingdom",
    "spain": "Spain", "italy": "Italy", "japan": "Japan", "china": "China",
    "india": "India", "russia": "Russia", "south korea": "South Korea",
    "thailand": "Thailand", "australia": "Australia", "new zealand": "New Zealand",
    "south africa": "South Africa", "egypt": "Egypt", "nigeria": "Nigeria",
    "turkey": "Turkey", "indonesia": "Indonesia", "uganda": "Uganda",
    "finland": "Finland", "saudi arabia": "Saudi Arabia", "oman": "Oman",
    "kyrgyzstan": "Kyrgyzstan", "bolivia": "Bolivia", "paraguay": "Paraguay",
    "uruguay": "Uruguay", "venezuela": "Venezuela", "guyana": "Guyana",
    "colombia": "Colombia", "panama": "Panama", "costa rica": "Costa Rica",
    "guatemala": "Guatemala", "cuba": "Cuba", "jamaica": "Jamaica",
    "haiti": "Haiti", "dominican republic": "Dominican Republic",
    "portugal": "Portugal", "greece": "Greece", "netherlands": "Netherlands",
    "belgium": "Belgium", "switzerland": "Switzerland", "austria": "Austria",
    "poland": "Poland", "sweden": "Sweden", "norway": "Norway",
    "denmark": "Denmark", "finland": "Finland", "ireland": "Ireland",
    "ukraine": "Ukraine", "romania": "Romania", "hungary": "Hungary",
    "czech republic": "Czech Republic", "croatia": "Croatia",
    "philippines": "Philippines", "malaysia": "Malaysia", "vietnam": "Vietnam",
    "myanmar": "Myanmar", "cambodia": "Cambodia", "nepal": "Nepal",
    "sri lanka": "Sri Lanka", "bangladesh": "Bangladesh", "pakistan": "Pakistan",
    "iran": "Iran", "iraq": "Iraq", "israel": "Israel",
    # US states -> normalize
    "california": "United States", "nevada": "United States",
    "florida": "United States", "texas": "United States",
    "scotland": "United Kingdom", "wales": "United Kingdom",
}

CONTINENT_MAP = {
    "africa": "Africa", "asia": "Asia", "europe": "Europe",
    "north america": "North America", "south america": "South America",
    "oceania": "Oceania", "australia": "Oceania",
    "central america": "North America", "middle east": "Asia",
}

def parse_location(text):
    """Extract city, country, continent from model response."""
    if not text:
        return None, None, None

    text_lower = text.lower()
    country = None
    continent = None
    city = None

    # Strategy 1: "taken in" pattern (paper's format)
    m = re.search(r'taken in\s+(?:a city in\s+)?(?:the\s+)?(.+?),\s*(.+?),\s*(Africa|Asia|Europe|North America|South America|Oceania)', text, re.IGNORECASE)
    if m:
        city = m.group(1).strip()
        country = m.group(2).strip().rstrip('.')
        continent = m.group(3).strip()
        # Validate country
        c_lower = country.lower()
        if c_lower in ALL_COUNTRIES:
            country = ALL_COUNTRIES[c_lower]
        return city, country, continent

    # Strategy 2: "Location:" pattern
    m = re.search(r'[Ll]ocation:\s*(.+?)$', text, re.MULTILINE)
    if m:
        parts = [p.strip().rstrip('.') for p in m.group(1).split(',')]
        if len(parts) >= 3:
            city = parts[0]
            country = parts[-2]
            continent = parts[-1]
        elif len(parts) == 2:
            country = parts[0]
            continent = parts[1]
        if country:
            c_lower = country.lower()
            if c_lower in ALL_COUNTRIES:
                country = ALL_COUNTRIES[c_lower]
        return city, country, continent

    # Strategy 3: Search for country/continent in conclusion area
    conclusion = text[len(text)//2:]  # second half of text

    for ckey, cname in ALL_COUNTRIES.items():
        if ckey in conclusion.lower():
            country = cname
            break
    if not country:
        for ckey, cname in ALL_COUNTRIES.items():
            if ckey in text_lower:
                country = cname
                break

    for ckey, cname in CONTINENT_MAP.items():
        if ckey in text_lower:
            continent = cname
            break

    # Try to extract city
    if country:
        cl = country.lower()
        for term in [cl]:
            idx = text_lower.rfind(term)
            if idx > 0:
                snippet = text[max(0,idx-200):idx]
                cities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', snippet)
                skip = {"The","This","Step","Based","Image","However","Final","Prediction",
                       "Location","South","North","East","West","Combining","Visible","For",
                       "Let","Q1","Q2","Example","Based","After","Given"}
                for c in reversed(cities):
                    if c not in skip and len(c) > 2 and c.lower() != cl:
                        city = c
                        break

    return city, country, continent


def run_inference(data_rows, prompt, method_name, output_path, max_new_tokens=1280):
    """Run VLM inference on images."""
    print(f"\n--- {method_name} (max_new_tokens={max_new_tokens}) ---")

    results = []
    for i, row in enumerate(data_rows):
        img_path = row['image_path']
        gt = {
            "image_path": img_path,
            "ground_truth_city": row.get('city', ''),
            "ground_truth_country": row.get('country'),
            "ground_truth_continent": row.get('continent'),
            "ground_truth_lat": float(row['lat']) if row.get('lat') else None,
            "ground_truth_lon": float(row['lon']) if row.get('lon') else None,
        }

        if not os.path.exists(img_path):
            gt.update({"predicted_city": None, "predicted_country": None,
                       "predicted_continent": None, "error": "missing"})
            results.append(gt)
            continue

        try:
            t0 = time.time()
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": prompt}
            ]}]
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text_input], images=image_inputs, videos=None,
                             padding=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                        do_sample=True, temperature=0.7, top_p=0.9)
            trim = [o[len(inp):] for inp, o in zip(inputs["input_ids"], gen_ids)]
            response = processor.batch_decode(trim, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)[0]
            elapsed = time.time() - t0

            city, country, continent = parse_location(response)
            gt.update({
                "predicted_city": city,
                "predicted_country": country,
                "predicted_continent": continent,
                "model_response": response,
                "inference_time": round(elapsed, 2),
            })

        except Exception as e:
            print(f"  ERROR {os.path.basename(img_path)}: {e}")
            gt.update({"predicted_city": None, "predicted_country": None,
                       "predicted_continent": None, "error": str(e)})

        results.append(gt)

        if (i+1) % 5 == 0:
            c = gt.get('predicted_country', 'None')
            print(f"  [{i+1}/{len(data_rows)}] {os.path.basename(img_path)[:25]} -> {c} ({gt.get('inference_time',0):.0f}s)")
        if (i+1) % 10 == 0:
            with open(output_path, 'w') as f:
                json.dump(results, f)
            nv = sum(1 for r in results if r.get('predicted_country'))
            print(f"  Checkpoint: {i+1} done, {nv} valid")

    with open(output_path, 'w') as f:
        json.dump(results, f)
    nv = sum(1 for r in results if r.get('predicted_country'))
    print(f"  {method_name} DONE: {nv}/{len(results)} valid")
    return results


# Load dataset
data_rows = []
with open('/home/user/data/geoclip/sample_balanced.csv', 'r') as f:
    for row in csv.DictReader(f):
        data_rows.append(row)

countries = {}
for r in data_rows:
    countries[r['country']] = countries.get(r['country'], 0) + 1
print(f"Dataset: {len(data_rows)} images, Countries: {countries}")

# Determine token limit based on model size
max_tokens = 1536 if MODEL_SIZE == "32B" else 1280

# Run GeoCoT with paper's actual prompt
run_inference(data_rows, GEO_COT_PROMPT, f"GeoCoT_{MODEL_SIZE}",
              f"/home/user/results/geocot_{MODEL_SIZE}_predictions.json", max_tokens)

# Run standard CoT baseline
run_inference(data_rows, COT_PROMPT, f"CoT_{MODEL_SIZE}",
              f"/home/user/results/cot_{MODEL_SIZE}_predictions.json", max_tokens)

# Free GPU memory
del model
torch.cuda.empty_cache()

print("\n=== All Inference Complete ===")
PYEOF

# =============================================================================
# Evaluate
# =============================================================================
echo ""
echo "=== Evaluating Results ==="
bash /home/user/scripts/evaluate.sh /home/user/results

echo ""
echo "=========================================="
echo "Run complete - $(date)"
echo "=========================================="
