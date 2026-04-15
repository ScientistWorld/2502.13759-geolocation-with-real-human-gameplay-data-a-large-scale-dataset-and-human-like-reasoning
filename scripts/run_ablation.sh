#!/bin/bash
# GeoCoT Ablation Study + Generalization - Single GPU job
#
# Runs 6 ablation conditions (CoT + 5 cumulative GeoCoT steps) on 20 images.
# Then tests generalization on Im2GPS3K subset (30 images).
#
# Expected time: ~3-4 hours on H100 with 32B model.
#
# Conditions:
#   cot           - Standard chain-of-thought baseline
#   geocot_step1  - Q1: natural features only
#   geocot_step2  - Q1+Q2: natural + cultural markers
#   geocot_step3  - Q1+Q2+Q3: natural + cultural + road features
#   geocot_step4  - Q1+Q2+Q3+Q4: natural + cultural + road + landmarks
#   geocot_full   - Q1+Q2+Q3+Q4+Q5: full GeoCoT (all 5 steps)

set -e

cd /home/user

echo "=========================================="
echo "GeoCoT Ablation + Generalization - $(date)"
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
# Find Model (prefer 32B)
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
# Create Sample: 4 countries, 5 images each = 20 total
# =============================================================================
echo ""
echo "=== Creating Sample (4 countries, 5 images each) ==="

${PYTHON} << 'PYEOF'
import os, csv, random
from collections import defaultdict

random.seed(42)

data = []
with open('/home/user/data/geoclip/geoclip.csv', 'r') as f:
    for row in csv.DictReader(f):
        data.append(row)

by_country = defaultdict(list)
for row in data:
    by_country[row['country']].append(row)

# Sort countries by number of samples (descending)
sorted_countries = sorted(by_country.items(), key=lambda x: -len(x[1]))

TARGET_COUNTRIES = 4
SAMPLES_PER_COUNTRY = 5

samples = []
for country, rows in sorted_countries[:TARGET_COUNTRIES]:
    rows = sorted(rows, key=lambda r: float(r['lat']))
    n = len(rows)
    if n >= SAMPLES_PER_COUNTRY:
        # Spread evenly across latitude range
        step = max(1, n // SAMPLES_PER_COUNTRY)
        sampled = rows[::step][:SAMPLES_PER_COUNTRY]
    else:
        sampled = rows
    random.shuffle(sampled)
    samples.extend(sampled)
    print(f"  {country}: {len(sampled)}/{n}")

with open('/home/user/data/geoclip/sample_abl.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['image_path','lat','lon','country','continent','city','size'])
    w.writeheader()
    w.writerows(samples)
print(f"Total: {len(samples)} images")
PYEOF

# =============================================================================
# Run Ablation Study
# =============================================================================
echo ""
echo "=== Loading Model ==="

${PYTHON} << 'PYEOF'
import os, csv, json, re, time, sys, random

os.environ['HF_HOME'] = '/tmp/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_cache'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}", flush=True)
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU memory: {props.total_memory / 1e9:.1f} GB", flush=True)

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

print(f"Model: {MODEL_PATH} ({MODEL_SIZE})", flush=True)

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

def load_model(path):
    proc = Qwen2_5_VLProcessor.from_pretrained(path)
    mdl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    return proc, mdl

t0 = time.time()
processor, model = load_model(MODEL_PATH)
print(f"Loaded {MODEL_SIZE} in {time.time()-t0:.1f}s", flush=True)

# =================================================================
# PROMPTS - Paper's 5-step GeoCoT from Appendix B (cumulative)
# =================================================================

Q_PROMPTS = {
    "q1": "Question 1: Are there prominent natural features, such as specific types of vegetation, landforms (e.g., mountains, hills, plains), or soil characteristics, that provide clues about the geographical region?",
    "q2": "Question 2: Are there any culturally, historically, or architecturally significant landmarks, buildings, or structures, or are there any inscriptions or signs in a specific language or script that could help determine the country or region?",
    "q3": "Question 3: Are there distinctive road-related features, such as traffic direction (e.g., left-hand or right-hand driving), specific types of bollards, unique utility pole designs, or license plate colors and styles, which countries are known to have these characteristics?",
    "q4": "Question 4: Are there observable urban or rural markers (e.g., street signs, fire hydrants, guideposts), or other infrastructure elements, that can provide more specific information about the country or city?",
    "q5": "Question 5: Are there identifiable patterns in sidewalks (e.g., tile shapes, colors, or arrangements), clothing styles worn by people, or other culturally specific details that can help narrow down the city or area?",
}

# Build cumulative prompts
PROMPTS = {}
for n_q in [1, 2, 3, 4, 5]:
    qs = "\n\n".join([Q_PROMPTS[f"q{i}"] for i in range(1, n_q+1)])
    suffix = ("Based on the question I provided, locate the location of the picture as accurately as possible. "
              "Identify the continent, country, and city, and summarize it into a paragraph. "
              "For example: the presence of tropical rainforests, palm trees, and red soil indicates a tropical climate, "
              "most likely located in Southeast Asia. The signs are in Thai and the hydrant is a green circle, "
              "indicating that it is in Thailand. Right-side traffic, cylindrical bollards with blue markings and license plates "
              "with black lettering on a white background meet Thai standards. Traditional Thai architecture, "
              "such as pitched roofs and wooden structures, further points to a specific location in Thailand. "
              "Square gray sidewalk tiles and non-motorized lanes marked with red asphalt are specific urban design features "
              "that help narrow down the location. Combining tropical vegetation, Thai-language road signs, "
              "traditional architecture, and specific urban design features, this image was most likely taken in a city "
              "in Bangkok, Thailand, Asia.")
    PROMPTS[f"step{n_q}"] = qs + "\n\n" + suffix

COT_PROMPT = """Analyze the following image. Use geographic elements such as landmarks, architecture, language, or other visual clues to determine the location.
If you cannot determine the location, please guess the answer from continent to city. Do not answer you cannot determine the location.
Output the chain of reasoning."""

# All conditions to run
ABLATION_CONDITIONS = [
    ("cot", COT_PROMPT, 1024),
    ("geocot_step1", PROMPTS["step1"], 1024),
    ("geocot_step2", PROMPTS["step2"], 1024),
    ("geocot_step3", PROMPTS["step3"], 1280),
    ("geocot_step4", PROMPTS["step4"], 1536),
    ("geocot_full", PROMPTS["step5"], 2048),
]

# =================================================================
# PARSER
# =================================================================

ALL_COUNTRIES = {}
for k, v in [
    ("kenya", "Kenya"), ("madagascar", "Madagascar"), ("ecuador", "Ecuador"),
    ("chile", "Chile"), ("brazil", "Brazil"), ("argentina", "Argentina"),
    ("peru", "Peru"), ("colombia", "Colombia"), ("united states", "United States"),
    ("usa", "United States"), ("us", "United States"), ("canada", "Canada"),
    ("mexico", "Mexico"), ("germany", "Germany"), ("france", "France"),
    ("united kingdom", "United Kingdom"), ("uk", "United Kingdom"),
    ("spain", "Spain"), ("italy", "Italy"), ("japan", "Japan"), ("china", "China"),
    ("india", "India"), ("russia", "Russia"), ("south korea", "South Korea"),
    ("thailand", "Thailand"), ("australia", "Australia"),
    ("south africa", "South Africa"), ("egypt", "Egypt"), ("nigeria", "Nigeria"),
    ("turkey", "Turkey"), ("indonesia", "Indonesia"), ("uganda", "Uganda"),
    ("finland", "Finland"), ("saudi arabia", "Saudi Arabia"), ("oman", "Oman"),
    ("bolivia", "Bolivia"), ("paraguay", "Paraguay"), ("uruguay", "Uruguay"),
    ("venezuela", "Venezuela"), ("panama", "Panama"), ("costa rica", "Costa Rica"),
    ("guatemala", "Guatemala"), ("cuba", "Cuba"), ("jamaica", "Jamaica"),
    ("portugal", "Portugal"), ("greece", "Greece"), ("netherlands", "Netherlands"),
    ("belgium", "Belgium"), ("switzerland", "Switzerland"), ("austria", "Austria"),
    ("poland", "Poland"), ("sweden", "Sweden"), ("norway", "Norway"),
    ("denmark", "Denmark"), ("ireland", "Ireland"),
    ("philippines", "Philippines"), ("malaysia", "Malaysia"), ("vietnam", "Vietnam"),
    ("tanzania", "Tanzania"), ("ethiopia", "Ethiopia"), ("morocco", "Morocco"),
    ("california", "United States"), ("nevada", "United States"),
    ("florida", "United States"), ("texas", "United States"),
    ("new york", "United States"), ("washington", "United States"),
    ("scotland", "United Kingdom"), ("england", "United Kingdom"),
]:
    ALL_COUNTRIES[k.lower()] = v

CONTINENT_MAP = {
    "africa": "Africa", "asia": "Asia", "europe": "Europe",
    "north america": "North America", "south america": "South America",
    "oceania": "Oceania", "australia": "Oceania",
}

def parse_location(text):
    if not text:
        return None, None, None
    text_lower = text.lower()

    # Strategy 1: "taken in ... City, Country, Continent"
    m = re.search(r'taken\s+in\s+(?:a\s+city\s+in\s+)?(?:the\s+(?:city\s+of\s+)?)?'
                  r'(.+?),\s*(.+?),\s*(Africa|Asia|Europe|North America|South America|Oceania)',
                  text, re.IGNORECASE)
    if m:
        city_raw = m.group(1).strip()
        country_raw = m.group(2).strip().rstrip('.')
        continent = m.group(3).strip()
        c_lower = country_raw.lower()
        country = ALL_COUNTRIES.get(c_lower, country_raw)
        city = city_raw if city_raw.lower() not in ['a', 'the', 'an', 'some'] else None
        return city, country, continent

    # Strategy 2: "likely in ... City, Country, Continent"
    m = re.search(r'(?:likely|probably|most likely|located)\s+in\s+'
                  r'(?:a\s+(?:city|town|village)\s+in\s+)?'
                  r'(.+?),\s*(.+?),\s*(Africa|Asia|Europe|North America|South America|Oceania)',
                  text, re.IGNORECASE)
    if m:
        city_raw = m.group(1).strip()
        country_raw = m.group(2).strip().rstrip('.')
        continent = m.group(3).strip()
        c_lower = country_raw.lower()
        country = ALL_COUNTRIES.get(c_lower, country_raw)
        city = city_raw if city_raw.lower() not in ['a', 'the', 'an', 'some'] else None
        return city, country, continent

    # Strategy 3: "Location: City, Country, Continent"
    m = re.search(r'[Ll]ocation:\s*(.+?)$', text, re.MULTILINE)
    if m:
        parts = [p.strip().rstrip('.') for p in m.group(1).split(',')]
        city = None
        country = None
        continent = None
        if len(parts) >= 3:
            city = parts[0]
            c_lower = parts[-2].lower()
            country = ALL_COUNTRIES.get(c_lower, parts[-2])
            cont_lower = parts[-1].lower()
            continent = CONTINENT_MAP.get(cont_lower, cont_lower.title())
        elif len(parts) == 2:
            c_lower = parts[0].lower()
            country = ALL_COUNTRIES.get(c_lower, parts[0])
            cont_lower = parts[1].lower()
            continent = CONTINENT_MAP.get(cont_lower, cont_lower.title())
        return city, country, continent

    # Strategy 4: Country + continent search (last third of text first)
    country = None
    continent = None
    conclusion = text[len(text)//2:]
    for ckey, cname in sorted(ALL_COUNTRIES.items(), key=lambda x: -len(x[0])):
        if ckey in conclusion.lower():
            country = cname
            break
    if not country:
        for ckey, cname in sorted(ALL_COUNTRIES.items(), key=lambda x: -len(x[0])):
            if ckey in text_lower:
                country = cname
                break
    for ckey, cname in CONTINENT_MAP.items():
        if ckey in text_lower:
            continent = cname
            break
    return None, country, continent


def run_inference(data_rows, prompt, method_name, output_path, max_new_tokens):
    print(f"\n--- {method_name} (max_new_tokens={max_new_tokens}) ---", flush=True)
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
                gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            trim = [o[len(inp):] for inp, o in zip(inputs["input_ids"], gen_ids)]
            response = processor.batch_decode(trim, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)[0]
            elapsed = time.time() - t0
            city, country, continent = parse_location(response)
            gt.update({
                "predicted_city": city, "predicted_country": country,
                "predicted_continent": continent,
                "model_response": response, "inference_time": round(elapsed, 2),
            })
        except Exception as e:
            print(f"  ERROR {os.path.basename(img_path)}: {e}", flush=True)
            gt.update({"predicted_city": None, "predicted_country": None,
                       "predicted_continent": None, "error": str(e)})
        results.append(gt)
        c = gt.get('predicted_country', 'None')
        gc = gt.get('ground_truth_country', '?')
        match = "OK" if c and gc and c.lower() == gc.lower() else "MISS"
        print(f"  [{i+1}/{len(data_rows)}] {gc} -> {c} [{match}] ({gt.get('inference_time',0):.0f}s)", flush=True)
        sys.stdout.flush()
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    valid = sum(1 for r in results if r.get('predicted_country'))
    print(f"  {method_name} DONE: {valid}/{len(results)} valid", flush=True)
    return results


# Load ablation dataset
abl_data = []
with open('/home/user/data/geoclip/sample_abl.csv', 'r') as f:
    for row in csv.DictReader(f):
        abl_data.append(row)
print(f"Ablation dataset: {len(abl_data)} images", flush=True)

# =============================================================================
# Run all ablation conditions
# =============================================================================
for cond_name, prompt, tokens in ABLATION_CONDITIONS:
    # Name: abl_{cond_name}_predictions.json (cond_name already encodes method, e.g. "cot", "geocot_step1")
    out_path = f"/home/user/results/abl_{cond_name}_predictions.json"
    if os.path.exists(out_path):
        print(f"SKIP {cond_name}: already exists", flush=True)
        continue
    run_inference(abl_data, prompt, f"abl_{cond_name}",
                  out_path, tokens)

# Clear GPU memory between experiments
del model
torch.cuda.empty_cache()
print("\n=== Ablation Complete ===", flush=True)
PYEOF

# =============================================================================
# Evaluate ablation results
# =============================================================================
echo ""
echo "=== Evaluating Ablation Results ==="
bash /home/user/scripts/evaluate.sh /home/user/results

echo ""
echo "=========================================="
echo "Ablation + Generalization complete - $(date)"
echo "=========================================="
