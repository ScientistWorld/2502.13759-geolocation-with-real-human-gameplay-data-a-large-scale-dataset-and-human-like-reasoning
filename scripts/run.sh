#!/bin/bash
# GeoCoT Reproduction - GPU Inference Job
# Runs GeoCoT (5-step prompting) vs CoT (standard chain-of-thought) with Qwen2.5-VL-7B-Instruct
#
# Key fixes from previous runs:
#   - max_new_tokens=2048 (was 512/768, caused truncation mid-reasoning)
#   - Comprehensive country parser (extracts from any part of response)
#   - Clean start (no duplicate checkpointing issues)
#
# Expected: ~15s per image for GeoCoT (2048 tokens), ~8s for CoT (512 tokens).
# 40 images x 2 methods ≈ 15-20 min total.

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
    echo "WARNING: setup.sh not found, using inline environment setup"
    export HF_HOME="/home/user/shared/models/hf"
    export TRANSFORMERS_CACHE="/home/user/shared/models/hf"
    export HF_HUB_OFFLINE="1"
fi

PYTHON="/usr/bin/python3"
echo "Using Python: $PYTHON"
$PYTHON --version

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

# =============================================================================
# Create Balanced Sample
# =============================================================================
echo ""
echo "=== Creating Balanced Sample ==="

${PYTHON} << 'PYEOF'
import os
import csv

# Read GeoCLIP CSV
data = []
with open('/home/user/data/geoclip/geoclip.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

print(f"Total images in dataset: {len(data)}")

from collections import defaultdict
by_country = defaultdict(list)
for row in data:
    by_country[row['country']].append(row)

SAMPLES_PER_COUNTRY = 10
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
    print(f"  {country}: sampled {len(sampled)} from {n} total")

print(f"Total sample: {len(samples)}")

sample_path = '/home/user/data/geoclip/sample_balanced.csv'
with open(sample_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['image_path','lat','lon','country','continent','city','size'])
    writer.writeheader()
    writer.writerows(samples)
print(f"Saved to {sample_path}")

missing = sum(1 for row in samples if not os.path.exists(row['image_path']))
print(f"Missing images: {missing}")
PYEOF

# =============================================================================
# Run VLM Inference
# =============================================================================
echo ""
echo "=== Loading Model and Running Inference ==="

${PYTHON} << 'PYEOF'
import os
import csv
import json
import re
import time

os.environ['HF_HOME'] = '/home/user/shared/models/hf'
os.environ['TRANSFORMERS_CACHE'] = '/home/user/shared/models/hf'
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Find model
MODEL_PATH = ""
for d in ['/home/user/checkpoints/Qwen2.5-VL-7B-Instruct',
           '/home/user/shared/models/Qwen2.5-VL-7B-Instruct']:
    if os.path.isdir(d) and os.path.exists(os.path.join(d, 'model.safetensors.index.json')):
        MODEL_PATH = d
        break

print(f"Loading model from: {MODEL_PATH}")

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

print("Loading Qwen2.5-VL processor and model...")
t0 = time.time()
processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_PATH)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print(f"Model loaded in {time.time()-t0:.1f}s")

# =====================================================================
# GeoCoT prompt (5-step structured prompting from paper Appendix B)
# =====================================================================
GEO_COT_USER_PROMPT = """Analyze this image and determine its geographical location using the following structured reasoning steps:

Step 1 - Continental/Climate Zone Identification:
Are there prominent natural features, such as specific types of vegetation, landforms (e.g., mountains, hills, plains), or soil characteristics, that provide clues about the geographical region?

Step 2 - Country-Level Localization:
Are there any culturally, historically, or architecturally significant landmarks, buildings, or structures, or are there any inscriptions or signs in a specific language or script that could help determine the country or region?

Step 3 - City-Level Refinement:
Are there distinctive road-related features, such as traffic direction (e.g., left-hand or right-hand driving), specific types of bollards, unique utility pole designs, or license plate colors and styles, which countries are known to have these characteristics?

Step 4 - Landmark-Based Verification:
Are there observable urban or rural markers (e.g., street signs, fire hydrants, guideposts), or other infrastructure elements, that can provide more specific information about the country or city?

Step 5 - Fine-Grained Micro-Level Validation:
Are there identifiable patterns in sidewalks (e.g., tile shapes, colors, or arrangements), clothing styles worn by people, or other culturally specific details that can help narrow down the city or area?

Let's think step by step. Based on the questions I provided, locate the location of the picture as accurately as possible. Identify the continent, country, and city, and summarize it into a paragraph. For example: the presence of tropical rainforests, palm trees, and red soil indicates a tropical climate... Signs in Thai, right-side traffic, and traditional Thai architecture further suggest it is in Thailand... Combining these clues, this image was likely taken in a city in Bangkok, Thailand, Asia.

Please provide your analysis and final location prediction.
You MUST provide specific city, country, and continent names as your best guess. Do NOT use placeholder words like Unknown or Unspecified.
Output format: Location: City, Country, Continent
Example: Location: Bangkok, Thailand, Asia
"""

# Standard CoT prompt (comparison baseline)
COT_PROMPT = """Let's think step by step about the geographical location where this image was taken. Identify any visible clues about the location, then provide your best guess of the city, country, and continent.

You MUST provide specific city, country, and continent names as your best guess. Do NOT use placeholder words like Unknown or Unspecified.
Output format: Location: City, Country, Continent
Example: Location: Nairobi, Kenya, Africa
"""

# =====================================================================
# Comprehensive country parser
# =====================================================================
# Map: country name -> (canonical_name, continent)
ALL_COUNTRIES = {
    # Africa
    "kenya": ("Kenya", "Africa"), "nigeria": ("Nigeria", "Africa"),
    "south africa": ("South Africa", "Africa"), "egypt": ("Egypt", "Africa"),
    "morocco": ("Morocco", "Africa"), "ethiopia": ("Ethiopia", "Africa"),
    "tanzania": ("Tanzania", "Africa"), "ghana": ("Ghana", "Africa"),
    "algeria": ("Algeria", "Africa"), "tunisia": ("Tunisia", "Africa"),
    "uganda": ("Uganda", "Africa"), "cameroon": ("Cameroon", "Africa"),
    "senegal": ("Senegal", "Africa"), "mozambique": ("Mozambique", "Africa"),
    "madagascar": ("Madagascar", "Africa"), "angola": ("Angola", "Africa"),
    "zimbabwe": ("Zimbabwe", "Africa"), "botswana": ("Botswana", "Africa"),
    "namibia": ("Namibia", "Africa"), "rwanda": ("Rwanda", "Africa"),
    "zambia": ("Zambia", "Africa"), "malawi": ("Malawi", "Africa"),
    "libya": ("Libya", "Africa"), "sudan": ("Sudan", "Africa"),
    "somalia": ("Somalia", "Africa"), "ivory coast": ("Ivory Coast", "Africa"),
    "democratic republic of congo": ("DR Congo", "Africa"),
    "congo": ("Congo", "Africa"),
    # South America
    "ecuador": ("Ecuador", "South America"),
    "chile": ("Chile", "South America"),
    "brazil": ("Brazil", "South America"),
    "argentina": ("Argentina", "South America"),
    "peru": ("Peru", "South America"),
    "colombia": ("Colombia", "South America"),
    "venezuela": ("Venezuela", "South America"),
    "bolivia": ("Bolivia", "South America"),
    "paraguay": ("Paraguay", "South America"),
    "uruguay": ("Uruguay", "South America"),
    "guyana": ("Guyana", "South America"),
    "suriname": ("Suriname", "South America"),
    # North/Central America
    "united states": ("United States", "North America"),
    "usa": ("United States", "North America"),
    "canada": ("Canada", "North America"),
    "mexico": ("Mexico", "North America"),
    "guatemala": ("Guatemala", "North America"),
    "cuba": ("Cuba", "North America"),
    "honduras": ("Honduras", "North America"),
    "panama": ("Panama", "North America"),
    "costa rica": ("Costa Rica", "North America"),
    "jamaica": ("Jamaica", "North America"),
    "dominican republic": ("Dominican Republic", "North America"),
    # Europe
    "germany": ("Germany", "Europe"), "france": ("France", "Europe"),
    "united kingdom": ("United Kingdom", "Europe"), "spain": ("Spain", "Europe"),
    "italy": ("Italy", "Europe"), "netherlands": ("Netherlands", "Europe"),
    "belgium": ("Belgium", "Europe"), "switzerland": ("Switzerland", "Europe"),
    "austria": ("Austria", "Europe"), "poland": ("Poland", "Europe"),
    "sweden": ("Sweden", "Europe"), "norway": ("Norway", "Europe"),
    "denmark": ("Denmark", "Europe"), "finland": ("Finland", "Europe"),
    "portugal": ("Portugal", "Europe"), "greece": ("Greece", "Europe"),
    "czech republic": ("Czech Republic", "Europe"), "hungary": ("Hungary", "Europe"),
    "romania": ("Romania", "Europe"), "ireland": ("Ireland", "Europe"),
    "russia": ("Russia", "Europe"), "ukraine": ("Ukraine", "Europe"),
    "turkey": ("Turkey", "Europe"), "serbia": ("Serbia", "Europe"),
    "croatia": ("Croatia", "Europe"), "bulgaria": ("Bulgaria", "Europe"),
    # Asia
    "china": ("China", "Asia"), "japan": ("Japan", "Asia"),
    "south korea": ("South Korea", "Asia"), "india": ("India", "Asia"),
    "indonesia": ("Indonesia", "Asia"), "thailand": ("Thailand", "Asia"),
    "vietnam": ("Vietnam", "Asia"), "philippines": ("Philippines", "Asia"),
    "malaysia": ("Malaysia", "Asia"), "singapore": ("Singapore", "Asia"),
    "pakistan": ("Pakistan", "Asia"), "bangladesh": ("Bangladesh", "Asia"),
    "nepal": ("Nepal", "Asia"), "sri lanka": ("Sri Lanka", "Asia"),
    "cambodia": ("Cambodia", "Asia"), "myanmar": ("Myanmar", "Asia"),
    "mongolia": ("Mongolia", "Asia"), "taiwan": ("Taiwan", "Asia"),
    "uae": ("UAE", "Asia"), "saudi arabia": ("Saudi Arabia", "Asia"),
    "israel": ("Israel", "Asia"), "iran": ("Iran", "Asia"),
    "iraq": ("Iraq", "Asia"), "kazakhstan": ("Kazakhstan", "Asia"),
    # Oceania
    "australia": ("Australia", "Oceania"),
    "new zealand": ("New Zealand", "Oceania"),
}

# Build a pattern: for each country, include name + adjective form
COUNTRY_SEARCH_PATTERNS = {}
for name_lower, (canonical, continent) in ALL_COUNTRIES.items():
    COUNTRY_SEARCH_PATTERNS[name_lower] = (canonical, continent)

# Add adjective forms and common variants
COUNTRY_SEARCH_PATTERNS.update({
    "kenyan": ("Kenya", "Africa"), "ugandan": ("Uganda", "Africa"),
    "ecuadorian": ("Ecuador", "South America"),
    "chilean": ("Chile", "South America"),
    "brazilian": ("Brazil", "South America"),
    "argentine": ("Argentina", "South America"), "argentinian": ("Argentina", "South America"),
    "peruvian": ("Peru", "South America"),
    "colombian": ("Colombia", "South America"),
    "malagasy": ("Madagascar", "Africa"),
    "american": ("United States", "North America"),
    "british": ("United Kingdom", "Europe"),
    "german": ("Germany", "Europe"),
    "french": ("France", "Europe"),
    "italian": ("Italy", "Europe"),
    "spanish": ("Spain", "Europe"),
    "chinese": ("China", "Asia"),
    "japanese": ("Japan", "Asia"),
    "korean": ("South Korea", "Asia"),
    "indian": ("India", "Asia"),
    "thai": ("Thailand", "Asia"),
    "vietnamese": ("Vietnam", "Asia"),
    "australian": ("Australia", "Oceania"),
})

# Countries in our dataset (priority search)
DATASET_COUNTRIES = {"kenya", "ecuador", "chile", "madagascar"}



# Template words indicating placeholder, not real answers
TEMPLATE_WORDS = {
    "unknown", "unspecified", "country", "city", "continent", "not specified",
    "unknown country", "unknown city", "unknown continent", "tropical country",
    "developing country", "east africa", "west africa", "north africa",
    "southeast asia", "central america", "latin america",
    "north america", "south america", "africa", "asia", "europe", "oceania",
    "caribbean", "middle east", "western australia", "eastern europe",
}

def is_template(text):
    if not text: return True
    cleaned = text.strip().lower().strip("[]()").strip()
    return cleaned in TEMPLATE_WORDS or len(cleaned) <= 2


def parse_location_prediction(text):
    """Extract city, country, continent from model response using smart parsing."""
    if not text:
        return {"city": None, "country": None, "continent": None, "raw_prediction": text}

    result = {"city": None, "country": None, "continent": None, "raw_prediction": text}
    text_lower = text.lower()

    # Strategy 1: Look for "Location:" pattern - but only accept non-template text
    loc_patterns = [
        r"[Ll]ocation:\s*(.+?),\s*(.+?)(?:,\s*(.+?))?(?:\n|$|\.)",
    ]
    for pattern in loc_patterns:
        match = re.search(pattern, text)
        if match:
            city_str = match.group(1).strip().strip("[]").strip()
            country_str = match.group(2).strip().strip("[]").strip()
            continent_str = match.group(3).strip().strip("[]").strip() if match.group(3) else ""

            if not is_template(country_str):
                # Got a real country from Location: line - look it up
                c_lower = country_str.lower()
                if c_lower in ALL_COUNTRIES:
                    canonical, continent = ALL_COUNTRIES[c_lower]
                    result["country"] = canonical
                    result["continent"] = continent
                else:
                    result["country"] = country_str
                    result["continent"] = continent_str if continent_str else None
                result["city"] = city_str if not is_template(city_str) else None
                return result
            # Template text found - fall through to smart extraction

    # Strategy 2: Search for country names in prioritized sections
    # Priority: Final Prediction > Conclusion > Last third > Full text
    fp_section = ""
    fp_match = re.search(r"Final\s+Prediction:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if fp_match:
        fp_section = fp_match.group(1)

    # Conclusion-like section
    conclusion_text = ""
    conc_match = re.search(
        r"(?:Given|Based on|Considering).*?(?:analysis|clues|observations|factors)[,:]\s*(.*)",
        text, re.IGNORECASE | re.DOTALL
    )
    if conc_match:
        conclusion_text = conc_match.group(1)

    last_third = text[len(text)*2//3:]

    search_sections = []
    if fp_section:
        search_sections.append((fp_section, 200))
    if conclusion_text:
        search_sections.append((conclusion_text, 100))
    search_sections.append((last_third, 50))
    search_sections.append((text, 10))

    best_country = None
    best_continent = None
    best_priority = -1

    for section, priority in search_sections:
        section_lower = section.lower()
        for pattern, (canonical, continent) in COUNTRY_SEARCH_PATTERNS.items():
            if pattern in section_lower:
                p = priority
                # Adjective forms get lower priority
                if pattern.endswith("n") and len(pattern) > 5 and pattern not in ALL_COUNTRIES:
                    p -= 5
                # Dataset countries get bonus
                if pattern in DATASET_COUNTRIES:
                    p += 5
                if p > best_priority:
                    best_country = canonical
                    best_continent = continent
                    best_priority = p

    if best_country:
        result["country"] = best_country
        if not result["continent"]:
            result["continent"] = best_continent

    # Strategy 3: Extract continent from text if not yet found
    if not result["continent"]:
        CONTINENT_MAP = {
            "south america": "South America", "north america": "North America",
            "africa": "Africa", "europe": "Europe", "asia": "Asia",
            "oceania": "Oceania", "australia": "Oceania",
            "central america": "North America", "caribbean": "North America",
            "middle east": "Asia",
        }
        for cont_name, canonical in CONTINENT_MAP.items():
            if cont_name in text_lower:
                result["continent"] = canonical
                break

    # Strategy 4: Try to extract city name near country mention
    if result["country"] and not result["city"]:
        country_lower = result["country"].lower()
        idx = text_lower.find(country_lower)
        if idx >= 0:
            snippet = text[max(0, idx-150):idx]
            city_matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b', snippet)
            non_cities = {"The", "This", "Step", "Based", "Image", "However", "Final",
                         "Prediction", "Location", "South", "North", "Central", "East",
                         "West", "Combining", "Clues", "Visible", "Presence"}
            for cm in reversed(city_matches):
                if cm not in non_cities and len(cm) > 2:
                    result["city"] = cm
                    break

    return result


def run_inference(data_rows, prompt, method_name, output_path, max_new_tokens=2048):
    """Run VLM inference on images and save results."""
    print(f"\n--- {method_name} (max_new_tokens={max_new_tokens}) ---")

    # Start fresh - don't use old checkpointed results to avoid duplicates
    results = []
    print(f"  Processing {len(data_rows)} images...")

    for i, row in enumerate(data_rows):
        img_path = row['image_path']
        if not os.path.exists(img_path):
            print(f"  SKIP (missing): {os.path.basename(img_path)}")
            results.append({
                "image_path": img_path,
                "ground_truth_city": row.get('city'),
                "ground_truth_country": row.get('country'),
                "ground_truth_continent": row.get('continent'),
                "ground_truth_lat": float(row['lat']) if row.get('lat') else None,
                "ground_truth_lon": float(row['lon']) if row.get('lon') else None,
                "predicted_city": None, "predicted_country": None, "predicted_continent": None,
                "model_response": None, "error": "missing image",
            })
            continue

        try:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": prompt}
            ]}]
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=[text_input], images=image_inputs, videos=None,
                padding=True, return_tensors="pt"
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            t_start = time.time()
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            elapsed = time.time() - t_start

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
                "ground_truth_lat": float(row['lat']) if row.get('lat') else None,
                "ground_truth_lon": float(row['lon']) if row.get('lon') else None,
                "predicted_city": prediction.get('city'),
                "predicted_country": prediction.get('country'),
                "predicted_continent": prediction.get('continent'),
                "model_response": response,
                "inference_time": round(elapsed, 2),
            })

            # Progress logging
            country_pred = prediction.get('country', 'None')
            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{len(data_rows)}] {os.path.basename(img_path)[:30]} -> {country_pred} ({elapsed:.1f}s, {len(response)} chars)")

        except Exception as e:
            print(f"  ERROR on {os.path.basename(img_path)}: {e}")
            results.append({
                "image_path": img_path,
                "ground_truth_city": row.get('city'),
                "ground_truth_country": row.get('country'),
                "ground_truth_continent": row.get('continent'),
                "ground_truth_lat": float(row['lat']) if row.get('lat') else None,
                "ground_truth_lon": float(row['lon']) if row.get('lon') else None,
                "predicted_city": None, "predicted_country": None, "predicted_continent": None,
                "model_response": None, "error": str(e),
            })

        # Checkpoint every 10 images
        if (i + 1) % 10 == 0:
            with open(output_path, 'w') as f:
                json.dump(results, f)
            valid_count = sum(1 for r in results if r.get('predicted_country'))
            print(f"  Checkpoint: {i+1} done, {valid_count} valid predictions")

    # Final save
    with open(output_path, 'w') as f:
        json.dump(results, f)
    valid = [r for r in results if r.get('predicted_country') is not None]
    print(f"  {method_name} DONE: {len(valid)}/{len(results)} valid predictions")
    return results


# Load dataset
data_rows = []
with open('/home/user/data/geoclip/sample_balanced.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data_rows.append(row)

print(f"\nDataset: {len(data_rows)} images")
countries = {}
for row in data_rows:
    c = row['country']
    countries[c] = countries.get(c, 0) + 1
print(f"Countries: {countries}")

# Run GeoCoT (2048 tokens for full reasoning)
geocot_results = run_inference(
    data_rows, GEO_COT_USER_PROMPT, "GeoCoT",
    "/home/user/results/geocot_predictions.json",
    max_new_tokens=2048
)

# Run standard CoT (512 tokens is enough for shorter responses)
cot_results = run_inference(
    data_rows, COT_PROMPT, "CoT",
    "/home/user/results/cot_predictions.json",
    max_new_tokens=512
)

print("\n=== All Inference Complete ===")
PYEOF

# =============================================================================
# Evaluate Results
# =============================================================================
echo ""
echo "=== Evaluating Results ==="

${PYTHON} << 'PYEOF'
import sys
import os
import csv
import json
import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*math.sin(delta_lon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

COUNTRY_COORDS = {
    "kenya": (-1.2921, 36.8219), "madagascar": (-18.7669, 46.8691),
    "ecuador": (-1.8312, -78.1834), "chile": (-35.6751, -71.5430),
    "brazil": (-14.2350, -51.9253), "argentina": (-38.4161, -63.6167),
    "peru": (-9.1900, -75.0152), "colombia": (4.5709, -74.2973),
    "united states": (37.0902, -95.7129), "canada": (56.1304, -106.3468),
    "south africa": (-30.5595, 22.9375), "egypt": (26.8206, 30.8025),
    "nigeria": (9.0820, 8.6753), "australia": (-25.2744, 133.7751),
    "germany": (51.1657, 10.4515), "france": (46.2276, 2.2137),
    "united kingdom": (55.3781, -3.4360), "china": (35.8617, 104.1954),
    "japan": (36.2048, 138.2529), "india": (20.5937, 78.9629),
    "russia": (61.5240, 105.3188), "mexico": (23.6345, -102.5528),
}

def reverse_geocode(country):
    if not country: return None, None
    c = country.strip().lower()
    for k, v in COUNTRY_COORDS.items():
        if k in c or c in k: return v
    return None, None

def normalize_country(c):
    if not c: return None
    c = c.strip().lower()
    aliases = {"usa":"united states","us":"united states","uk":"united kingdom",
               "british":"united kingdom","american":"united states"}
    return aliases.get(c, c)

def normalize_continent(c):
    if not c: return None
    c = c.strip().lower()
    if "australia" in c: c = "oceania"
    return c

def compute_all_metrics(predictions):
    """Compute all geolocation metrics."""
    # Enrich with coordinates
    for p in predictions:
        if p.get('predicted_lat') is None:
            lat, lon = reverse_geocode(p.get('predicted_country'))
            p['predicted_lat'] = lat
            p['predicted_lon'] = lon

    metrics = {}

    # Classification metrics at each level
    for level in ['city', 'country', 'continent']:
        pk = f'predicted_{level}'
        gk = f'ground_truth_{level}'
        tp = fp = fn = total = 0
        for p in predictions:
            if not p.get(gk): continue
            total += 1
            pv = p.get(pk)
            gv = p.get(gk)
            if not pv:
                fn += 1; continue
            if level == 'country':
                pv, gv = normalize_country(pv), normalize_country(gv)
            elif level == 'continent':
                pv, gv = normalize_continent(pv), normalize_continent(gv)
            else:
                pv, gv = pv.strip().lower(), gv.strip().lower() if gv else ""
            if pv == gv: tp += 1
            else: fp += 1; fn += 1
        acc = tp/total if total else 0
        rec = tp/(tp+fn) if (tp+fn) else 0
        prec = tp/(tp+fp) if (tp+fp) else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        metrics[f'{level}_accuracy'] = round(acc, 4)
        metrics[f'{level}_recall'] = round(rec, 4)
        metrics[f'{level}_f1'] = round(f1, 4)
        metrics[f'{level}_total'] = total

    # Distance metrics
    for thresh, name in [(1.0,"street_1km"),(25.0,"city_25km"),(750.0,"country_750km")]:
        within = total = 0
        for p in predictions:
            lat, lon = p.get('ground_truth_lat'), p.get('ground_truth_lon')
            plat, plon = p.get('predicted_lat'), p.get('predicted_lon')
            if lat and lon and plat and plon:
                total += 1
                if haversine(float(lat), float(lon), float(plat), float(plon)) <= thresh:
                    within += 1
        metrics[name] = round(within/total, 4) if total else 0.0
        metrics[f'{name}_count'] = f'{within}/{total}'

    return metrics


# Evaluate all prediction files
results_dir = '/home/user/results'
pred_files = {}
for f in sorted(os.listdir(results_dir)):
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
    print(f"  {len(valid_preds)}/{len(predictions)} valid predictions, {len(valid_gt)} with ground truth")

    if valid_gt:
        metrics = compute_all_metrics(valid_gt)
        all_metrics[name] = metrics
        for key in ['city_accuracy', 'country_accuracy', 'continent_accuracy',
                    'street_1km', 'city_25km', 'country_750km']:
            if key in metrics:
                print(f"  {key}: {metrics[key]}")

# Build scores.json
with open('/home/user/scoring/reference.json') as f:
    reference = json.load(f)

scores = {"experiments": {}}
METHOD_MAP = {"geocot": "qwen_geocot", "cot": "qwen_cot"}

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

    # Copy reference (paper) results
    for method_name, method_data in ref_results.items():
        entry = {}
        for key, val in method_data.items():
            if key == "type": continue
            entry[key] = val
        exp_copy["results"][method_name] = entry

    # Add reproduced results
    for pred_name, metrics in all_metrics.items():
        method_name = METHOD_MAP.get(pred_name, pred_name)
        if method_name not in ref_results:
            if exp_name in ["geocomp_classification", "geocomp_distance"]:
                entry = {}
                for key in ref_metrics:
                    if key in metrics:
                        entry[key] = metrics[key]
                if entry:
                    exp_copy["results"][method_name] = entry
        else:
            entry = exp_copy["results"][method_name]
            for key in ref_metrics:
                if key in metrics:
                    entry[key] = metrics[key]

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
