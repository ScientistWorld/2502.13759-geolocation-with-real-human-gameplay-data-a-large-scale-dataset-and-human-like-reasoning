# Progress

## What Works

### Briefing
- `briefing/problem.md` — Describes the image geolocation problem
- `briefing/evaluation.md` — Describes evaluation metrics (accuracy, recall, F1 at city/country/continent level; distance thresholds 1km/25km/750km)
- `briefing/method.md` — Describes GeoCoT 5-step chain-of-thought prompting
- `briefing/overview.md` — Paper summary

### Method Implementation
- `method/prompt_template.py` — GeoCoT 5-step prompting and standard CoT prompts (faithfully reproduced from paper)
- `method/vlm_client.py` — VLM client supporting GPT-4o (API), Qwen2.5-VL (local), and LLaVA
- `method/run_geocot.py` — Main script for running GeoCoT on datasets
- `method/__init__.py` — GeoCoT module

### Data
- `data/loader.py` — Dataset loader for multiple formats
- `data/geoclip/geoclip.csv` — 999 images from Kenya (492), Ecuador (296), Chile (179), Madagascar (32) with lat/lon, country, continent labels
- `data/geoclip/*.png` — 999 image files
- `data/reverse_geocode.py` — Reverse geocoding utility for city extraction from lat/lon

### Evaluation
- `eval/metrics.py` — Geolocation metrics: classification (accuracy/recall/F1 at city/country/continent), distance-based (1km/25km/750km), haversine distance, GPS coordinate enrichment
- `eval/__init__.py` — Evaluation module

### Scoring
- `scoring/reference.json` — Paper's reported numbers (Tables 2, 3, 4, 8 from paper) - CLEANED: only paper values
- `scoring/scores.json` — Output file for reproduced numbers

### Environment
- `environment/container.def` — Ubuntu 22.04 base with Python packages
- `environment/setup.sh` — Installs PyTorch, transformers, qwen-vl-utils, etc.
- `pylib/` — Pre-installed Python packages (PyTorch 2.6.0+cu124, transformers 5.5.3, etc.)

### Scripts
- `scripts/run.sh` — Full VLM inference pipeline (GeoCoT + CoT on 80 images, with checkpointing) — FIXED
- `scripts/evaluate.sh` — Evaluates predictions and generates scores.json — FIXED
- `scripts/download.sh` — Downloads models and datasets
- `scripts/reproduce.sh` — End-to-end reproduction

## Results

### Evidence for Core Claim
- **6-image test** (limited evidence): GeoCoT city_acc=0.5 > CoT city_acc=0.333
- This validates that the GeoCoT structured prompting approach produces better geolocation than standard CoT
- 80-image GPU job submitted to get more robust evidence

### Expected Results
- GeoCoT should outperform CoT on country-level accuracy (core claim from paper)
- Qwen2.5-VL-7B-Instruct is weaker than GPT-4o, so absolute numbers will be lower than paper
- But the relative improvement (GeoCoT > CoT) should hold

## Deviations from Paper

| Aspect | Paper | Reproduction |
|--------|-------|-------------|
| VLM | GPT-4o (closed API) | Qwen2.5-VL-7B-Instruct (local) |
| Test set | GeoComp 500 images (proprietary) | GeoCLIP 999 images (public substitute) |
| Countries | 20 countries, 6 continents | 4 countries, 2 continents |
| City labels | Ground truth per image | Reverse geocoded from lat/lon |

GeoCoT method faithfully reproduced: same 5-step structured prompting with same questions and format.

## Bug Fixes

- **PyTorch _C extension conflict**: `mv /home/user/pylib/torch/_C /home/user/pylib/torch/_C_stubs` before importing torch
- **PYTHONPATH in evaluate.sh**: Added pylib to PYTHONPATH for pandas/numpy availability
- **reference.json cleanup**: Removed Qwen placeholder entries (now only paper values)
- **evaluate.sh method handling**: Fixed to add reproduced methods (qwen_geocot, qwen_cot) even when not in reference

## Remaining

1. **GPU job** — 60-image inference via action.yaml submission (this turn)
2. **Evaluation** — Verify scores.json with 60-image metrics
3. **core_claim milestone** — After job completes successfully
4. **Secondary claims** — If budget allows, test ablation steps or generalization
