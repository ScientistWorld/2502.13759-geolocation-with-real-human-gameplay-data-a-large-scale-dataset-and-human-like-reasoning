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
- `eval/metrics.py` — Geolocation metrics: classification (accuracy/recall/F1 at city/country/continent), distance-based (1km/25km/750km), haversine distance
- `eval/__init__.py` — Evaluation module

### Scoring
- `scoring/reference.json` — Paper's reported numbers (Tables 2, 3, 4, 8)
- `scoring/scores.json` — Output file for reproduced numbers

### Environment
- `environment/container.def` — Ubuntu 22.04 base with Python packages
- `environment/setup.sh` — Installs PyTorch, transformers, qwen-vl-utils, etc.
- `pylib/` — Pre-installed Python packages (PyTorch 2.6.0+cu124, transformers 5.5.3, etc.)

### Scripts
- `scripts/run.sh` — Full VLM inference pipeline (GeoCoT + CoT on 80 images, with checkpointing)
- `scripts/evaluate.sh` — Evaluates predictions and generates scores.json
- `scripts/download.sh` — Downloads models and datasets
- `scripts/reproduce.sh` — End-to-end reproduction (to be completed)

## Results

Previous GPU job ran only 6 images (buggy). Current job running 80 balanced images.

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

- **Critical**: Previous run.sh had `export CUDA_VISIBLE_DEVICES=""` which blocked GPU usage. Removed.
- **Checkpointing**: Now saves every 10 images instead of every 20.
- **Sample size**: Reduced from 160 to 80 for better time management.

## Remaining

1. **GPU job** — Run GeoCoT and CoT inference on 80 images (submitted)
2. **Evaluation** — Verify scores.json with real metrics
3. **Validation** — Run `python validate.py --compare`
4. **Secondary claims** — If budget allows, test on more countries or additional metrics
5. **Wrap-up** — Update MILESTONES.md with final milestone
