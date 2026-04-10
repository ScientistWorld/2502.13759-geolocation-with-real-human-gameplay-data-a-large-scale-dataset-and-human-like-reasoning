# Progress

## What Works

### Briefing
- `briefing/problem.md` — Describes the image geolocation problem
- `briefing/evaluation.md` — Describes evaluation metrics
- `briefing/method.md` — Describes GeoCoT prompting framework
- `briefing/overview.md` — Paper summary

### Method Implementation
- `method/prompt_template.py` — GeoCoT 5-step prompting and standard CoT prompts
- `method/vlm_client.py` — VLM client supporting GPT-4o (API), Qwen2.5-VL (local), and LLaVA (local)
- `method/run_geocot.py` — Main script to run GeoCoT on a dataset
- `method/__init__.py` — GeoCoT module

### Data
- `data/loader.py` — Dataset loader for multiple formats
- `data/geoclip/geoclip.csv` — 999 images from Kenya, Ecuador, Chile, Madagascar with lat/lon, country, continent labels
- `data/geoclip/*.png` — 999 image files
- `data/reverse_geocode.py` — Reverse geocoding utility for city extraction from lat/lon

### Evaluation
- `eval/metrics.py` — Geolocation metrics: classification (accuracy/recall/F1), distance (1km/25km/750km), haversine distance
- `eval/__init__.py` — Evaluation module

### Scoring
- `scoring/reference.json` — Paper's reported numbers (Tables 2, 3, 4, 8)
- `scoring/scores.json` — Output file for reproduced numbers (placeholder 0.0 values)

### Environment
- `environment/container.def` — Uses Ubuntu 22.04 base with Python packages
- `environment/setup.sh` — Installs PyTorch, transformers, qwen-vl-utils, etc.
- `pylib/` — Pre-installed Python packages (PyTorch 2.6.0+cu124, transformers 5.5.3, etc.)

### Scripts
- `scripts/run.sh` — Full VLM inference pipeline (GeoCoT + CoT on 160 images)
- `scripts/evaluate.sh` — Evaluates predictions and generates scores.json
- `scripts/download.sh` — Downloads models and datasets

## Results

GPU job pending. The reproduction uses:
- **VLM**: Qwen2.5-VL-7B-Instruct at `/home/user/checkpoints/Qwen2.5-VL-7B-Instruct`
- **Dataset**: GeoCLIP 999 images (Kenya: 492, Ecuador: 296, Chile: 179, Madagascar: 32)
- **Sample**: 160 images (40 per country, spatially balanced)
- **Experiments**: GeoCoT vs standard CoT on same images
- **Metrics**: country/continent accuracy, distance-based (1km/25km/750km)

Expected: GeoCoT should outperform standard CoT due to structured multi-step geographical reasoning.

## Deviations from Paper

| Aspect | Paper | Reproduction |
|--------|-------|-------------|
| VLM | GPT-4o (closed API) | Qwen2.5-VL-7B-Instruct (local) |
| Test set | GeoComp 500 images | GeoCLIP 999 images (substitute) |
| Countries | 20 countries, 6 continents | 4 countries, 2 continents |
| City labels | Ground truth per image | Reverse geocoded from lat/lon |

GeoCoT method faithfully reproduced: same 5-step structured prompting with same questions and format.

## Remaining

1. **GPU job** — Run GeoCoT and CoT inference on 160 images
2. **Evaluation** — Generate scores.json with real metrics
3. **Validation** — Run `python validate.py --compare`
4. **Wrap-up** — Update MILESTONES.md with final milestone
