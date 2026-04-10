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

### Data
- `data/loader.py` — Dataset loader for multiple formats
- `data/geoclip/geoclip.csv` — 999 images from Kenya, Ecuador, Chile, Madagascar with lat/lon, country, continent labels
- `data/geoclip/*.png` — Image files

### Evaluation
- `eval/metrics.py` — Geolocation metrics: classification (accuracy/recall/F1), distance (1km/25km/750km), haversine distance

### Scoring
- `scoring/reference.json` — Paper's reported numbers + our geoclip reproduction experiment
- `scoring/scores.json` — Output file for reproduced numbers

### Environment
- `environment/container.def` — Uses Azure Linux 3.0 from MCR with Python packages
- `environment/setup.sh` — Installs PyTorch, transformers, pandas, etc.

### Scripts
- `scripts/run.sh` — Runs GeoCoT and CoT experiments with LLaVA-1.5-7B
- `scripts/evaluate.sh` — Evaluates predictions
- `scripts/download.sh` — Downloads models and datasets

## Results

No results yet — job pending. The reproduction uses:
- **VLM**: LLaVA-1.5-7B (already downloaded locally)
- **Dataset**: GeoCLIP 999 images (Kenya, Ecuador, Chile, Madagascar)
- **Experiment**: GeoCoT vs standard CoT on same 50 images

Expected: GeoCoT should outperform standard CoT due to structured multi-step geographical reasoning.

## Remaining

1. **Container build** — Need to verify container builds with MCR Azure Linux image
2. **Run experiments** — Execute GeoCoT and CoT on GPU node
3. **Evaluate results** — Generate scores.json and compare against paper
4. **Validate workspace** — Run `python validate.py`

## Issues

- **Container build failure** — Previous Alpine Docker build failed due to rate limiting. Switched to MCR Azure Linux.
- **No GPT-4o API** — Using LLaVA-1.5-7B as local VLM
- **Limited dataset** — GeoCLIP has only 4 countries (Kenya, Ecuador, Chile, Madagascar) across 2 continents. Not as diverse as paper's 500-image GeoComp test set or Im2GPS3K.
- **Im2GPS3K unavailable** — Stanford URL returns 404, tarball is corrupted

## Deviations from Paper

| Aspect | Paper | Reproduction |
|--------|-------|-------------|
| VLM | GPT-4o (closed API) | LLaVA-1.5-7B (local, 7B params) |
| Test set | GeoComp 500 images | GeoCLIP 999 images (substitute) |
| Dataset | GeoComp, Im2GPS3K | GeoCLIP (limited geography) |
| Metrics | Full Table 2/3/4/8 | Subset (geoclip has no city labels) |

GeoCoT method itself is faithfully reproduced: same 5-step structured prompting with the same questions and format. Only the VLM and dataset differ.
