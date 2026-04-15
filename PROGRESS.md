# Progress

## What Works

### Briefing
- `briefing/problem.md` — Describes the image geolocation problem (method-agnostic)
- `briefing/evaluation.md` — Describes evaluation metrics (method-agnostic)
- `briefing/method.md` — Describes GeoCoT 5-step chain-of-thought prompting
- `briefing/overview.md` — Paper summary

### Method Implementation
- `method/prompt_template.py` — GeoCoT 5-step prompting and standard CoT prompts
- `method/vlm_client.py` — VLM client supporting GPT-4o (API), Qwen2.5-VL (local), LLaVA
- `method/run_geocot.py` — Main script for running GeoCoT on datasets
- `method/__init__.py` — GeoCoT module

### Data
- `data/loader.py` — Dataset loader
- `data/geoclip/geoclip.csv` — 999 images from Kenya, Ecuador, Chile, Madagascar
- `data/geoclip/*.png` — 999 image files
- `data/reverse_geocode.py` — Reverse geocoding utility

### Evaluation
- `eval/metrics.py` — Full geolocation metrics (classification + distance)
- `scripts/evaluate.sh` — End-to-end evaluation with scores.json output

### Scoring
- `scoring/reference.json` — Paper's reported numbers (Tables 2, 3, 4, 8)
- `scoring/scores.json` — Output file for reproduced numbers

### Environment
- `environment/container.def` — Ubuntu 22.04 base
- `environment/setup.sh` — Installs PyTorch, transformers, qwen-vl-utils

### Scripts
- `scripts/run.sh` — **FIXED**: Full 5-step GeoCoT prompt, 12 images, CoT first
- `scripts/evaluate.sh` — **FIXED**: Clean separation of reference vs reproduced
- `scripts/download.sh` — Downloads Qwen2.5-VL models
- `scripts/reproduce.sh` — End-to-end reproduction

## Results (Previous — with WRONG 2-question prompt)

### 7B Model (Qwen2.5-VL-7B-Instruct) — 40 images
| Method | Country Acc | Continent Acc | City 25km |
|--------|------------|---------------|-----------|
| CoT    | 0.075 (3/40) | 0.125 (5/40) | 0.25      |
| GeoCoT | 0.000 (0/40) | 0.125 (5/40) | 0.00      |

### 32B Model (Qwen2.5-VL-32B-Instruct) — 10 images (cancelled)
| Method | Country Acc |
|--------|------------|
| GeoCoT | 0.000 (0/10) |

**Note**: These results used the WRONG 2-question prompt. New job submitted with correct 5-step prompt.

## Current Job
- **Model**: Qwen2.5-VL-32B-Instruct
- **Sample**: 12 images (3 per country)
- **Prompt**: Full 5-step GeoCoT from Appendix B
- **Order**: CoT first, then GeoCoT (to ensure baseline)
- **Decoding**: Greedy (temperature=0)
- **max_new_tokens**: 256 (CoT), 600 (GeoCoT)

## Deviations from Paper

| Aspect | Paper | Reproduction |
|--------|-------|-------------|
| VLM | GPT-4o (closed API) | Qwen2.5-VL-32B-Instruct (local) |
| Test set | GeoComp 500 images (20 countries) | GeoCLIP 999 images (4 countries) |
| Countries | 20 countries, 6 continents | 4 countries, 2 continents |
| Sample size | 500 | 12 |
| City labels | Ground truth per image | Reverse geocoded from lat/lon |

GeoCoT method faithfully reproduced: same 5-step structured prompting with same questions and format from Appendix B.

## Remaining
1. **Await GPU job results** — CoT + GeoCoT with correct prompt
2. **Evaluate** — Verify GeoCoT > CoT (core claim)
3. **Update scores.json** — With actual results
4. **Consider additional experiments** if budget allows
