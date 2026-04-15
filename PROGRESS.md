# Progress

## What Works

### Briefing
- `briefing/problem.md` — Method-agnostic image geolocation problem definition
- `briefing/evaluation.md` — Method-agnostic evaluation metrics (classification + distance)
- `briefing/method.md` — GeoCoT 5-step chain-of-thought description
- `briefing/overview.md` — Paper summary

### Method Implementation
- `method/prompt_template.py` — Paper's 5-step GeoCoT prompt from Appendix B
- `method/vlm_client.py` — VLM client (GPT-4o, Qwen2.5-VL, LLaVA)
- `method/run_geocot.py` — Main GeoCoT runner
- `method/__init__.py` — GeoCoT module

### Data
- `data/geoclip/geoclip.csv` — 999 images from Kenya, Ecuador, Chile, Madagascar
- `data/geoclip/*.png` — 999 image files (73 MB)

### Evaluation
- `eval/metrics.py` — Full geolocation metrics (classification + distance)
- `scripts/evaluate.sh` — End-to-end evaluation pipeline

### Scoring
- `scoring/reference.json` — Paper's reported numbers (Tables 2, 3, 4, 8)
- `scoring/scores.json` — Reset to empty (will be populated by evaluate.sh)

### Environment
- `environment/container.def` — Ubuntu 22.04 base
- `environment/setup.sh` — Installs torch, transformers, qwen-vl-utils

### Scripts
- `scripts/run.sh` — **Optimized**: 5-step GeoCoT, 8 images, 32B model
- `scripts/evaluate.sh` — Evaluates and writes scores.json
- `scripts/download.sh` — Downloads Qwen2.5-VL models
- `scripts/reproduce.sh` — End-to-end reproduction

## Results (Previous Runs — ALL WRONG PROMPT)

### Previous Bug: 2-question prompt vs 5-step
All previous runs used `Q1: street elements + Q2: sidewalk patterns` instead of the paper's ACTUAL 5-step GeoCoT:
1. Natural features / climate zone
2. Cultural markers / language / architecture
3. Road features / license plates
4. Urban markers / street signs
5. Sidewalk patterns / clothing

### 7B Results (40 images, wrong prompt)
| Method | Country Acc | Continent Acc |
|--------|-----------|---------------|
| CoT    | 0.075     | 0.125         |
| GeoCoT | 0.000     | 0.125         |

### 32B Results (10 images, wrong prompt)
| Method | Country Acc |
|--------|-----------|
| GeoCoT | 0.000     |

**Note**: These used the WRONG prompt. New job uses correct 5-step prompt.

## Current Job
- **Model**: Qwen2.5-VL-32B-Instruct (preferred), 7B fallback
- **Sample**: 2 countries × 2 images = 8 total
- **Prompt**: Full 5-step GeoCoT from Appendix B
- **Order**: CoT first (faster), then GeoCoT
- **Decoding**: Greedy (do_sample=False)
- **Tokens**: 256 (CoT), 512 (GeoCoT)
- **Expected time**: ~30-40 min (within 60-min timeout)

## Core Claim
Paper: GeoCoT > CoT on country-level accuracy (0.64 vs 0.623)
- Reproduction: With correct prompt, does GeoCoT outperform CoT?

## Deviations from Paper
| Aspect | Paper | Reproduction |
|--------|-------|-------------|
| VLM | GPT-4o (closed API) | Qwen2.5-VL-32B-Instruct (local) |
| Test set | GeoComp 500 images | GeoCLIP 8 images (2 countries) |
| Countries | 20 countries | 2 countries |
| Sample | 500 | 8 |
| Prompt | Full 5-step (Appendix B) | Full 5-step (Appendix B) ✓ |

## Remaining
1. **GPU job** — Submit optimized run and await results
2. **Evaluate** — Verify GeoCoT > CoT (core claim)
3. **Scores** — Update scores.json with results
4. **Milestone** — Push to core_claim if results support claim
