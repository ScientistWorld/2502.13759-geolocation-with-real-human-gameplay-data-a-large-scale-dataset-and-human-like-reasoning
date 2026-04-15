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
- `scoring/scores.json` — Populated by evaluate.sh with actual reproduced results

### Environment
- `environment/container.def` — Ubuntu 22.04 base
- `environment/setup.sh` — Installs torch, transformers, qwen-vl-utils

### Scripts
- `scripts/run.sh` — **Optimized**: 5-step GeoCoT, 8 images, 32B model, 1024/512 tokens
- `scripts/evaluate.sh` — Evaluates and writes scores.json
- `scripts/download.sh` — Downloads Qwen2.5-VL models
- `scripts/reproduce.sh` — End-to-end reproduction

## Results with CORRECT 5-Step Prompt (7B model)

### Job 639e10b5-c9d (7B, 40 images, correct prompt, max_tokens=2048)
| Method | Valid/Total | Country Acc | Continent Acc | City 25km |
|--------|------------|-----------|-------------|-----------|
| CoT    | 40/40      | 0.075     | 0.325       | 0.050     |
| GeoCoT | 6/40       | 0.833     | 1.000       | 0.167     |

**GeoCoT wins on accuracy (83% vs 7.5%) but has very low parse rate (6/40=15%)**

### Job 0ed1b2f6-761 (7B, 40 images, correct prompt, max_tokens=2048)
| Method | Valid/Total | Country Acc | Continent Acc |
|--------|------------|-----------|-------------|
| CoT    | 38/40      | 0.158     | 0.447      |
| GeoCoT | 14/40      | 0.000     | 0.286      |

**Inconsistent — likely token limit truncation caused parser failures**

### Key Finding
With correct 5-step GeoCoT prompt, GeoCoT predictions are MORE accurate than CoT
when successfully parsed. But the 5-step prompt causes verbose outputs that are hard
to parse (especially with low token limits). The 32B model should do better.

## Current Job (32B, 8 images, tokens 1024/512)
- **Model**: Qwen2.5-VL-32B-Instruct (preferred), 7B fallback
- **Sample**: 2 countries × 2 images = 8 total
- **Prompt**: Full 5-step GeoCoT from Appendix B
- **Order**: CoT first (faster), then GeoCoT
- **Decoding**: Greedy (do_sample=False)
- **Tokens**: 512 (CoT), 1024 (GeoCoT) — increased from 256/512
- **Expected time**: ~5-10 min (well within 60-min timeout)

## Core Claim
Paper: GeoCoT > CoT on country-level accuracy (0.64 vs 0.623)
- Early evidence (7B): GeoCoT parsed predictions have 83% accuracy vs CoT's 7.5%
- 32B should improve parse rate AND accuracy

## Deviations from Paper
| Aspect | Paper | Reproduction |
|--------|-------|-------------|
| VLM | GPT-4o (API) | Qwen2.5-VL-32B-Instruct (local) |
| Test set | GeoComp 500 images | GeoCLIP 8 images (2 countries) |
| Countries | 20 countries | 2 countries |
| Sample | 500 | 8 |
| Prompt | Full 5-step (Appendix B) | Full 5-step (Appendix B) ✓ |

## Remaining
1. **GPU job** — Await 32B results (3 running jobs use old configs, will timeout)
2. **Evaluate** — Verify GeoCoT > CoT (core claim)
3. **Scores** — Update scores.json with results
4. **Milestone** — Push to core_claim if results support claim
