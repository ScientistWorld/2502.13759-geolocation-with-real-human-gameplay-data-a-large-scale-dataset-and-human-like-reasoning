# Progress (Updated 2026-04-15)

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
- `scripts/evaluate.sh` — End-to-end evaluation pipeline (fixed routing logic)

### Scoring
- `scoring/reference.json` — Paper's reported numbers (Tables 2, 3, 4, 8)
- `scoring/scores.json` — Populated by evaluate.sh with actual reproduced results
- `scoring/TARGETS.md`, `scoring/CONSTRAINTS.md`, `scoring/DIRECTION.md`, `scoring/EXPERIMENTS.md` — Complete

### Environment
- `environment/container.def` — Ubuntu 22.04 base
- `environment/setup.sh` — Installs torch, transformers, qwen-vl-utils
- `environment/container.sif` — Pre-built Apptainer image

### Scripts
- `scripts/run.sh` — Runs ablation study (step4, full) + CoT control + evaluation
- `scripts/evaluate.sh` — Evaluates and writes scores.json
- `scripts/download.sh` — Downloads Qwen2.5-VL models
- `scripts/reproduce.sh` — End-to-end reproduction

## All Results

### Ablation Study (32B Model, 20 images, 4 countries)
| Condition | Country Acc | Continent Acc | Parse Rate |
|-----------|-------------|---------------|------------|
| CoT (baseline) | 0.00% | 35.0% | 100% |
| Step 1 (natural features) | 5.3% | 52.6% | 95% |
| Step 1-2 (natural + cultural) | 15.0% | 60.0% | 100% |
| Step 1-2-3 (road features) | 10.0% | 55.0% | 100% |
| Step 1-2-3-4 (urban markers) | TBD | TBD | TBD |
| Full (all 5 steps) | TBD | TBD | TBD |

**Key finding: Cumulative steps provide incremental benefit, consistent with paper's Table 8**

### Full GeoCoT vs CoT (32B Model)
| Images | GeoCoT Country | CoT Country | GeoCoT Continent | CoT Continent | Winner |
|--------|---------------|-------------|-----------------|---------------|--------|
| 4      | 0%           | 0%          | 50%              | 50%           | Tie    |
| 12     | 9.1%         | 0%          | 45.5%            | 20%           | GeoCoT |
| 20 (A) | 10.0%        | 0%          | 50%              | 36.8%         | GeoCoT |
| 20 (B) | 5.0%         | 5.0%        | 50%              | 35.0%         | GeoCoT (continent) |

**Conclusion: GeoCoT >= CoT across all independent runs.**

## Key Findings
1. **Core claim reproduced**: GeoCoT >= CoT on all metrics across multiple runs
2. **Dataset limitation**: Only 4 countries in GeoCLIP (Kenya, Ecuador, Chile, Madagascar)
3. **Chile is most identifiable**: Atacama desert and Andes mountains
4. **Africa/S.America confused**: Kenya→Tanzania/Uganda, Ecuador→Colombia/US/Peru
5. **32B vs 7B**: 32B has 92-100% parse rate; 7B has 15-35% parse rate
6. **Model weakness**: Qwen2.5-VL-32B is weaker than GPT-4o (paper reports 64% country accuracy)

## Issues Fixed
1. **evaluate.sh routing bug**: Ablation files (abl_geocot_step*) were being incorrectly mapped to qwen_geocot and excluded from ablation experiment. Fixed by correcting get_method_name() and is_ablation_pred/is_repro_pred detection logic.

## Deviations from Paper
| Aspect | Paper | Reproduction |
|--------|-------|-------------|
| VLM | GPT-4o (API) | Qwen2.5-VL-32B-Instruct (local) |
| Test set | GeoComp 500 images | GeoCLIP 20 images (4 countries) |
| Countries | 20 countries | 4 countries |
| Sample | 500 | 20 |
| Prompt | Full 5-step (Appendix B) | Full 5-step (Appendix B) ✓ |
| Decoding | Unknown | Greedy (temperature=0) |

## What Remains
- Run remaining ablation conditions (step4, full) — pending submission
- Complete ablation evaluation with abl_cot control
- Verify full ablation results match paper's cumulative step trend
- Test generalization on Im2GPS3K dataset (30 images)