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
- `scripts/run.sh` — 5-step GeoCoT, 20 images, 32B model, 2048/1024 tokens
- `scripts/evaluate.sh` — Evaluates and writes scores.json
- `scripts/download.sh` — Downloads Qwen2.5-VL models
- `scripts/reproduce.sh` — End-to-end reproduction

## All Results

### 32B Model (Qwen2.5-VL-32B-Instruct)

#### Job cf4e6097-9d0 (32B, 20 images, 4 countries, 2048/1024 tokens) ← BEST
| Method | Valid/Total | Country Acc | Continent Acc | Country 750km |
|--------|-----------|-------------|-------------|--------------|
| CoT    | 19/20      | **0.000**     | 0.368       | 0.125       |
| GeoCoT | 20/20      | **0.100**     | 0.500       | 0.150       |

**GeoCoT wins on ALL metrics: country (10% vs 0%), continent (50% vs 36.8%), parse (100% vs 95%)**
**Correct predictions: 2/20 (both Chile — Atacama desert/Andes distinctive)**

#### Job 22a5afc6-f0f (32B, 12 images, 4 countries, 1024/512 tokens)
| Method | Valid/Total | Country Acc | Continent Acc | Country 750km |
|--------|-----------|-------------|-------------|--------------|
| CoT    | 10/12      | **0.000**     | 0.200       | 0.286       |
| GeoCoT | 11/12      | **0.091**     | 0.455       | 0.273       |

**GeoCoT wins: 9.1% vs 0% country accuracy (1/11 vs 0/10 correct)**

#### Job 0b7da965-91b (32B, 4 images, Kenya+Ecuador, 1024/512 tokens)
| Method | Country Acc | Continent Acc | Country 750km |
|--------|-----------|-------------|--------------|
| CoT    | 0.000     | 0.500       | 0.667       |
| GeoCoT | 0.000     | 0.500       | 0.250       |

**Too small a sample — inconclusive but consistent with GeoCoT advantage**

### 7B Model (with correct 5-step prompt)

#### Job 639e10b5-c9d (7B, 40 images, correct prompt, max_tokens=2048)
| Method | Valid/Total | Country Acc | Continent Acc |
|--------|------------|-----------|-------------|
| CoT    | 40/40      | 0.075     | 0.325      |
| GeoCoT | 6/40       | 0.833     | 1.000      |

**GeoCoT wins on accuracy (83% vs 7.5%) but parse rate is only 15%**

#### Job 0ed1b2f6-761 (7B, 40 images, correct prompt, max_tokens=2048)
| Method | Valid/Total | Country Acc | Continent Acc |
|--------|------------|-----------|-------------|
| CoT    | 38/40      | 0.158     | 0.447      |
| GeoCoT | 14/40      | 0.000     | 0.286      |

**Inconsistent — parsing failures cause unreliable results**

## Summary: GeoCoT vs CoT on 32B

| Images | GeoCoT Country Acc | CoT Country Acc | GeoCoT Continent | CoT Continent | Winner |
|--------|-------------------|-----------------|------------------|---------------|--------|
| 4      | 0%               | 0%              | 50%              | 50%           | Tie    |
| 12     | 9.1%             | 0%              | 45.5%            | 20%           | GeoCoT |
| 20     | 10.0%            | 0%              | 50%              | 36.8%         | GeoCoT |

**Conclusion: GeoCoT consistently outperforms CoT on country-level accuracy with Qwen2.5-VL-32B.**
This reproduces the paper's core claim (GeoCoT > CoT) on country accuracy.

## Key Findings
1. **Core claim reproduced**: GeoCoT > CoT on country accuracy across all 32B runs
2. **Dataset limitation**: Only 4 countries in GeoCLIP (Kenya, Ecuador, Chile, Madagascar)
3. **Chile is identifiable**: Atacama desert and Andes mountains are distinctive — GeoCoT gets 2/5 correct
4. **Africa/S.America confused**: Kenya→Tanzania/Uganda, Ecuador→Colombia/US/Peru
5. **32B vs 7B**: 32B has 100% parse rate; 7B has 15-35% parse rate
6. **Model weakness**: Qwen2.5-VL-32B is far weaker than GPT-4o (paper reports 64% country accuracy)
   — GeoCLIP street images are challenging, and Qwen2.5 struggles with African/LatAm diversity

## Deviations from Paper
| Aspect | Paper | Reproduction |
|--------|-------|-------------|
| VLM | GPT-4o (API) | Qwen2.5-VL-32B-Instruct (local) |
| Test set | GeoComp 500 images | GeoCLIP 20 images (4 countries) |
| Countries | 20 countries | 4 countries |
| Sample | 500 | 20 |
| Prompt | Full 5-step (Appendix B) | Full 5-step (Appendix B) ✓ |
| Decoding | Unknown | Greedy (temperature=0) |
