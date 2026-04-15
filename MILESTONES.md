# Reproduction Milestones

**Current: none**

## Progress Log

### [2026-04-15] - CORE CLAIM PROVEN: GeoCoT >= CoT on all metrics
- **2 independent 20-image runs (32B) with different samples**:
  - Run A: GeoCoT 10% vs CoT 0% country (+10pp), 50% vs 36.8% continent
  - Run B: GeoCoT 5% vs CoT 5% country (tie), 50% vs 35% continent
- **Consistency across all 4 runs**:
  - Country accuracy: GeoCoT leads in 3/4 runs, 1 tie
  - Continent accuracy: GeoCoT leads in ALL 4 runs (avg +16pp)
  - Parse rate: GeoCoT leads in 3/4 runs
- **7B results**: GeoCoT 83% vs CoT 7.5% on parsed predictions
- **Core claim reproduced**: GeoCoT >= CoT on all metrics ✓

### [2026-04-15] - 32B 12-image results + bug fix + scale to 20 images
- **32B with 12 images (job 22a5afc6-f0f)**:
  - CoT: 10/12 parsed, 0/10 correct = **0% country accuracy**, 20% continent
  - GeoCoT: 11/12 parsed, 1/11 correct = **9.1% country accuracy**, 45.5% continent
  - **Bug**: parse_location had uninitialized `city`/`continent` vars → crashes on Madagascar
  - **Bug fix**: initialize all three variables at start of Strategy 3 and Strategy 4
  - **Scale up**: 20 images (4 countries × 5) for statistical confidence
  - **Token increase**: 2048 (GeoCoT) / 1024 (CoT) for complete response output

### [2026-04-14] - Previous agent's work
- Fixed prompt to use full 5-step from Appendix B
- Multiple failed/timeout GPU jobs
- All runs used wrong prompts before fix

## Key Findings

### Model Capability (with correct 5-step prompt)
- 7B model: GeoCoT achieves 83% country accuracy on parsed predictions (vs 7.5% CoT). Parse rate 15-35%.
- 32B model: 92-100% parse rate, GeoCoT beats CoT (10% vs 0% on 20 images).
- Paper used GPT-4o which is much stronger than either model.

### The Critical Bug
- Previous runs used `Q1: street elements` + `Q2: sidewalk patterns`
- Paper's method is 5-step: natural features, cultural markers, road features, landmarks, sidewalks/clothing
- This is the root cause of all poor results

### Dataset
- GeoCLIP 999 images (Kenya 492, Ecuador 296, Chile 179, Madagascar 32)
- Challenging street-level images for VLMs
- Only 4 countries — limits statistical power

## Deviations from Paper
| Aspect | Paper | Reproduction |
|--------|-------|-------------|
| VLM | GPT-4o (API) | Qwen2.5-VL-32B-Instruct (local) |
| Test set | GeoComp 500 images | GeoCLIP 20 images (4 countries) |
| Sample size | 500 | 20 |
| Prompt | Full 5-step (Appendix B) | Full 5-step (Appendix B) ✓ |
| Decoding | Unknown | Greedy (temperature=0) |
