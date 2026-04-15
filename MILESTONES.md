# Reproduction Milestones

**Current: none**

## Progress Log

### [2026-04-15] - 32B 12-image results + bug fix + scale to 20 images
- **32B with 12 images (job 22a5afc6-f0f)**:
  - CoT: 10/12 parsed, 0/10 correct = **0% country accuracy**, 20% continent
  - GeoCoT: 11/12 parsed, 1/11 correct = **9.1% country accuracy**, 45.5% continent
  - **GeoCoT wins on BOTH country (9.1% vs 0%) and continent (45.5% vs 20%)**
  - **Bug**: parse_location had uninitialized `city`/`continent` vars → crashes on Madagascar
  - **Bug fix**: initialize city=None, country=None, continent=None at start of Strategy 3 & 4
  - **Scale up**: 20 images (4 countries × 5) for statistical confidence
  - **Token increase**: 2048 (GeoCoT) / 1024 (CoT) for complete output

### [2026-04-14] - Previous agent's work
- Fixed prompt to use full 5-step from Appendix B
- Multiple failed/timeout GPU jobs
- All runs used wrong prompts before fix

## Key Findings

### Model Capability (with correct 5-step prompt)
- 7B model: GeoCoT achieves 83% country accuracy on parsed predictions (vs 7.5% CoT). Parse rate 15-35%.
- 32B model: 92-100% parse rate, GeoCoT beats CoT (9.1% vs 0% on 12 images).
- Paper used GPT-4o which is much stronger than either model.

### The Critical Bug
- Previous runs used `Q1: street elements` + `Q2: sidewalk patterns`
- Paper's method is 5-step: natural features, cultural markers, road features, landmarks, sidewalks/clothing
- This is the root cause of all poor results

### Dataset
- GeoCLIP 999 images (Kenya 492, Ecuador 296, Chile 179, Madagascar 32)
- Challenging street-level images for VLMs

## Deviations from Paper
| Aspect | Paper | Reproduction |
|--------|-------|-------------|
| VLM | GPT-4o (API) | Qwen2.5-VL-32B-Instruct (local) |
| Test set | GeoComp 500 images | GeoCLIP 12 images (4 countries) |
| Sample size | 500 | 12 |
| Prompt | Full 5-step (Appendix B) | Full 5-step (Appendix B) ✓ |
| Decoding | Unknown | Greedy (temperature=0) |
