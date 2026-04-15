# Reproduction Milestones

**Current: none**

## Progress Log

### [2026-04-15] - 32B results + scale to 12 images
- **32B with correct prompt (job 0b7da965-91b)**:
  - CoT: 4/4 parsed, 0/4 correct = **0% country accuracy**, 50% continent
  - GeoCoT: 4/4 parsed, 0/4 correct = **0% country accuracy**, 50% continent
  - **Parsing: 100% (was 15-35% on 7B) — token increase fixed parsing!**
  - Both methods equally wrong on 4 images — too small a sample
- **New job**: 12 images (4 countries × 3), 32B model, 1024/512 tokens — ~13 min

### [2026-04-14] - Previous agent's work
- Fixed prompt to use full 5-step from Appendix B
- Multiple failed/timeout GPU jobs
- All runs used wrong prompts before fix

## Key Findings

### Model Capability (with correct 5-step prompt)
- 7B model: GeoCoT achieves 83% country accuracy on parsed predictions (vs 7.5% CoT). Parse rate 15-35%.
- 32B model: 100% parse rate (fixed), but 0% country accuracy on 4 images. Needs larger sample.
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
