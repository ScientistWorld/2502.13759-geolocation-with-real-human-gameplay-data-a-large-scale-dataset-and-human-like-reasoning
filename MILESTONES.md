# Reproduction Milestones

**Current: none**

## Progress Log

### [2026-04-15] - Correct prompt results + new job
- **7B with correct 5-step prompt (job 639e10b5-c9d)**:
  - GeoCoT: 6/40 parsed, 5/6 correct = **83.33% country accuracy**
  - CoT: 40/40 parsed, 3/40 correct = **7.5% country accuracy**
  - GeoCoT wins on parsed accuracy by 11x but parse rate is only 15%
- **7B with correct prompt + 2048 tokens (job 0ed1b2f6-761)**:
  - GeoCoT: 14/40 parsed, 0/14 correct (inconsistent)
  - CoT: 38/40 parsed, 6/38 correct = 15.79%
- **Token limit fix**: 512->1024 (GeoCoT), 256->512 (CoT)
- **3 running jobs** (2bcf428c-531, 791d97e8-d57, c8fc103e-bc0) use OLD configs — will timeout
- **New job**: 8 images, 32B model, 1024/512 tokens — should complete in ~5-10 min

### [2026-04-14] - Previous agent's work
- Fixed prompt to use full 5-step from Appendix B
- Multiple failed/timeout GPU jobs
- All runs used wrong prompts before fix

## Key Findings

### Model Capability (with correct 5-step prompt)
- 7B model: With correct prompt, GeoCoT achieves 83% country accuracy on parsed predictions (vs 7.5% CoT). But parse rate is only 15-35%.
- 32B model: Should improve both parse rate AND accuracy
- Paper used GPT-4o which is much stronger than either

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
| Test set | GeoComp 500 images | GeoCLIP 8 images (2 countries) |
| Sample size | 500 | 8 |
| Prompt | Full 5-step (Appendix B) | Full 5-step (Appendix B) ✓ |
| Decoding | Unknown | Greedy (temperature=0) |
