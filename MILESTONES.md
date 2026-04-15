# Reproduction Milestones

**Current: none**

## Progress Log

### [2026-04-15] - none
- Picking up workspace from previous agent
- Previous runs failed due to: wrong prompt (2Q vs 5-step), timeout (too many images), old configs
- **Root cause**: ALL previous runs used WRONG prompt (2 questions from GitHub vs paper's 5-step Appendix B)
- Previous 32B job: processed 10/80 images before timeout (wrong prompt + too many images)
- Previous 7B jobs: loaded model then got cancelled (wrong prompt + timeout issues)
- **Fix**: New optimized run.sh with:
  - Paper's ACTUAL 5-step GeoCoT prompt (Appendix B)
  - 2 countries, 2 images each = 8 total
  - 32B model (preferred), 7B fallback
  - Greedy decoding (do_sample=False)
  - Real-time stdout flushing
  - CoT first (faster), then GeoCoT

### [2026-04-14] - Previous agent's work
- Fixed prompt to use full 5-step from Appendix B
- Multiple failed/timeout GPU jobs
- All runs used wrong prompts before fix

## Key Findings

### Model Capability
- 7B model: Too weak for geolocation (0% country accuracy with old prompt)
- 32B model: More capable but slow (~5 min/image with greedy)
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
