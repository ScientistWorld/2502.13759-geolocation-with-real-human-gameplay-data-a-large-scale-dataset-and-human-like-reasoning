# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-15 00:00] - Fixing GeoCoT implementation
- **Root cause identified**: Previous runs used WRONG prompt (2 questions from GitHub code vs paper's 5-step Appendix B prompt)
- **Previous results with wrong prompt**:
  - 7B GeoCoT (2Q prompt): 0/40 country accuracy
  - 7B CoT: 3/40 country accuracy (0.075)
  - 32B GeoCoT (2Q prompt): 0/10 country accuracy (cancelled)
- **Fix**: Implemented paper's actual 5-step prompt from Appendix B:
  1. Natural features / climate zone
  2. Cultural markers / language / architecture
  3. Road features / license plates
  4. Urban markers / street signs
  5. Sidewalk patterns / clothing
- **Additional improvements**:
  - Reduced sample to 12 images (3 per country) for time budget
  - Run CoT first to ensure baseline results even if timeout
  - Greedy decoding for speed + reproducibility
  - Better parser with multiple extraction strategies
- **Job submitted**: Qwen2.5-VL-32B-Instruct on 12 images

### [2026-04-14] - Previous agent's work
- Multiple GPU jobs with 7B model (poor results)
- One 32B job cancelled after 10 images (timeout)
- Core issue: wrong prompt + weak model

## Key Findings

### Model Capability
- 7B model: Too weak for geolocation, often misidentifies countries
- 32B model: More capable but slow (~3-5 min/image)
- Paper used GPT-4o which is much stronger than either

### Dataset
- GeoCLIP 999 images (Kenya 492, Ecuador 296, Chile 179, Madagascar 32)
- These are street-level images, challenging for VLMs
- Chile and Ecuador particularly hard (less distinctive features)

## Deviations from Paper
| Aspect | Paper | Reproduction |
|--------|-------|-------------|
| VLM | GPT-4o (API) | Qwen2.5-VL-32B-Instruct (local) |
| Dataset | GeoComp 500 images | GeoCLIP 999 images (4 countries) |
| Sample | 500 images | 12 images (3 per country) |
| Prompt | Full 5-step (Appendix B) | Full 5-step (Appendix B) ✓ |
| Decoding | Unknown | Greedy (temperature=0) |
