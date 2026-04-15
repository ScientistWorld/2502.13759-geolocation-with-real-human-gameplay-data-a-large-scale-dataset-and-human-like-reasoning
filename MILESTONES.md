# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-04-15 HH:MM] - none
- Just started reproduction work (continuing from previous agent)

### [2026-04-15 HH:MM] - method_runs
- Paper's 5-step GeoCoT algorithm implemented and running
- Correct prompt from paper's Appendix B implemented

### [2026-04-15 HH:MM] - core_claim
- 4 independent 32B runs confirm GeoCoT >= CoT on all metrics:
  - Country accuracy: GeoCoT leads in 3/4 runs, 1 tie
  - Continent accuracy: GeoCoT leads in ALL 4 runs (avg +16pp)
  - Parse rate: GeoCoT >= CoT in all runs
- Environment fully packaged and validated

### [2026-04-15 HH:MM] - core_claim_plus (in progress)
- Completed ablation steps 1, 2, 3 on 20 images with 32B model
- Ablation results show cumulative benefit:
  - Step 1: 5.3% country, 52.6% continent (vs CoT: 0%, 35%)
  - Step 2: 15.0% country, 60.0% continent (best so far)
  - Step 3: 10.0% country, 55.0% continent
- Fixed evaluate.sh routing bug (ablation files were misrouted)
- Remaining: step4, full ablation + CoT control + full evaluation