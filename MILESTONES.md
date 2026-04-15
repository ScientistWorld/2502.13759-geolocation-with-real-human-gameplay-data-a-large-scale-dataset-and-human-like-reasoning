# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-04-15 HH:MM] - none
- Just started reproduction work (continuing from previous agent)

### [2026-04-15 HH:MM] - method_runs
- Paper's 5-step GeoCoT algorithm implemented and running
- Correct prompt from paper's Appendix B implemented

### [2026-04-15 HH:MM] - core_claim
- 4 independent 32B runs confirm GeoCoT >= CoT on all metrics
- Environment fully packaged and validated

### [2026-04-15 HH:MM] - core_claim_plus (in progress)
- Fixed evaluate.sh routing bug (ablation files misrouted to qwen_geocot instead of geocot_stepN)
- Ablation steps 1, 2, 3 confirmed with correct routing:
  - Step 1: 5.3% country, 52.6% continent (vs CoT: 0%, 35%)
  - Step 1-2: 15.0% country, 60.0% continent
  - Step 1-2-3: 10.0% country, 55.0% continent
- Submitted job for remaining ablation steps (step4, full)