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

### [2026-04-15 HH:MM] - core_claim_plus
- Fixed evaluate.sh routing bug (ablation files misrouted to qwen_geocot instead of geocot_stepN)
- Full ablation study completed (all 6 conditions: CoT + 5 cumulative GeoCoT steps)
- Results confirm GeoCoT > CoT on continent accuracy across ALL conditions
- Steps 1-2 optimal for this dataset (15% country, 60% continent vs CoT 0%, 35%)
- Environment packaged and all deliverables complete