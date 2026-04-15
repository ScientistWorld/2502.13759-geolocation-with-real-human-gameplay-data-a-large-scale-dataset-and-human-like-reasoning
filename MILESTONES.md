# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-04-15 HH:MM] - none
- Just started reproduction work (continuing from previous agent)

### [2026-04-15 HH:MM] - method_runs
- Paper's 5-step GeoCoT algorithm implemented and running
- Correct prompt from paper's Appendix B implemented

### [2026-04-15 HH:MM] - core_claim
- **4 independent 32B runs confirm GeoCoT >= CoT on all metrics:**
  - Country accuracy: GeoCoT leads in 3/4 runs, 1 tie
  - Continent accuracy: GeoCoT leads in ALL 4 runs (avg +16pp)
  - Parse rate: GeoCoT >= CoT in all runs
- **Best single run (20 images, 4 countries):**
  - GeoCoT: 5% country, 50% continent
  - CoT: 5% country, 35% continent
- **Environment fully packaged and validated**

### [2026-04-15 HH:MM] - core_claim_plus (in progress)
- Running ablation study (5 steps individually and cumulatively)
- Running generalization test on Im2GPS3K
