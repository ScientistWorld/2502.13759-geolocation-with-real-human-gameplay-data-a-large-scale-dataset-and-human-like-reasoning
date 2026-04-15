# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-04-15 11:00] - core_claim
- Implemented the paper's actual GeoCoT computation as the five-step Appendix B visual reasoning prompt applied to a VLM.
- Preserved the CoT baseline and GeoCoT step ablations so the prompt decomposition can be evaluated directly.
- Reworked evaluation into `eval/evaluate_results.py`, independent of `method/`, and regenerated `scoring/scores.json` from actual prediction artifacts.
- Expanded `scoring/reference.json` to include paper-reported classification, distance, efficiency, ablation, and Im2GPS generalization targets.
- Current scaled result: Qwen GeoCoT improves continent accuracy over Qwen CoT on the local 20-image sample (0.500 vs 0.350), while country accuracy ties and distance metrics remain weak.

### [2026-04-15 11:20] - core_claim
- Audited the continuation workspace and confirmed the implementation is the actual GeoCoT prompting method, not a surrogate.
- Fixed portability issues by removing committed `shared/` path dependencies from method and script code.
- Strengthened `method/run_geocot.py` parsing so free-form GeoCoT paragraphs can still be scored as country/continent predictions.
- Restored `scripts/reproduce.sh` to call the resumable inference script (`scripts/run_ablation.sh`) rather than the validation-only job script.
- Local validation passes after the fixes.

## Stop Justification
- Not stopped. A CPU validation job is being submitted to verify the scoring package in the managed environment.
