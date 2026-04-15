# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-04-15 11:00] - core_claim
- Implemented the paper's actual GeoCoT computation as the five-step Appendix B visual reasoning prompt applied to a VLM.
- Preserved the CoT baseline and GeoCoT step ablations so the prompt decomposition can be evaluated directly.
- Reworked evaluation into `eval/evaluate_results.py`, independent of `method/`, and regenerated `scoring/scores.json` from actual prediction artifacts.
- Expanded `scoring/reference.json` to include paper-reported classification, distance, efficiency, ablation, and Im2GPS generalization targets.
- Current scaled result: Qwen GeoCoT improves continent accuracy over Qwen CoT on the local 20-image sample (0.500 vs 0.350), while country accuracy ties and distance metrics remain weak.

### [2026-04-15 11:05] - core_claim
- Audited the continuation workspace and confirmed the implementation is the actual GeoCoT prompting method, not a surrogate.
- Fixed portability issues by removing committed `shared/` path dependencies from method and script code.
- Strengthened `method/run_geocot.py` parsing so free-form GeoCoT paragraphs can still be scored as country/continent predictions.
- Restored `scripts/reproduce.sh` to call the resumable inference script (`scripts/run_ablation.sh`) rather than the validation-only job script.
- Local validation passes after the fixes.

### [2026-04-15 11:10] - core_claim_plus
- CPU validation job `e762ae95-bec` completed successfully in the managed container.
- The job regenerated `scoring/scores.json` from the real prediction artifacts and `python validate.py --compare` passed.
- Container validation confirmed import separation and no committed `shared/` path dependencies.
- Restored `core_claim_plus` because the package now includes the core CoT-vs-GeoCoT comparison plus multiple GeoCoT step ablation settings. The result remains scaled: GeoCoT improves continent-level accuracy over CoT, while country/distance metrics are weak on the 20-image substitute sample.

## Stop Justification
- Completed at milestone `core_claim_plus`. Higher milestones would require additional benchmark coverage such as Im2GPS/Im2GPS3K or a larger GeoComp-style run, which is not currently available in the packaged artifacts.
