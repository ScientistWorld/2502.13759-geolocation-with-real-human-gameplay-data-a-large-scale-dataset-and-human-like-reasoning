# Verification Report

## Initial Assessment
The workspace already claimed `core_claim_plus` and had a working GeoCoT reproduction package: method code, reusable evaluation scripts, scored prediction artifacts, train/test evaluation slices, and a prior managed-container validation job. The claimed milestone is justified because the implementation runs the paper's prompting algorithm directly and includes both the core CoT comparison and multiple step-ablation settings.

## Issues Found
- Minor: method-agnostic documentation still contained template text and solution-specific naming that would leak the paper's method to from-scratch scientist agents.
- Minor: `scripts/evaluate.sh` emitted an unreproduced reference-only experiment with an empty `results` block, which made the scored experiment set less clear.
- Minor: evaluator routing was tied too closely to the reproduced method filenames for the main experiments, making it less reusable for future methods that produce compatible prediction files.
- Minor: the ablation scorer temporarily included a control output that is not part of the paper-reported ablation reference table, causing validation to fail until the route was tightened.
- Cosmetic: an untracked root-level core dump was present; tracked root-level container/system artifacts were left in place as existing system files.

## Fixes Applied
- Rewrote `briefing/problem.md`, `briefing/evaluation.md`, and `scoring/EXPERIMENTS.md` to describe the problem and experiments without exposing the paper's algorithm.
- Updated `scoring/TARGETS.md`, `scoring/CONSTRAINTS.md`, and `scoring/DIRECTION.md` to give future scientists precise, high-level guidance while avoiding internal method details.
- Renamed the reproduced ablation experiment to `ablation_explanation_scaffolds` and synchronized the evaluator, reference file, scores files, and target documentation.
- Changed the evaluator to omit experiments with no reproduced results and to score any non-ablation compatible prediction file in the main experiments.
- Regenerated `scoring/scores.json`, `scoring/scores_train.json`, and `scoring/scores_test.json` through the evaluation scripts.
- Moved the untracked root-level core dump into `tmp/root_artifacts/`.

## Final Judgment
The workspace is ready to serve as a research gym with clear limitations. Improve-mode agents can use the full GeoCoT implementation, prediction artifacts, and ablation setup. From-scratch agents can use the method-agnostic problem, data, and evaluation files without seeing the implementation details. The main limitation is scale: the reproduced run uses a small 20-image accessible sample, so the benchmark is useful for iteration but not a full-strength reproduction of the paper's full GeoComp and Im2GPS evaluation.

- Milestone: core_claim_plus
- Ready for gym use: partial
- Confidence: medium
- Key limitation: scaled 20-image substitute sample with weak city/country/distance reproduction and no reproduced Im2GPS generalization run.
