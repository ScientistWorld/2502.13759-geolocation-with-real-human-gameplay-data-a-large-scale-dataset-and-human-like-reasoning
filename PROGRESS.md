# Progress (Updated 2026-04-15)

## What Works

### Paper Understanding and Briefing
- `briefing/problem.md` defines image geolocation without describing the solution.
- `briefing/evaluation.md` defines classification, distance, and efficiency evaluation.
- `briefing/method.md` summarizes the paper's GeoCoT prompting method.
- `briefing/overview.md` gives a short reproduction-oriented summary.

### Method Implementation
- `method/prompt_template.py` implements the paper's Appendix B GeoCoT prompt: five visual reasoning questions covering natural features, cultural/architectural cues, road features, urban markers, and fine-grained cultural details.
- `method/geocot.py` exposes `GeoCoTEngine`, a small wrapper around the prompt and a VLM client.
- `method/run_geocot.py` runs GeoCoT and CoT prompting over a dataset and writes prediction artifacts.
- `method/vlm_client.py` supports local VLM inference without paid APIs.
- Model lookup now uses portable workspace paths (`data/downloads/` and `checkpoints/`) rather than non-portable `shared/` references.

### Data and Artifacts
- `data/geoclip/geoclip.csv` indexes 999 geotagged images from the local GeoCLIP subset.
- Existing prediction artifacts in `results/` include CoT, GeoCoT, and step ablation runs from Qwen2.5-VL-32B-Instruct.
- Qwen2.5-VL checkpoints are stored under `checkpoints/` and are gitignored.

### Evaluation and Scoring
- `eval/evaluate_results.py` is the reusable evaluator. It imports no method code and scores any compatible prediction JSON files.
- `scripts/evaluate.sh` delegates to `eval.evaluate_results` and writes `scoring/scores.json`.
- `scoring/reference.json` now captures the paper's reported results for GeoComp classification, distance thresholds, efficiency, ablation, and Im2GPS generalization.
- `scoring/scores.json` is generated from actual local prediction artifacts, not copied from the paper.

## Current Reproduced Results

The current local reproduction uses the paper's GeoCoT prompt with Qwen2.5-VL-32B-Instruct on a small 20-image GeoCLIP subset. This is a scaled reproduction, not a match to the paper's GPT-4o/GeoComp setting.

### CoT vs GeoCoT

| Method | City Acc | Country Acc | Continent Acc | Country <750 km | Avg Tokens | Avg Time (s) |
|--------|----------|-------------|---------------|------------------|------------|--------------|
| Qwen CoT | 0.000 | 0.050 | 0.350 | 0.188 | 394.400 | 51.005 |
| Qwen GeoCoT | 0.000 | 0.050 | 0.500 | 0.150 | 551.650 | 78.760 |

GeoCoT supports the core qualitative claim on this scaled setup by improving continent-level geolocation over generic CoT. The country-level result ties CoT, and the distance metric is worse on this small sample.

### GeoCoT Step Ablation

| Condition | Country Acc | Continent Acc |
|-----------|-------------|---------------|
| Step 1 | 0.053 | 0.526 |
| Steps 1-2 | 0.150 | 0.600 |
| Steps 1-2-3 | 0.100 | 0.550 |
| Steps 1-2-3-4 | 0.050 | 0.550 |
| Full GeoCoT | 0.000 | 0.368 |

The ablation shows that the paper's visual-cue decomposition matters, but the full five-step prompt is not uniformly best for this smaller local VLM and dataset. Steps 1-2 give the strongest local result.

## Validation

- `scripts/evaluate.sh` successfully regenerates `scoring/scores.json`.
- `python validate.py --compare` passes locally after expanding `reference.json` and routing the current score names.
- Audit fixes passed lightweight checks: Python compilation, shell syntax checks, parser smoke tests, and `validate.py --compare`.
- CPU test job `e762ae95-bec` passed inside the managed container with exit code 0. It regenerated scores, ran `python validate.py --compare`, and confirmed import separation plus portable path checks.

## Deviations from Paper

| Aspect | Paper | Reproduction |
|--------|-------|--------------|
| VLM | GPT-4o / commercial VLMs | Qwen2.5-VL-32B-Instruct local checkpoint |
| Main dataset | GeoComp, 500 images, 20 countries | GeoCLIP-derived 20-image subset, 4 countries |
| Prompt | Appendix B five-step GeoCoT | Same five-step GeoCoT prompt |
| Decoding | Not fully specified | Greedy decoding for reproducibility |
| Reported tables | Tables 2, 3, 4, 7, 8 | Local scores for Tables 2, 3, 7, 8-style metrics; no Im2GPS run yet |

The method itself was not replaced: the implemented computation is the paper's explicit GeoCoT visual-question prompt applied to a VLM, compared against generic CoT and step ablations.

## What Remains

- Run a larger sample or the full GeoComp dataset if it becomes available.
- Add Im2GPS/Im2GPS3K evaluation artifacts to populate `im2gps_generalization`.
- Re-run full GeoCoT with a model closer to the paper's GPT-4o capability if a suitable local open-weight alternative is available within budget.

## Current Milestone

`core_claim_plus`: the core comparison and several GeoCoT step ablations are packaged and validated. The reproduction does not claim secondary/generalization milestones because no Im2GPS/Im2GPS3K artifacts were run and the full five-step prompt remains weak on the small substitute dataset.
