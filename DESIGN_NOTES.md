# Design Notes — Train/Test Evaluation (Iteration 8, Designer Stage 2)

## Changes from First Pass

### 1. Consolidated slice evaluators to reuse top-level metric functions

**Before**: Both `eval/train/evaluate_results.py` and `eval/test/evaluate_results.py` duplicated ~80 lines of utility code (haversine, COUNTRY_COORDS, normalize_country, normalize_continent, reverse_geocode, compute_metrics, method_name, route_experiment).

**After**: Both slices now `from eval.evaluate_results import compute_metrics, method_name, route_experiment`. The top-level evaluator was already fixed with the correct denominator semantics (missing predicted coords count as failures in distance thresholds). The slices inherit those semantics without reimplementation.

This also eliminates a prior inconsistency: the first-pass slice evaluators used a slightly different normalization regex (no strip of `*_` characters) than the top-level evaluator.

### 2. Added `_slice` marker in per-method metrics

Both slices now set `metrics["_slice"] = SPLIT_NAME` in the computed output. This makes the slice origin traceable in any downstream analysis of the score files. It's excluded from the score output via `and not key.startswith("_")` when building entries.

### 3. Replaced local compute_metrics with shared version

The first-pass `compute_metrics` in the slices had a bug: it checked `total` as denominator for every classification level, rather than using the per-level ground-truth count. The shared version from `eval/evaluate_results.py` is correct.

## Design Decisions and Rationale

### Split Style: 50/50 geographic partition

The 20-image sample has 4 countries, 5 images each. We split 10/10 by geographic clustering rather than randomly, so each country is represented in both slices and neither slice is systematically harder (the images are close in difficulty within each country).

The split covers all 4 countries and all 20 images. No test image appears in train and vice versa. This is a clean IID split that tests whether improvements generalize across the same geographic distribution, not across OOD regions.

### Why not a different split style?

- **Metric triangulation**: Not applicable here — classification and distance metrics are already independent dimensions of the same task. The geocomp_classification experiment already measures country accuracy, continent accuracy, and F1 simultaneously.
- **Out-of-distribution**: Would require a 5th country not in the sample, which doesn't exist. Splitting off one country would leave only 3 countries and reduce power.
- **Difficulty stratification**: Not easily sortable; geolocation difficulty depends on the specific image content.
- **Ablated inputs**: Not applicable — geolocation predictions are on natural images, not inputs with ablated features.

The geographic partition is the most appropriate style for this paper's claim (GeoCoT improves accuracy through better visual reasoning) and this dataset (4 countries, 5 images each).

### Train/test metric coverage

All 4 experiments are evaluated on both slices:
- `geocomp_classification`: country and continent accuracy/recall/F1
- `geocomp_distance`: street_1km, city_25km, country_750km
- `geocomp_efficiency`: avg_generated_tokens, avg_inference_time_s
- `ablation_explanation_scaffolds`: country and continent accuracy/recall/F1 for ablation variants

Efficiency metrics are identical on train and test because the same prediction artifacts are scored — the efficiency of a fixed model on a given image is deterministic. This means efficiency cannot be gamed separately on train vs test. However, efficiency functions as a constraint: a scientist who wants to maximize accuracy at the cost of 10x more tokens or time would see this reflected across both slices.

### What the gap detects

A shortcut method that "learns" train-specific patterns (e.g., a biased country classifier that memorizes train images) would score differently on test, creating a visible gap. A genuine GeoCoT improvement — a prompting strategy that better captures visual reasoning — would score similarly on both slices.

With 10 images per slice and 4 countries, the expected gap for a random classifier is ~25% country accuracy (1/4 countries) vs ~50% for an informed one. Shortcuts would show variance between slices; real improvements would show consistency.

### Limitation: sample size

The 20-image sample is too small for statistically significant train/test gaps. With 10 images per slice, the 95% confidence interval on a proportion is roughly ±30% at 50% accuracy. The split is structurally sound but the evaluation power is limited by the sample size. A gym user should treat the train/test gap as directional signal, not a statistically significant test.

### Limitation: no city-level evaluation

The sample lacks city labels, so all experiments omit city_accuracy/city_recall/city_f1. A scientist optimizing city-level accuracy (the paper's primary claim) has no target in this benchmark. This is a known scale-down from the paper's 500-image GeoComp evaluation.

## Runtime Observations

- Both slices use the same prediction artifacts (`results/*.json`) — no re-inference is needed.
- Evaluation is fast: JSON loading + classification scoring over 10 images per file.
- Scripts are self-contained: a future agent just runs `evaluate_train.sh` or `evaluate_test.sh`.
- No cross-imports between train and test; reading either folder reveals nothing about the other.