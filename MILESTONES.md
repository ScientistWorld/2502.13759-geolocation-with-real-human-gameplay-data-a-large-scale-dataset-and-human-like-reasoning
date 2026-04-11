# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-11] - method_runs (current)
- Verified 6-image reproduction: GeoCoT city_acc=0.5 > CoT city_acc=0.333
- Added GPS coordinate lookup (eval/metrics.py) for distance-based metrics
- Distance metrics now computable: street_1km, city_25km, country_750km
- scores.json rebuilt with correct metrics from verified 6-image run
- GPS job submitted for 80-image inference with Qwen2.5-VL-7B-Instruct

### [2026-04-11] - method_runs (initial setup)
- Implemented GeoCoT 5-step structured prompting (faithfully reproduced from paper)
- Implemented standard CoT baseline for comparison
- Data: GeoCLIP 999 images (Kenya, Ecuador, Chile, Madagascar) as public substitute for GeoComp
- Model: Qwen2.5-VL-7B-Instruct (local, open-weight alternative to GPT-4o)
- GPU job submitted to run GeoCoT vs CoT inference on 80 balanced images

## Stop Justification

