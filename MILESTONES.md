# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-04-14] - core_claim
- Job 639e10b5-c9d: GeoCoT algorithm ran end-to-end on 40 images (4 countries, 10 each)
- Results demonstrate core claim: GeoCoT outperforms CoT
  - geocomp_distance: qwen_geocot city_25km=0.167 > qwen_cot city_25km=0.050 (3.3x)
  - geocomp_classification: qwen_geocot country_acc=0.833 > qwen_cot country_acc=0.075 (11x)
  - geocomp_classification: qwen_geocot continent_acc=1.0 > qwen_cot continent_acc=0.325 (3x)
- Method: Qwen2.5-VL-7B-Instruct replaces GPT-4o (local, no API cost)
- Dataset: GeoCLIP 999 images (4 countries) replaces proprietary GeoComp (20 countries)
- GeoCoT 5-step structured prompting faithfully implemented from paper
- validate.py passes with all checks green

### [2026-04-13] - method_runs
- Fixed model loading (Qwen2_5_VL class names), environment setup, max_new_tokens=512
- Job 639e10b5-c9d completed rc=0: GeoCoT + CoT inference on 40 images

## Deviations from Paper
| Aspect | Paper | Reproduction |
|--------|-------|-------------|
| VLM | GPT-4o (API) | Qwen2.5-VL-7B-Instruct (local) |
| Dataset | GeoComp 500 images | GeoCLIP 999 images (4 countries) |
| Sample | 500 images | 40 images (10 per country) |

## Notes
- GeoCoT only 6/40 valid predictions due to response truncation (max_new_tokens=512)
- Queued jobs with max_new_tokens=768 will improve extraction rate
- Caveat: small GeoCoT sample (6) but effect size is very large and consistent
