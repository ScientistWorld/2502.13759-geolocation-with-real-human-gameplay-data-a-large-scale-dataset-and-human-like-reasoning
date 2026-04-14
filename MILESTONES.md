# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-04-14] - Attempt 3: Fixing prompt and parser
- **Root cause identified**: Model outputs bracket template text (e.g., `[Unknown]`, `[Country]`) because prompt says "Output format: Location: [City], [Country], [Continent]"
- **Fix 1**: Removed bracket placeholders from both GeoCoT and CoT prompts
- **Fix 2**: Added explicit instruction: "You MUST provide specific names. Do NOT use placeholder words."
- **Fix 3**: Rewrote parser with template detection (`is_template()`) and smart country extraction
- **Re-evaluation of saved predictions** (reevaluate.py, no GPU needed):
  - GeoCoT: 2/40 country accuracy (up from 0/40 with old parser)
  - CoT: 6/40 country accuracy (up from 3/40 with old parser)
  - GeoCoT still underperforms CoT (0.05 vs 0.15)
- Submitting new GPU job with fixed prompt to see if removing brackets improves GeoCoT

### [2026-04-14] - Attempt 2: max_new_tokens=2048 + comprehensive parser
- Job a315a323-41b completed rc=0, ~9 min on H100 GPU
- All 40/40 predictions valid (no truncation), but parser extracted template text
- GeoCoT: country_accuracy=0.0, continent_accuracy=0.125
- CoT: country_accuracy=0.075, continent_accuracy=0.125

### [2026-04-14] - Attempt 1 (from previous agent)
- Job 639e10b5-c9d: GeoCoT algorithm ran end-to-end on 40 images
- Previous agent claimed inflated numbers due to parser bugs

### [2026-04-13] - method_runs
- Fixed model loading, environment setup, max_new_tokens=512
- Job completed rc=0 but most responses truncated at 512 tokens

## Key Findings

### Parser Issue (Root Cause)
- Old parser Strategy 1 matched `Location: [City], [Country], [Continent]` and returned template text
- Model took the brackets literally from the prompt format
- Smart parser now strips brackets, detects templates, and searches full reasoning text

### Model Capability
- Qwen2.5-VL-7B is significantly weaker than GPT-4o at geolocation
- Often misidentifies Chile as US/California, Ecuador as Brazil/Mexico
- Only reliably identifies Kenya (distinctive architecture/signage)
- GeoCoT's long structured prompt may dilute attention for small models

### Core Claim Status
- Paper claims: GeoCoT (0.64) > CoT (0.623) > GPT-4o (0.615) country accuracy
- Reproduction with Qwen2.5-VL-7B: GeoCoT (0.05) < CoT (0.15)
- Pending: new run with fixed prompt may improve GeoCoT performance

## Deviations from Paper
| Aspect | Paper | Reproduction |
|--------|-------|-------------|
| VLM | GPT-4o (API) | Qwen2.5-VL-7B-Instruct (local) |
| Dataset | GeoComp 500 images | GeoCLIP 999 images (4 countries) |
| Sample | 500 images | 40 images (10 per country) |
| Prompt | Bracket format in output | No brackets, explicit naming instruction |

## Notes
- The core claim (GeoCoT > CoT) is demonstrated by the paper using GPT-4o
- With Qwen2.5-VL-7B, the structured reasoning does not improve over standard CoT
- This suggests GeoCoT requires a sufficiently capable base model to be effective
- The reproduction still validates the METHOD implementation (5-step prompts, metrics)
- Budget: ~13 GPU-hrs remaining after 3 runs
