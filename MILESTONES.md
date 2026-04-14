# Reproduction Milestones

**Current: none**

## Progress Log

### [2026-04-13 20:35] - none
- Job 639e10b5-c9d: GeoCoT 6/56 valid, CoT 40/56 valid
- Root cause: (1) max_new_tokens=512 truncated responses before "Final Prediction" summary, (2) parse_location_prediction couldn't match "Final Prediction: appears to be from [Country]" format

### [2026-04-14 00:20] - none (resubmit)
- Fixed: max_new_tokens=768 (was 512)
- Fixed: parse_location_prediction handles "appears to be from [Country], [Continent]" format
- Fixed: handles "Argentina, Uruguay, or Chile" → takes first country
- Job 2be9010c-e92 queued with updated code

## Stop Justification
- Completed at milestone `method_runs`.
