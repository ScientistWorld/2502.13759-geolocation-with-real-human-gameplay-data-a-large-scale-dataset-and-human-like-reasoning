# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-10 04:38] - none
- Initial workspace setup from previous agent
- Paper read: GeoCoT for image geolocation with 5-step chain-of-thought reasoning

### [2026-04-10 05:xx] - method_runs
- Implemented GeoCoT 5-step prompting framework faithfully following the paper
- VLM client supports Qwen2.5-VL-7B-Instruct (local), GPT-4o (API), LLaVA
- GeoCLIP dataset: 999 images across 4 countries (Kenya, Ecuador, Chile, Madagascar)
- Previous GPU job ran but only processed 6 images due to CUDA_VISIBLE_DEVICES="" bug
- Fixed critical bug: removed CUDA_VISIBLE_DEVICES="" that blocked GPU usage
- Reduced sample to 80 balanced images (20 per country)
- Improved checkpointing (save every 10 images)
- Resubmitting GPU job with fixes

### [2026-04-10 16:xx] - core_claim (in progress)
- GPU job submitted with fixed run.sh
- Expecting: GeoCoT outperforms CoT on country-level accuracy (primary claim)
- Sample: 80 images across 4 countries with Qwen2.5-VL-7B-Instruct
