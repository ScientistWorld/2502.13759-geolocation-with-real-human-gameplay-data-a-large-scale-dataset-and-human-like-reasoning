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

### [2026-04-10 16:xx] - method_runs (regressed)
- GPU job #2 failed with PyTorch C extension error
- Root cause: /home/user/pylib source-tree torch conflicted with overlay CUDA torch
- Fix: removed /home/user from PYTHONPATH in run.sh; rewritten setup.sh

### [2026-04-11 00:xx] - method_runs
- Rewrote setup.sh and run.sh to fix torch conflict
- Removed /home/user from PYTHONPATH - only add method/ and eval/ subdirs
- Overlay will be rebuilt on next job submission
- Submitting GPU job with fixes
