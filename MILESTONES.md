# Reproduction Milestones

**Current: method_runs (submitted)**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-04-04] - none
- Read paper thoroughly: GeoCoT (Geographical Chain-of-Thought) for image geolocation
- Paper proposes a 5-step structured prompting framework for VLMs
- Core claim: GeoCoT improves geolocation accuracy by up to 25% vs baselines
- Key challenge: GeoComp dataset and GPT-4o not directly available

### [2026-04-04] - method_runs (in progress)
- Implemented GeoCoT prompting method (5 reasoning steps)
- Implemented VLM client supporting Qwen2.5-VL and LLaVA
- Implemented geolocation evaluation metrics (classification + distance)
- Set up environment with CUDA container
- Created download scripts for model and data
- Created method.sh, baseline.sh, evaluate.sh, reproduce.sh scripts

### [2026-04-05] - method_runs (submitted for GPU execution)
- Discovered Stanford Im2GPS3K URL is 404 (no longer available)
- Found GeoCLIP-data on HuggingFace as alternative (999 images from Kenya/Ecuador/Madagascar/Chile)
- Downloaded full GeoCLIP dataset (70MB) to /home/user/data/geoclip/
- Updated data loader to support GeoCLIP dataset
- Updated VLM client for LLaVA-1.5-7B at /home/user/shared/models/llava-v1.5-7b/
- Updated run.sh to use GeoCLIP + LLaVA pipeline
- Updated container.def to install uv and set PYTHONPATH
- GPU job submitted: GeoCoT vs CoT on 100 GeoCLIP images, evaluate, generate scores.json

### [2026-04-07] - method_runs (submitted for GPU execution, fixed)
- Fixed container.def: switched from library: bootstrap (requires Sylabs remote) to docker://ubuntu:22.04
- Downloaded all 999 GeoCLIP images to /home/user/data/geoclip/ via HuggingFace
- Pre-installed torch 2.6.0+cu124, transformers, pandas to /dev/shm/pylib
- Fixed LLaVAClient: uses llava repo LlavaLlamaForCausalLM (not HF AutoModel)
  - Registers LlavaConfig with model_type=llava_llama override
  - Monkeypatches CLIPVisionConfig/CLIPModel.from_pretrained for offline CLIP loading
  - Uses proper conversation format with <im_start><image><im_end> tokens
- Cleaned up setup.sh to only do pip installs (no model/data downloads)
- GPU job re-submitted with fixes

### [2026-04-07] - core_claim (pending)
- Depends on GPU job completing successfully
- Need to verify GeoCoT outperforms CoT on country/continent accuracy

## Stop Justification

<!-- Do not edit this unless you decide to stop -->
