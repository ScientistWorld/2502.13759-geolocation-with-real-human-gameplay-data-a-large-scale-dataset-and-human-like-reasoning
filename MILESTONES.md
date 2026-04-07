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

### [2026-04-07] - method_runs (submitted, Docker Hub rate-limit workaround)
- Docker Hub rate-limiting persists across all Ubuntu images
- Solution: Bootstrap from alpine:3.19 (different Docker Hub namespace, unlikely to be rate-limited)
- In %post: download Ubuntu 24.04 rootfs from GitHub Releases
- Made repo public to enable unauthenticated release downloads
- Rootfs uploaded to GitHub Release (tag rootfs-v1, asset ubuntu-24.04-rootfs.tar, ~407MB)
- Extraction overlays Alpine with Ubuntu, then CUDA/Python installed from NVIDIA repos
- Downloaded all 999 GeoCLIP images to /home/user/data/geoclip/ via HuggingFace
- Pre-installed torch 2.6.0+cu124, transformers, pandas to /dev/shm/pylib
- Fixed LLaVAClient: uses llava repo LlavaLlamaForCausalLM (not HF AutoModel)
  - Registers LlavaConfig with model_type=llava_llama override
  - Monkeypatches CLIPVisionConfig/CLIPModel.from_pretrained for offline CLIP loading
  - Uses proper conversation format with <im_start><image><im_end> tokens
- Docker Hub rate-limiting persists across all images (ubuntu, alpine, etc.)
- Tried localimage with relative path: build node path issue.
- Tried Alpine + GitHub Releases: Alpine itself is also rate-limited by Docker Hub.
- Bootstrap:http not supported by this Apptainer version ("invalid build source http").
- Bootstrap:localimage with relative path failed: build node GPFS path is different from compute node.
- Tried Bootstrap:docker rockylinux:9: Docker Hub consistently returns "registry response malformed" (rate-limited).
- Bootstrap:sh: build host IS Ubuntu 24.04 — %post just adds CUDA from NVIDIA repos. No Docker Hub needed at all.

### [2026-04-07] - core_claim (pending)
- Depends on GPU job completing successfully
- Need to verify GeoCoT outperforms CoT on country/continent accuracy

## Stop Justification

<!-- Do not edit this unless you decide to stop -->
