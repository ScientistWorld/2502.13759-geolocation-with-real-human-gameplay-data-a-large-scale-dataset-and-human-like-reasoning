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

### [2026-04-07-08] - method_runs (submitted, Docker Hub rate-limit workaround)
- Docker Hub rate-limiting: all images (ubuntu, alpine, etc.) returning 429 TOOMANYREQUESTS
- Downloaded Ubuntu 24.04 minimal rootfs from cloud-images.ubuntu.com CDN to /home/user/shared/container/rootfs_build/
- Tried localimage with relative path: build node path issue (GPFS not accessible).
- Tried Bootstrap:http, Bootstrap:sh — not supported by Apptainer 1.4.5
- Bootstrap:docker with Red Hat UBI 9 from registry.access.redhat.com — ALSO rate-limited ("registry response malformed")
- Bootstrap:yum fails: cannot create device nodes in build directory ("operation not permitted")
- Retry: docker://index.docker.io/library/ubuntu:24.04 (fully qualified) — STILL rate-limited
- Switch to: Bootstrap:localimage with pre-extracted Ubuntu 24.04 rootfs
  - Build node accesses GPFS at `/scratch/gpfs/ZHUANGL/tl0463/ResearchGym/Infrastruture/swarm/` (not `/home/user/`)
  - Compute node accesses same GPFS at `/home/user/` — same filesystem, different mount points
  - `From: /scratch/gpfs/ZHUANGL/tl0463/ResearchGym/Infrastruture/swarm/shared/container/rootfs`
  - No Docker Hub pull needed — bypasses rate limits entirely

## Stop Justification

<!-- Do not edit this unless you decide to stop -->
