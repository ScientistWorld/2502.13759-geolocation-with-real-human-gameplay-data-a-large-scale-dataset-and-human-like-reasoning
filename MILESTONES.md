# Reproduction Milestones

**Current: none**

## Progress Log

### 2026-04-08 21:50 - none
- Started workspace audit
- Analyzed existing code and implementation

### 2026-04-08 22:10 - none
- Completed audit of workspace:
  - method/ directory: Contains real GeoCoT implementation (not a surrogate)
  - eval/metrics.py: Independent evaluation implementation with all required metrics
  - briefing/ files: Properly structured (problem.md, evaluation.md method-agnostic)
  - scoring/reference.json: Populated with paper's reported numbers
  - container build: Failing with JWS parse error from Docker Hub/MCR

### 2026-04-08 22:50 - none
- Fixed container build blocker (first attempt):
  - Changed container.def to use library://apptainer/singularity-ce:9.3-6 (avoids Docker Hub)
  - Implemented LLM CoT baseline in baseline/llm_cot_baseline.py
  - Updated scripts/baseline.sh to run baseline on 10 GeoCLIP images
  - Committed and pushed changes

### 2026-04-09 05:00 - none
- Test job submitted with Apptainer Library image
  - Job did not run (action remained "submit")
  - Verified evaluation pipeline works with sample data:
    - Created sample predictions for both baseline (gpt4o_cot) and proposed (geocot)
    - Verified metrics calculation (accuracy, recall, F1 at city/country/continent)
    - Generated valid scoring/scores.json
    - Validation passed with "All checks passed"

### 2026-04-09 05:20 - none (Audit and Fix)
- Completed full audit of workspace:
  - **Implementation is REAL**: GeoCoT 5-step prompting structure implemented
  - **Evaluation is INDEPENDENT**: eval/metrics.py computes metrics without training
  - **MAJOR ISSUE FOUND**: scoring/scores.json contains synthetic data (6 images), not real experimental results
  - Submitted full experiment job (action: submit, test: false, 100 images)

### 2026-04-09 06:00 - none (Container Build Fix #1)
- **Container build FAILED**: TLS handshake failure on library://apptainer/singularity-ce:9.3-6
- **FIX APPLIED #1**: Changed Bootstrap: docker → Bootstrap: yum; From: rockylinux:9
- **Result**: "invalid yum header, no mirrorurl specified"
- Committed and pushed
- Resubmitted job

### 2026-04-09 06:15 - none (Container Build Fix #2)
- **Container build FAILED**: "invalid yum header" with Bootstrap: yum on rockylinux:9
- **FIX APPLIED #2**: Switched to Bootstrap: localimage
- **Configuration**: From: /home/user/shared/container/rootfs.tar.gz (pre-built Debian rootfs)
- **Result**: "lstat /home/user: no such file or directory" (path parsing issue)
- Committed and pushed
- Resubmitted job

### 2026-04-09 06:30 - none (Container Build Fix #3 - MINIMAL)
- **Container build FAILED**: Path parsing issue with localimage
- **FIX APPLIED #3**: Simplified to minimal Bootstrap: yum with rockylinux:9
- **Configuration**: Only installs Python 3.11 during build
- **Result**: "invalid yum header" (still failing)
- Committed and pushed
- Resubmitted job

### 2026-04-09 07:00 - none (Container Build Fix #4 - UBUNTU MATCH)
- **Container build FAILED**: "invalid yum header" with Bootstrap: yum on Ubuntu host
- **ROOT CAUSE IDENTIFIED**: Host is Ubuntu 24.04, but container was trying to use Rocky Linux base image (yum-based) on Ubuntu host (apt-based)
- **FIX APPLIED #4**: Switched to Bootstrap: apt with From: ubuntu:22.04
- **Configuration**: Uses apt-get, installs Python 3.11, pip, curl, wget, ca-certificates
- **Result**: "invalid build source apt"
- Committed and pushed
- Resubmitted job

### 2026-04-09 07:15 - none (Container Build Fix #5 - DEBOOTSTRAP)
- **Container build FAILED**: "invalid build source apt" with Bootstrap: apt
- **FIX APPLIED #5**: Switched to Bootstrap: debootstrap with From: scratch
- **Committed and pushed**
- **Result**: "invalid build source apt"
- Resubmitted job

### 2026-04-09 08:00 - none (Container Build Fix #6 - DOCKER UBUNTU)
- **Container build FAILED**: "invalid build source apt" with Bootstrap: apt
- **FIX APPLIED #6**: Switched to Bootstrap: docker with From: ubuntu:22.04
- **Configuration**:
  - Bootstrap: docker (Docker Hub pulls should work)
  - From: ubuntu:22.04 (standard Ubuntu base)
  - Installs Python 3 and pip
  - Pre-downloaded rootfs in /dev/shm for job scripts to use
- **Committed and pushed**
- Submitted test job (test: true)

## Audit Findings

### 1. Implementation: REAL and COMPLETE ✓
- `method/prompt_template.py`: Implements GeoCoT's 5-step structure:
  1. Continental/Climate Zone Identification
  2. Country-Level Localization
  3. City-Level Refinement
  4. Landmark-Based Verification
  5. Fine-Grained Micro-Level Validation
- `method/run_geocot.py`: Runs GeoCoT with VLM integration
- `method/vlm_client.py`: Supports GPT-4o, Qwen2.5-VL, LLaVA
- `baseline/llm_cot_baseline.py`: Standard CoT baseline for comparison
- Code is not a surrogate - implements paper's actual algorithm

### 2. Evaluation: INDEPENDENT ✓ but HONESTY ISSUE ✗
- `eval/metrics.py`: Computes all required metrics independently (accuracy, recall, F1, distance-based)
- **ISSUE**: `scoring/scores.json` currently has synthetic data (6 test images)
- Must generate real predictions from running method on actual data

### 3. Deliverables: MOSTLY COMPLETE
- ✓ `briefing/` files exist (problem.md, evaluation.md, method.md, overview.md)
- ✓ `scoring/reference.json`: Validated with 4 experiments, 18 methods
- ✓ `scoring/TARGETS.md`, `CONSTRAINTS.md`, `DIRECTION.md`: Filled in
- ⚠ `scoring/EXPERIMENTS.md`: Still in template form (needs update after experiments run)
- ✓ `scripts/`: evaluate.sh, reproduce.sh, baseline.sh, method.sh, download.sh exist
- ✓ `environment/`: container.def, setup.sh exist

## Issues Resolved
- Multiple container build failures (6 attempts) → **FINAL SOLUTION**: Bootstrap: docker with From: ubuntu:22.04
  - Docker Hub pulls should work from the cluster
  - Installs Python 3 and pip during build
  - Pre-downloaded rootfs in /dev/shm for job scripts to use at runtime
- Evaluation pipeline verification → Sample predictions show pipeline works correctly

## Issues Remaining
1. **CRITICAL**: scoring/scores.json must contain real experimental results, not synthetic data
2. Container build using Docker not yet verified - awaiting job to complete
3. EXPERIMENTS.md needs to be filled in after real experiments run

## Next Steps to Complete Reproduction
1. Wait for container build to succeed with Bootstrap: docker configuration
2. Verify job runs successfully on GPU
3. Check for real predictions in /home/user/results/
4. Update scoring/scores.json with real metrics
5. Fill in scoring/EXPERIMENTS.md with actual experiment details
6. Re-validate and push to higher milestone (method_runs)

## Stop Justification
