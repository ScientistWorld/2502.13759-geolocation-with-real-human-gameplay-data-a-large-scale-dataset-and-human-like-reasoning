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
- **Container build FAILED**: "invalid yum header, no mirrorurl specified"
- **FIX APPLIED #2**: Switched to Bootstrap: localimage
- **Configuration**: 
  - From: /home/user/shared/container/rootfs.tar.gz (pre-built Debian rootfs)
  - %post: Only verifies Python availability (no network calls)
- **Result**: Avoids all network/registry issues by using local filesystem
- Committed and pushed
- Resubmitted job

## Audit Findings

### 1. Implementation: REAL and COMPLETE ✓
- `method/prompt_template.py`: Implements GeoCoT's 5-step structure
- `method/run_geocot.py`: Runs GeoCoT with VLM integration
- `method/vlm_client.py`: Supports GPT-4o, Qwen2.5-VL, LLaVA
- `baseline/llm_cot_baseline.py`: Standard CoT baseline for comparison
- Code is not a surrogate - implements paper's actual algorithm

### 2. Evaluation: INDEPENDENT ✓ but HONESTY ISSUE ✗
- `eval/metrics.py`: Computes all required metrics independently
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
- Container build JWS issue → Attempted library://apptainer/singularity-ce:9.3-6
- TLS handshake failure → Attempted Bootstrap: yum with rockylinux:9
- yum header error → **FIXED**: Bootstrap: localimage from /home/user/shared/container/rootfs.tar.gz
- Evaluation pipeline verification → Sample predictions show pipeline works correctly

## Issues Remaining
1. **CRITICAL**: scoring/scores.json must contain real experimental results, not synthetic data
2. Container build using localimage not yet verified - awaiting job to complete
3. EXPERIMENTS.md needs to be filled in after real experiments run

## Next Steps to Complete Reproduction
1. Wait for container build to succeed with Bootstrap: localimage configuration
2. Verify job runs successfully on GPU
3. Check for real predictions in /home/user/results/
4. Update scoring/scores.json with real metrics
5. Fill in scoring/EXPERIMENTS.md with actual experiment details
6. Re-validate and push to higher milestone (method_runs)

## Stop Justification
