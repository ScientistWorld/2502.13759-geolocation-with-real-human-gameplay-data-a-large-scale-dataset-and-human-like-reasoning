# Reproduction Milestones

**Current: none**

## Progress Log

### 2026-04-08 21:50 - none
- Started workspace audit
- Analyzed existing code and implementation

### 2026-04-09 05:20 - none (Initial Audit)
- Completed audit of workspace:
  - **Implementation is REAL**: GeoCoT 5-step prompting structure implemented
  - **Evaluation is INDEPENDENT**: eval/metrics.py computes metrics without training
  - **MAJOR ISSUE FOUND**: scoring/scores.json contains synthetic data (6 images), not real experimental results
- Verified evaluation pipeline works with sample data

### 2026-04-09 06:00-08:00 - none (Container Build Iterations)
- 6 container build attempts all failed:
  1. library://apptainer/singularity-ce:9.3-6 → TLS handshake failure
  2. Bootstrap: yum with rockylinux:9 → Invalid yum header
  3. Bootstrap: localimage with rootfs.tar.gz → Path not accessible during build
  4. Bootstrap: apt with ubuntu:22.04 → "invalid build source apt"
  5. Bootstrap: debootstrap from scratch → "invalid build source apt"
  6. Bootstrap: apt only → "invalid build source apt"
  7. **Bootstrap: docker with ubuntu:22.04** → Submitted (unknown result)

### 2026-04-09 09:00 - none (7th Fix Attempt)
- **ROOT CAUSE**: /home/user is GPFS-mounted and not accessible during Apptainer build
- **FIX #7**: Bootstrap: localimage with /home/user/shared/container/rootfs.tar.gz
- The path is accessible because Apptainer build runs on the same node
- Pre-built Debian rootfs is already in shared storage
- Python packages installed at runtime via /dev/shm/pylib
- Resubmitted job with localimage bootstrap

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

### 2. Evaluation: INDEPENDENT ✓ but SCORES ISSUE ✗
- `eval/metrics.py`: Computes all required metrics independently
- **ISSUE**: `scoring/scores.json` has synthetic data (6 test images)
- Must generate real predictions from running method on actual data

### 3. Deliverables: MOSTLY COMPLETE
- ✓ `briefing/` files exist (problem.md, evaluation.md, method.md, overview.md)
- ✓ `scoring/reference.json`: Validated with 4 experiments, 18 methods
- ✓ `scoring/TARGETS.md`, `CONSTRAINTS.md`, `DIRECTION.md`: Filled in
- ⚠ `scoring/EXPERIMENTS.md`: Still in template form
- ✓ `scripts/`: evaluate.sh, reproduce.sh, baseline.sh, method.sh, download.sh exist
- ✓ `environment/`: container.def, setup.sh exist

## Issues Resolved
- 6 container build failures → **7th fix**: Bootstrap: localimage with shared rootfs
- Root cause: GPFS /home/user not accessible during build
- Solution: Use local tarball bootstrap from /home/user/shared (same filesystem, different mount)
- Evaluation pipeline verified with sample data

## Issues Remaining
1. **CRITICAL**: scoring/scores.json has synthetic data - awaiting GPU job results
2. Container build with localimage not yet verified
3. EXPERIMENTS.md needs update after real experiments run

## Next Steps
1. Wait for container build to succeed with Bootstrap: localimage
2. Verify job runs on GPU and produces real predictions
3. Update scoring/scores.json with real metrics
4. Push to method_runs milestone

## Stop Justification
