# Reproduction Milestones

**Current: none**

## Progress Log

### 2026-04-09 09:30 - none (9th Container Build Attempt)
- Previous 8 attempts failed due to:
  - /home/user GPFS not accessible during build
  - /dev/shm cleared on reboot
  - Docker Hub / RedHat / MCR registry issues
  - Apptainer bootstrap method not recognized (apt, yum, debootstrap)
- **RE-discovered**: Alpine + MCR approach worked in iteration 1 and reached method_runs
- **9th attempt**: Bootstrap: docker Alpine 3 + curl download Azure Linux layer from MCR
  - This bypasses the docker conveyor JWS issue
  - Alpine is tiny (~5MB) from Docker Hub
  - Azure Linux layer downloaded directly via curl

## Audit Findings

### 1. Implementation: REAL and COMPLETE ✓
- `method/prompt_template.py`: GeoCoT 5-step prompting
- `method/run_geocot.py`: VLM integration
- `method/vlm_client.py`: GPT-4o, Qwen2.5-VL, LLaVA support
- `baseline/llm_cot_baseline.py`: Standard CoT baseline
- Code is not a surrogate - implements paper's actual algorithm

### 2. Evaluation: INDEPENDENT ✓ but SCORES ISSUE ✗
- `eval/metrics.py`: Independent evaluation
- `scoring/scores.json`: Has synthetic data (needs real GPU results)

### 3. Deliverables: MOSTLY COMPLETE ✓
- All briefing, scoring, scripts, environment files in place

## Issues Remaining
1. **CRITICAL**: Container build not yet verified
2. **CRITICAL**: scoring/scores.json has synthetic data

## Next Steps
1. Verify container build succeeds (Alpine + MCR approach)
2. Run GeoCoT + baseline on GPU (10 images)
3. Update scores.json with real results
4. Push to method_runs milestone

## Stop Justification
