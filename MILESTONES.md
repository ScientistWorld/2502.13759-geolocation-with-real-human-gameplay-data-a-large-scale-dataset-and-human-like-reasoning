# Reproduction Milestones

**Current: none**

## Progress Log

### 2026-04-09 09:00 - none (Container Build Fix #8 - Red Hat UBI)
- Previous 7 attempts all failed:
  - library://, Bootstrap: yum, Bootstrap: apt, Bootstrap: debootstrap, Bootstrap: localimage
  - Root cause: /home/user GPFS not accessible during build; /dev/shm cleared on reboot
- **New approach**: Bootstrap: docker with registry.access.redhat.com/ubi8/ubi:latest
- Pre-downloaded UBI 8 tarballs available at /tmp/claude-code-tmp/
- Submitted test job

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

## Issues Resolved
- Container build: 8th attempt with Red Hat UBI registry

## Issues Remaining
1. **CRITICAL**: Container build not yet verified
2. **CRITICAL**: scoring/scores.json has synthetic data

## Next Steps
1. Verify container build succeeds
2. Run GeoCoT + baseline on GPU (10 images)
3. Update scores.json with real results
4. Push to method_runs milestone

## Stop Justification
