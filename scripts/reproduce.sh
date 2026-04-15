#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models were downloaded on an internet-enabled node
#   - Working directory is /home/user
#
# Workflow:
#   1. Verify downloaded models and tracked data are available
#   2. Run GeoCoT vs CoT inference with local VLM
#   3. Evaluate and generate scores.json

set -e

cd /home/user

echo "=== GeoCoT Full Reproduction ==="

# Step 1: verify data and model availability. Compute nodes have no internet,
# so run scripts/download.sh before submitting this reproduction job.
echo "Step 1: Verifying data and model availability..."
if [ ! -f "/home/user/data/geoclip/geoclip.csv" ]; then
    echo "ERROR: data/geoclip/geoclip.csv is missing."
    exit 1
fi

if [ ! -f "/home/user/data/downloads/Qwen2.5-VL-32B-Instruct/config.json" ] && \
   [ ! -f "/home/user/data/downloads/Qwen2.5-VL-7B-Instruct/config.json" ] && \
   [ ! -f "/home/user/checkpoints/Qwen2.5-VL-32B-Instruct/config.json" ] && \
   [ ! -f "/home/user/checkpoints/Qwen2.5-VL-7B-Instruct/config.json" ]; then
    echo "ERROR: no Qwen2.5-VL checkpoint found. Run scripts/download.sh before compute jobs."
    exit 1
fi

# Step 2: Run inference and evaluation. The ablation script is resumable and
# skips prediction files that already exist.
echo "Step 2: Running GeoCoT + CoT inference and evaluation..."
bash scripts/run_ablation.sh

echo ""
echo "=== Reproduction Complete ==="
echo "Results: /home/user/scoring/scores.json"
