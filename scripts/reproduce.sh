#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# Workflow:
#   1. Download models and datasets (if not present)
#   2. Run GeoCoT vs CoT inference with local VLM
#   3. Evaluate and generate scores.json

set -e

cd /home/user

echo "=== GeoCoT Full Reproduction ==="

# Step 1: Download data (idempotent)
echo "Step 1: Ensuring data is available..."
bash scripts/download.sh

# Step 2: Run inference and evaluation (all in one script)
echo "Step 2: Running GeoCoT + CoT inference and evaluation..."
bash scripts/run.sh

echo ""
echo "=== Reproduction Complete ==="
echo "Results: /home/user/scoring/scores.json"
