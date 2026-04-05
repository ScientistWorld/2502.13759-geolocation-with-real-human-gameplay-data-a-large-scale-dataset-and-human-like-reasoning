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
#   2. Run GeoCoT with local VLM
#   3. Run standard CoT baseline
#   4. Evaluate and generate scores.json

set -e

cd /home/user

echo "=== GeoCoT Full Reproduction ==="

# Step 1: Download data (idempotent)
echo "Step 1: Ensuring data is available..."
bash scripts/download.sh

# Step 2: Run the method
echo "Step 2: Running GeoCoT method..."
bash scripts/method.sh

# Step 3: Run baseline
echo "Step 3: Running baseline..."
bash scripts/baseline.sh

# Step 4: Evaluate
echo "Step 4: Evaluating..."
bash scripts/evaluate.sh

echo ""
echo "=== Reproduction Complete ==="
echo "Results: /home/user/scoring/scores.json"
