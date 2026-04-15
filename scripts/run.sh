#!/bin/bash
# CPU validation job for the GeoCoT reproduction package.
#
# This job deliberately does not run VLM inference. It re-scores the existing
# method artifacts under results/ and validates the workspace contract. Use
# scripts/method.sh for new GeoCoT inference runs.

set -euo pipefail

cd /home/user

echo "=========================================="
echo "GeoCoT scoring validation - $(date)"
echo "=========================================="

bash /home/user/scripts/evaluate.sh /home/user/results
python3 /home/user/validate.py --compare

echo "=========================================="
echo "Validation complete - $(date)"
echo "=========================================="
