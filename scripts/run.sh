#!/bin/bash
# CPU scoring job for the train/test evaluation split.

set -euo pipefail

cd /home/user

echo "=========================================="
echo "GeoCoT train/test scoring - $(date)"
echo "=========================================="

bash /home/user/scripts/evaluate_train.sh /home/user/results
bash /home/user/scripts/evaluate_test.sh /home/user/results

echo "=========================================="
echo "Train/test scoring complete - $(date)"
echo "=========================================="
