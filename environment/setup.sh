#!/bin/bash
# Evaluation-only submit: train/test rescoring reads saved JSON artifacts and
# uses only the Python standard library, so no runtime package installation is
# required on the compute node.

set -e

echo "=== Runtime setup not required for split scoring job ==="
