#!/bin/bash
# Minimal setup.sh for GeoCoT reproduction.
# Only installs the packages absolutely needed for Qwen2.5-VL inference.
# Everything else comes from pylib (already on GPFS, already mounted).

set -e

echo "=== Installing Python packages (minimal) ==="

# The Azure Linux base has python3 already. Verify.
python3 --version

# Install pip if needed
python3 -m pip --version >/dev/null 2>&1 || apt-get update && apt-get install -y python3-pip

export PIP_BREAK_SYSTEM_PACKAGES=1

# Only install the packages we ABSOLUTELY need:
# 1. transformers (for Qwen2VL model)
# 2. qwen-vl-utils (for image processing)
# Everything else (torch, pillow, numpy, pandas, etc.) is in pylib.
echo "Installing packages..."
python3 -m pip install \
    transformers>=4.40.0 \
    qwen-vl-utils \
    accelerate \
    protobuf \
    2>&1 | tail -10

# Verify
python3 -c "import transformers; print(f'transformers {transformers.__version__}')"
python3 -c "from qwen_vl_utils import process_vision_info; print('qwen_vl_utils OK')"

echo "=== Python packages installed successfully ==="