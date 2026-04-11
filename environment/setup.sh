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

# Remove pylib from sys.path so our CUDA torch takes priority.
# pylib has a source-tree torch that causes C extension errors.
# We handle this by removing it from sys.path BEFORE any other code runs.
# Since we can't modify sys.path before imports in setup.sh, we rely on
# the fact that setup_wrapper.sh runs this first, and then the overlay's
# site-packages (with CUDA torch) will be mounted before /home/user/pylib.

# Only install the packages we ABSOLUTELY need:
# 1. torch with CUDA support
# 2. transformers (for Qwen2VL model)
# 3. qwen-vl-utils (for image processing)
# Everything else (pillow, numpy, pandas, etc.) is in pylib.
echo "Installing PyTorch with CUDA..."
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3

echo "Installing remaining packages..."
python3 -m pip install \
    transformers>=4.40.0 \
    qwen-vl-utils \
    accelerate \
    protobuf \
    pillow \
    2>&1 | tail -10

# Verify
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'transformers {transformers.__version__}')"
python3 -c "from qwen_vl_utils import process_vision_info; print('qwen_vl_utils OK')"

echo "=== Python packages installed successfully ==="