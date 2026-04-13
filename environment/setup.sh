#!/bin/bash
# Runtime environment setup for GeoCoT reproduction.
# This script runs at JOB EXECUTION TIME inside the container.

set -e

echo "=== Setting up GeoCoT environment (runtime) ==="

# Remove any conflicting packages in user site-packages that would shadow system installs.
# The base Ubuntu image has old huggingface_hub in /root/.local that conflicts
# with the transformers version we install to /usr/local/lib.
python3 -m pip uninstall -y huggingface_hub 2>/dev/null || true
rm -rf /root/.local/lib/python3.10/site-packages/huggingface_hub* 2>/dev/null || true
rm -rf /root/.local/lib/python3.10/site-packages/transformers* 2>/dev/null || true

# Ensure uv is available
pip install --break-system-packages uv 2>/dev/null || pip install uv 2>/dev/null || true
export UV_SYSTEM_PYTHON=1 UV_BREAK_SYSTEM_PACKAGES=1

# Remove any broken system torch
python3 -m pip uninstall -y torch 2>/dev/null || true

# Install CUDA-enabled PyTorch and torchvision (qwen_vl_utils requires torchvision)
echo "Installing CUDA-enabled torch..."
uv pip install --system --no-cache \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124 2>/dev/null || \
python3 -m pip install --break-system-packages --no-cache \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install all remaining packages together
echo "Installing transformers and other packages..."
uv pip install --system --no-cache \
    transformers \
    accelerate \
    qwen_vl_utils \
    protobuf \
    pillow \
    numpy \
    pandas \
    huggingface_hub \
    2>/dev/null || \
python3 -m pip install --break-system-packages --no-cache \
    transformers \
    accelerate \
    qwen_vl_utils \
    protobuf \
    pillow \
    numpy \
    pandas \
    huggingface_hub

# Set HuggingFace cache to shared location
export HF_HOME="/home/user/shared/models/hf"
export TRANSFORMERS_CACHE="/home/user/shared/models/hf"
export HF_HUB_OFFLINE="1"

# Verify packages
echo "Verifying packages..."
python3 -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'transformers {transformers.__version__}')"
python3 -c "from transformers import Qwen2VLForConditionalGeneration; print('Qwen2VL import OK')"
python3 -c "from qwen_vl_utils import process_vision_info; print('qwen_vl_utils OK')"
python3 -c "import accelerate; print(f'accelerate {accelerate.__version__}')"

echo "=== Runtime environment setup complete ==="
