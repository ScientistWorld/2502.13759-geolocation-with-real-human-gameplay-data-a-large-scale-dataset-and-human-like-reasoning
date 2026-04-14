#!/bin/bash
# Runtime environment setup for GeoCoT reproduction.
# This script runs at JOB EXECUTION TIME inside the container.
# Packages persist in overlay between runs — only install what's missing.

set -e

echo "=== Setting up GeoCoT environment (runtime) ==="

export UV_SYSTEM_PYTHON=1 UV_BREAK_SYSTEM_PACKAGES=1

# Remove conflicting packages in user site-packages that shadow system installs.
pip uninstall -y huggingface_hub 2>/dev/null || true
rm -rf /root/.local/lib/python3.10/site-packages/huggingface_hub* 2>/dev/null || true
rm -rf /root/.local/lib/python3.10/site-packages/transformers* 2>/dev/null || true

# Set HuggingFace cache to shared location
export HF_HOME="/home/user/shared/models/hf"
export TRANSFORMERS_CACHE="/home/user/shared/models/hf"
export HF_HUB_OFFLINE="1"

# Check which packages are already present.
echo "Checking existing packages..."
if python3 -c "import torch; print('torch OK')" 2>/dev/null; then
    echo "  torch already present"
fi
if python3 -c "import torchvision; print('torchvision OK')" 2>/dev/null; then
    echo "  torchvision already present"
fi
if python3 -c "from transformers import Qwen2_5_VLForConditionalGeneration; print('Qwen2_5_VL OK')" 2>/dev/null; then
    echo "  transformers + Qwen2_5_VL already present"
fi

# Install torchvision ONLY if missing (needed by qwen_vl_utils even when torch is present).
if ! python3 -c "import torchvision" 2>/dev/null; then
    echo "Installing torchvision..."
    uv pip install --system torchvision --index-url https://download.pytorch.org/whl/cu124
fi

# Verify qwen_vl_utils works.
if python3 -c "from qwen_vl_utils import process_vision_info; print('qwen_vl_utils OK')" 2>/dev/null; then
    echo "  qwen_vl_utils already present"
else
    echo "Installing qwen_vl_utils..."
    uv pip install --system qwen_vl_utils
fi

echo "=== Runtime environment setup complete ==="
