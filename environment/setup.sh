#!/bin/bash
# Runtime environment setup for GeoCoT reproduction.
# This script runs at JOB EXECUTION TIME inside the container.

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
TORCH_OK=false
if python3 -c "import torch; print('torch', torch.__version__)" 2>/dev/null; then
    echo "  torch already present"
    TORCH_OK=true
fi

TRANSFORMERS_OK=false
if python3 -c "from transformers import Qwen2_5_VLForConditionalGeneration; print('transformers OK')" 2>/dev/null; then
    echo "  transformers + Qwen2.5VL already present"
    TRANSFORMERS_OK=true
fi

QVLU_OK=false
if python3 -c "from qwen_vl_utils import process_vision_info; print('qwen_vl_utils OK')" 2>/dev/null; then
    echo "  qwen_vl_utils already present"
    QVLU_OK=true
fi

# Only install missing packages.
if [ "$TORCH_OK" = false ]; then
    echo "Installing CUDA-enabled torch + torchvision..."
    uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi

if [ "$TRANSFORMERS_OK" = false ] || [ "$QVLU_OK" = false ]; then
    echo "Installing transformers and other packages..."
    uv pip install --system transformers accelerate qwen_vl_utils protobuf pillow numpy pandas huggingface_hub
fi

echo "=== Runtime environment setup complete ==="
