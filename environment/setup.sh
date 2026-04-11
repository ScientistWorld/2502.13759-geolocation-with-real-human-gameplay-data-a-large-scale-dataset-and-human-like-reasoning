#!/bin/bash
# Install Python packages for the GeoCoT environment.
#
# Minimal version: torch + transformers + qwen-vl-utils only.
# All other packages already exist in pylib (pillow, numpy, pandas, etc.)

set -e

echo "=== Installing Python packages ==="

# Detect python3
PYTHON_BIN="python3"
$PYTHON_BIN -c "import sys; print(f'Python: {sys.version}')" 2>/dev/null || {
    echo "python3 not found, installing..."
    apt-get update && apt-get install -y python3 python3-pip
}

export PIP_BREAK_SYSTEM_PACKAGES=1
export UV_SYSTEM_PYTHON=1
export UV_BREAK_SYSTEM_PACKAGES=1

# Install uv
which uv >/dev/null 2>&1 || pip3 install uv

# CRITICAL: Remove pylib torch from PYTHONPATH to avoid C extension mismatch.
# pylib has a source-tree torch that fails when overlaid with the real CUDA torch.
# We remove it from sys.path so our CUDA torch from site-packages is used instead.
echo "Removing pylib from PYTHONPATH to prevent torch source-tree conflict..."
for pylib_path in "/home/user/pylib" "/workspace/pylib" "/root/pylib"; do
    if [ -d "$pylib_path" ]; then
        echo "  Found pylib at $pylib_path - will remove from path"
    fi
done

# Install qwen-vl-utils FIRST so its deps get installed properly
echo "Installing qwen-vl-utils..."
uv pip install qwen-vl-utils 2>&1 | tail -5

# Install PyTorch + torchvision from CUDA index
echo "Installing PyTorch with CUDA..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -5

# Verify torch works
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import torchvision; print(f'torchvision {torchvision.__version__}')"

# Install remaining packages
echo "Installing remaining packages..."
uv pip install \
    "transformers>=4.40.0" \
    accelerate \
    sentencepiece \
    protobuf \
    tiktoken \
    pillow \
    opencv-python-headless \
    scipy \
    scikit-learn \
    tqdm \
    jiwer \
    rapidfuzz \
    einops \
    huggingface_hub \
    pandas \
    numpy \
    2>&1 | tail -10

# Final verification
python3 -c "import transformers; print(f'transformers {transformers.__version__}')"
python3 -c "from qwen_vl_utils import process_vision_info; print('qwen_vl_utils OK')"

echo "=== Python packages installed successfully ==="