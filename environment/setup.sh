#!/bin/bash
# Install Python packages for the GeoCoT environment.
#
# This script runs during container build (in the Apptainer overlay).
# All models and datasets are pre-downloaded on the login node.
#
# IMPORTANT: Install into SYSTEM Python, not a virtual environment.

set -e

echo "=== Installing Python packages ==="

# Detect python3
PYTHON_BIN="python3"
$PYTHON_BIN -c "import sys; print(f'Python: {sys.version}')" 2>/dev/null || {
    echo "python3 not found, installing..."
    apt-get update && apt-get install -y python3 python3-pip
}

# Use pip with --break-system-packages for Ubuntu 22.04
export PIP_BREAK_SYSTEM_PACKAGES=1

# Install uv for fast installation (pre-installed in the base)
which uv >/dev/null 2>&1 || {
    echo "Installing uv..."
    pip3 install uv
}
export UV_SYSTEM_PYTHON=1
export UV_BREAK_SYSTEM_PACKAGES=1

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -5 || \
uv pip install torch torchvision 2>&1 | tail -5

# Install all other packages
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
    qwen-vl-utils \
    2>&1 | tail -10

# Verify installation
python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null || echo "PyTorch not available (no CUDA in build env)"
python3 -c "import transformers; print(f'transformers {transformers.__version__}')"
python3 -c "import pandas; print(f'pandas {pandas.__version__}')"
python3 -c "import qwen_vl_utils; print('qwen_vl_utils OK')"

echo "=== Python packages installed successfully ==="
