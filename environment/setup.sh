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

# CRITICAL: Remove pre-installed non-CUDA torch/torchvision from site-packages.
# The pylib dir contains torch 2.5.1+cu124 without CUDA kernels.
# When qwen_vl_utils imports torchvision, it triggers the pylib version
# which fails with "operator torchvision::nms does not exist".
# We must remove it so our new CUDA-enabled torch is used instead.
echo "Removing pre-installed non-CUDA torchvision to prevent conflicts..."
rm -rf /root/.local/lib/python3.10/site-packages/torchvision 2>/dev/null || true
rm -rf /root/.local/lib/python3.10/site-packages/torch 2>/dev/null || true
echo "Pre-installed torch removal complete."

# Install qwen-vl-utils FIRST (before torch/torchvision) so it gets the correct deps
echo "Installing qwen-vl-utils first..."
uv pip install qwen-vl-utils 2>&1 | tail -5

# Install PyTorch + torchvision TOGETHER from CUDA index (prevents version mismatch)
echo "Installing PyTorch with matching torchvision..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -5

# Verify
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
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
python3 -c "import pandas; print(f'pandas {pandas.__version__}')"
python3 -c "from qwen_vl_utils import process_vision_info; print('qwen_vl_utils OK')"

echo "=== Python packages installed successfully ==="
