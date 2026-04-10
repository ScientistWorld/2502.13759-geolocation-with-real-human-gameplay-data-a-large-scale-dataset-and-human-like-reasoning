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

export PIP_BREAK_SYSTEM_PACKAGES=1

# Install uv for fast installation (pre-installed in the base)
which uv >/dev/null 2>&1 || {
    echo "Installing uv..."
    pip3 install uv
}
export UV_SYSTEM_PYTHON=1
export UV_BREAK_SYSTEM_PACKAGES=1

# CRITICAL: Remove pre-installed torch from site-packages before installing CUDA torch.
# The overlay mounts over system Python dirs. When pylib's torch gets in the
# PYTHONPATH before the overlay's torch, it causes "Failed to load C extensions".
echo "Removing pre-installed torch to prevent conflicts..."
rm -rf /root/.local/lib/python3.10/site-packages/torch 2>/dev/null || true
rm -rf /root/.local/lib/python3.10/site-packages/torchvision 2>/dev/null || true

# Check if CUDA PyTorch is already installed (overlay reused from previous build)
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "CUDA PyTorch already installed: $(python3 -c 'import torch; print(torch.__version__)')"
else
    echo "Installing PyTorch with matching torchvision (CUDA)..."
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -5
fi

# Verify PyTorch + CUDA
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python3 -c "import torchvision; print(f'torchvision {torchvision.__version__}')"

# Install only the packages actually needed for this project
echo "Installing remaining packages..."
uv pip install \
    transformers \
    sentencepiece \
    tiktoken \
    pillow \
    opencv-python-headless \
    tqdm \
    einops \
    huggingface_hub \
    pandas \
    numpy \
    --no-install-recommends \
    2>&1 | tail -10

# Install qwen-vl-utils (depends on transformers)
echo "Installing qwen-vl-utils..."
uv pip install qwen-vl-utils 2>&1 | tail -3

# Final verification
python3 -c "import transformers; print(f'transformers {transformers.__version__}')"
python3 -c "import pandas; print(f'pandas {pandas.__version__}')"
python3 -c "from qwen_vl_utils import process_vision_info; print('qwen_vl_utils OK')"

echo "=== Python packages installed successfully ==="
