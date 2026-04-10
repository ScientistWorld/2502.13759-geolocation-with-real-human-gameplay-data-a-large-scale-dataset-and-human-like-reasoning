#!/bin/bash
# Install Python packages into the container.
#
# This script runs during container build (in the Apptainer overlay).
# All models and datasets are pre-downloaded on the login node.
#
# IMPORTANT: Install into SYSTEM Python, not a virtual environment.

set -e

echo "=== Installing Python packages ==="

# Install uv if not present
which uv || pip3 install uv

# Use uv for fast installation
export UV_SYSTEM_PYTHON=1
export UV_BREAK_SYSTEM_PACKAGES=1

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3 || \
uv pip install torch torchvision 2>&1 | tail -3

# Install all other packages
echo "Installing remaining packages..."
uv pip install \
    transformers>=4.40.0 \
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
    2>&1 | tail -5

# Verify installation
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'transformers {transformers.__version__}')"
python3 -c "import pandas; print(f'pandas {pandas.__version__}')"

echo "=== Python packages installed successfully ==="
