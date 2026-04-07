#!/bin/bash
# Install Python packages into /dev/shm/pylib for the GPU container.
#
# This script runs during container build (in the Apptainer overlay).
# All models and datasets are pre-downloaded on the login node.
#
# IMPORTANT: Install into SYSTEM Python (/dev/shm/pylib), not a virtual environment.

set -e

PYPATH="/dev/shm/pylib"
mkdir -p "$PYPATH"
export PYTHONPATH="$PYPATH"

echo "=== Installing Python packages to $PYPATH ==="

# PyTorch 2.6.0 with CUDA 12.4 (required for loading .bin model files - CVE fix)
pip3 install --target "$PYPATH" torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -5 || \
pip3 install --target "$PYPATH" torch torchvision --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -5 || true

# All other packages
pip3 install --target "$PYPATH" \
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
    2>&1 | tail -5 || true

echo "Python packages installed to $PYPATH"
echo "Packages: $(ls $PYPATH | wc -l) items"

echo "=== Setup Complete ==="
