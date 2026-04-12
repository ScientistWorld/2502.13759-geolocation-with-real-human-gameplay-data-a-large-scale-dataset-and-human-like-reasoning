#!/bin/bash
# Runtime environment setup for GeoCoT reproduction.
# This script runs at JOB EXECUTION TIME (before running inference).
# It configures PYTHONPATH and LD_LIBRARY_PATH to use the bind-mounted pylib
# (which contains CUDA-enabled PyTorch).
#
# NOTE: This is NOT run during container build! The container already has
# CPU-compatible packages installed. This script prioritizes the pylib
# CUDA packages at runtime.

set -e

echo "=== Setting up GeoCoT environment (runtime) ==="

# Set LD_LIBRARY_PATH so CUDA/NCCL libraries from pylib are found.
export LD_LIBRARY_PATH="/home/user/pylib/torch/lib:/home/user/pylib/nvidia/nccl/lib:/home/user/pylib/nvidia/cuda_runtime/lib:/home/user/pylib/nvidia/cuda_cupti/lib:${LD_LIBRARY_PATH:-}"

# Prepend pylib to PYTHONPATH so Python finds CUDA-enabled packages first.
export PYTHONPATH="/home/user/pylib:${PYTHONPATH:-}"

# Set HuggingFace cache to shared location
export HF_HOME="/home/user/shared/models/hf"
export TRANSFORMERS_CACHE="/home/user/shared/models/hf"
export HF_HUB_OFFLINE="1"

# Remove broken system torch if it exists (has wrong NCCL symbols).
# This prevents import conflicts with pylib's CUDA-enabled torch.
echo "Checking for conflicting system torch..."
python3 -m pip uninstall -y torch 2>/dev/null || true

# Fix PyTorch _C extension conflict by renaming to _C_disabled.
# The pylib torch has _C.cpython-312-x86_64-linux-gnu.so which shadows
# system Python's types module when imported from pylib.
if [ -f "/home/user/pylib/torch/_C.cpython-312-x86_64-linux-gnu.so" ]; then
    if [ ! -f "/home/user/pylib/torch/_C_disabled.cpython-312-x86_64-linux-gnu.so" ]; then
        mv /home/user/pylib/torch/_C.cpython-312-x86_64-linux-gnu.so \
           /home/user/pylib/torch/_C_disabled.cpython-312-x86_64-linux-gnu.so
        echo "Fixed: renamed _C.so to _C_disabled.so"
    fi
fi

# Verify PyTorch loads with CUDA
echo "Verifying PyTorch..."
python3 -c "
import sys
sys.path.insert(0, '/home/user/pylib')
import torch
print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo "=== Runtime environment setup complete ==="
