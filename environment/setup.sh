#!/bin/bash
# Minimal setup.sh for GeoCoT reproduction.
# All Python packages (torch, transformers, qwen_vl_utils, accelerate) are
# pre-installed in /home/user/pylib. No pip installs needed.

set -e

echo "=== Setting up GeoCoT environment ==="
python3 --version

# Set LD_LIBRARY_PATH so CUDA/NCCL libraries from pylib are found.
export LD_LIBRARY_PATH="/home/user/pylib/torch/lib:/home/user/pylib/nvidia/nccl/lib:/home/user/pylib/nvidia/cuda_runtime/lib:/home/user/pylib/nvidia/cuda_cupti/lib:${LD_LIBRARY_PATH:-}"

# Prepend pylib to PYTHONPATH so Python finds working packages first.
export PYTHONPATH="/home/user/pylib:${PYTHONPATH:-}"

# Remove broken system torch (has wrong NCCL symbols) to prevent import conflicts.
echo "Removing broken system torch..."
python3 -m pip uninstall -y torch 2>/dev/null || true

# Verify all packages load correctly.
echo "Verifying packages..."
python3 -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'transformers {transformers.__version__}')"
python3 -c "from qwen_vl_utils import process_vision_info; print('qwen_vl_utils OK')"
python3 -c "import accelerate; print(f'accelerate {accelerate.__version__}')"
python3 -c "from google import protobuf; print('google.protobuf OK')"

echo "=== Environment setup complete ==="
