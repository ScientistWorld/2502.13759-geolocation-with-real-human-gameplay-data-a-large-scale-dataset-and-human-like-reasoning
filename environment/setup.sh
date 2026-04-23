#!/bin/bash
# Runtime environment setup for GeoCoT reproduction.
# This script runs at JOB EXECUTION TIME inside the container.

set -e

echo "=== Setting up GeoCoT environment (runtime) ==="

export UV_SYSTEM_PYTHON=1 UV_BREAK_SYSTEM_PACKAGES=1
# The scheduler environment can inject invalid certificate paths from the host.
# Clearing them avoids uv warnings during idempotent runtime installs.
unset SSL_CERT_FILE
unset SSL_CERT_DIR

# Remove conflicting packages in user site-packages that shadow system installs.
pip uninstall -y huggingface_hub 2>/dev/null || true
rm -rf /root/.local/lib/python3.10/site-packages/huggingface_hub* 2>/dev/null || true
rm -rf /root/.local/lib/python3.10/site-packages/transformers* 2>/dev/null || true

# Set HuggingFace cache inside the portable workspace downloads directory.
export HF_HOME="/home/user/data/downloads/hf_cache"
export TRANSFORMERS_CACHE="/home/user/data/downloads/hf_cache"
export HF_HUB_OFFLINE="1"

# Install all required packages in one command (overlay persists them).
echo "Installing required packages..."
uv pip install --system \
    torch torchvision \
    transformers qwen_vl_utils \
    accelerate protobuf pillow numpy pandas huggingface_hub

echo "=== Runtime environment setup complete ==="
