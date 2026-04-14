#!/bin/bash
# Download all data and models needed for this environment.
#
# Estimated total download size: ~85 GB
# Estimated disk usage after extraction: ~85 GB
#
# Downloads:
# - Qwen2.5-VL-7B-Instruct: ~14 GB (VLM model)
# - Qwen2.5-VL-32B-Instruct: ~65 GB (optional, stronger VLM)
# - GeoCLIP dataset: already in data/geoclip/ (included in repo)
#
# Must be run on a node with internet access before compute jobs.

set -e

cd /home/user

echo "=== GeoCoT Environment Download ==="

df -h /home/user | tail -1

# =============================================================================
# 1. Download VLM Model
# =============================================================================
echo ""
echo "=== Downloading Qwen2.5-VL Models ==="

# Try 32B model first (much stronger)
MODEL_32B_DIR="/home/user/data/downloads/Qwen2.5-VL-32B-Instruct"
if [ -d "$MODEL_32B_DIR" ] && [ -f "$MODEL_32B_DIR/config.json" ]; then
    echo "Qwen2.5-VL-32B-Instruct already exists, skipping."
else
    echo "Downloading Qwen2.5-VL-32B-Instruct (~65 GB)..."
    mkdir -p "$MODEL_32B_DIR"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-VL-32B-Instruct',
    local_dir='$MODEL_32B_DIR',
    local_dir_use_symlinks=False
)
print('32B model download complete.')
" || echo "Warning: 32B model download failed. Will use 7B model."
fi

# 7B model as fallback
MODEL_7B_DIR="/home/user/data/downloads/Qwen2.5-VL-7B-Instruct"
if [ -d "$MODEL_7B_DIR" ] && [ -f "$MODEL_7B_DIR/config.json" ]; then
    echo "Qwen2.5-VL-7B-Instruct already exists, skipping."
else
    echo "Downloading Qwen2.5-VL-7B-Instruct (~14 GB)..."
    mkdir -p "$MODEL_7B_DIR"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-VL-7B-Instruct',
    local_dir='$MODEL_7B_DIR',
    local_dir_use_symlinks=False
)
print('7B model download complete.')
"
fi

# =============================================================================
# 2. Verify downloads
# =============================================================================
echo ""
echo "=== Verifying Downloads ==="

for d in "$MODEL_32B_DIR" "$MODEL_7B_DIR"; do
    if [ -d "$d" ] && [ -f "$d/config.json" ]; then
        echo "$(basename $d): $(du -sh $d | cut -f1)"
    fi
done

echo ""
echo "=== Download Complete ==="
df -h /home/user | tail -1
