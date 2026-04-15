#!/bin/bash
# Download all data and models needed for this environment.
#
# Estimated total download size: ~85 GB
# Estimated disk usage after extraction: ~85 GB
#
# Downloads:
# - Qwen2.5-VL-7B-Instruct: ~14 GB (VLM model)
# - Qwen2.5-VL-32B-Instruct: ~65 GB (optional, stronger VLM)
# - GeoCLIP-derived evaluation sample: tracked in data/geoclip/ and verified below
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
# 2. Verify tracked evaluation sample
# =============================================================================
echo ""
echo "=== Verifying Tracked Evaluation Sample ==="

if [ ! -f "/home/user/data/geoclip/geoclip.csv" ]; then
    echo "ERROR: data/geoclip/geoclip.csv is missing. The GeoCLIP-derived sample is tracked in git and should be present after cloning this workspace."
    exit 1
fi

if ! find /home/user/data/geoclip -maxdepth 1 -name '*.png' -print -quit | grep -q .; then
    echo "ERROR: data/geoclip contains no PNG images. The tracked evaluation sample is incomplete."
    exit 1
fi

echo "GeoCLIP-derived sample: $(find /home/user/data/geoclip -maxdepth 1 -name '*.png' | wc -l) images"

# =============================================================================
# 3. Verify downloads
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
