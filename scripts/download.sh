#!/bin/bash
# Download all data and models needed for this environment.
#
# Estimated total download size: ~25 GB
# Estimated disk usage after extraction: ~35 GB
#
# Downloads:
# - Qwen2.5-VL-7B-Instruct: ~14 GB (VLM model)
# - Im2GPS3K dataset: ~500 MB (images)
# - Im2GPS dataset: ~100 MB (metadata)
#
# During reproduction, prefer downloading to shared directories
# (shared/datasets/, shared/models/) so other papers can reuse them.

set -e

cd /home/user

echo "=== GeoComp/GeoCoT Environment Download ==="

# Check available disk space
df -h /home/user | tail -1

# =============================================================================
# 1. Download VLM Model (Qwen2.5-VL)
# =============================================================================
echo ""
echo "=== Downloading Qwen2.5-VL-7B-Instruct ==="

MODEL_DIR="/home/user/shared/models/Qwen2.5-VL-7B-Instruct"
if [ -d "$MODEL_DIR" ]; then
    echo "Qwen2.5-VL already exists at $MODEL_DIR, skipping download."
else
    echo "Downloading Qwen2.5-VL-7B-Instruct from HuggingFace..."
    huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False
    echo "Download complete."
fi

# Also try to download via python if huggingface-cli not available
if [ ! -d "$MODEL_DIR" ]; then
    echo "Trying alternative download method..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-VL-7B-Instruct',
    local_dir='$MODEL_DIR',
    local_dir_use_symlinks=False
)
print('Download complete.')
"
fi

# =============================================================================
# 2. Download Im2GPS3K Dataset
# =============================================================================
echo ""
echo "=== Downloading Im2GPS3K Dataset ==="

IM2GPS3K_DIR="/home/user/data/im2gps3k"
mkdir -p "$IM2GPS3K_DIR"

if [ -f "$IM2GPS3K_DIR/im2gps3k.csv" ]; then
    echo "Im2GPS3K metadata already exists, skipping download."
else
    echo "Downloading Im2GPS3K dataset..."

    # Try official source first
    wget -nc -O "$IM2GPS3K_DIR/Im2GPS3K.tar.gz" \
        "https://graphics.stanford.edu/projects/location/Im2GPS3K.tar.gz" \
        2>/dev/null || \
    curl -L -o "$IM2GPS3K_DIR/Im2GPS3K.tar.gz" \
        "https://graphics.stanford.edu/projects/location/Im2GPS3K.tar.gz"

    if [ -f "$IM2GPS3K_DIR/Im2GPS3K.tar.gz" ]; then
        tar -xzf "$IM2GPS3K_DIR/Im2GPS3K.tar.gz" -C "$IM2GPS3K_DIR"
        rm "$IM2GPS3K_DIR/Im2GPS3K.tar.gz"
        echo "Im2GPS3K extracted."
    else
        echo "Warning: Could not download Im2GPS3K. Will use synthetic data for testing."
    fi
fi

# =============================================================================
# 3. Download Im2GPS Dataset
# =============================================================================
echo ""
echo "=== Downloading Im2GPS Dataset ==="

IM2GPS_DIR="/home/user/data/im2gps"
mkdir -p "$IM2GPS_DIR"

if [ -f "$IM2GPS_DIR/im2gps.csv" ]; then
    echo "Im2GPS metadata already exists, skipping download."
else
    wget -nc -O "$IM2GPS_DIR/im2gps.tar.gz" \
        "https://graphics.stanford.edu/projects/location/Im2GPS.tar.gz" \
        2>/dev/null || \
    curl -L -o "$IM2GPS_DIR/im2gps.tar.gz" \
        "https://graphics.stanford.edu/projects/location/Im2GPS.tar.gz"

    if [ -f "$IM2GPS_DIR/im2gps.tar.gz" ]; then
        tar -xzf "$IM2GPS_DIR/im2gps.tar.gz" -C "$IM2GPS_DIR"
        rm "$IM2GPS_DIR/im2gps.tar.gz"
        echo "Im2GPS extracted."
    fi
fi

# =============================================================================
# 4. Verify downloads
# =============================================================================
echo ""
echo "=== Verifying Downloads ==="

if [ -d "$MODEL_DIR" ]; then
    echo "Qwen2.5-VL: $(du -sh $MODEL_DIR | cut -f1)"
fi

if [ -d "$IM2GPS3K_DIR/images" ] || [ -f "$IM2GPS3K_DIR/im2gps3k.csv" ]; then
    echo "Im2GPS3K: $(du -sh $IM2GPS3K_DIR | cut -f1)"
fi

echo ""
echo "=== Download Complete ==="
df -h /home/user | tail -1
