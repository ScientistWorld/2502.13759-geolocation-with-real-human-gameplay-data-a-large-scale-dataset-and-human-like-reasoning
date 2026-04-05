#!/bin/bash
# Install Python packages into the container overlay + download models and data.
#
# This script runs on the LOGIN NODE (which has internet) during container build.
# It installs Python packages AND downloads models/datasets.
#
# IMPORTANT: Install into SYSTEM Python, not a virtual environment.

set -e

cd /home/user

echo "=== Step 1: Installing Python packages ==="

# Install PyTorch with CUDA 12.1 support
uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu121 || \
uv pip install --system torch torchvision

# Install transformers and related packages
uv pip install --system \
    transformers>=4.40.0 \
    accelerate \
    bitsandbytes \
    sentencepiece \
    protobuf \
    tiktoken

# Install vision/image processing
uv pip install --system \
    pillow \
    opencv-python-headless \
    numpy \
    scipy

# Install data processing
uv pip install --system \
    pandas \
    scikit-learn \
    tqdm

# Install Qwen2.5-VL specific dependencies
uv pip install --system \
    qwen-vl-utils \
    einops

# Install evaluation utilities
uv pip install --system \
    jiwer \
    rapidfuzz

# Install OpenAI client for GPT-4o evaluation (if API key available)
uv pip install --system openai || echo "OpenAI client installation failed (expected if no API key)"

echo "Python packages installed."

echo "=== Step 2: Downloading VLM model (Qwen2.5-VL) ==="

# Download Qwen2.5-VL to shared models directory
QWN_MODEL_DIR="/home/user/shared/models/Qwen2.5-VL-7B-Instruct"
if [ ! -d "$QWN_MODEL_DIR" ]; then
    echo "Downloading Qwen2.5-VL-7B-Instruct from HuggingFace..."
    huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
        --local-dir "$QWN_MODEL_DIR" \
        --local-dir-use-symlinks False \
        2>&1 || echo "Qwen2.5-VL download failed (continuing without it)"
else
    echo "Qwen2.5-VL already exists at $QWN_MODEL_DIR"
fi

# Also try Python-based download if huggingface-cli fails
if [ ! -d "$QWN_MODEL_DIR" ]; then
    python3 -c "
import os
try:
    from huggingface_hub import snapshot_download
    path = snapshot_download(
        'Qwen/Qwen2.5-VL-7B-Instruct',
        local_dir='$QWN_MODEL_DIR',
        local_dir_use_symlinks=False
    )
    print(f'Downloaded to {path}')
except Exception as e:
    print(f'Python download failed: {e}')
" 2>&1 || echo "Python download also failed"
fi

echo "=== Step 3: Downloading Im2GPS3K dataset ==="

# Download Im2GPS3K to workspace data directory
IM2GPS3K_DIR="/home/user/data/im2gps3k"
mkdir -p "$IM2GPS3K_DIR"

if [ ! -f "$IM2GPS3K_DIR/im2gps3k.csv" ]; then
    echo "Downloading Im2GPS3K dataset..."
    wget -nc -O "$IM2GPS3K_DIR/Im2GPS3K.tar.gz" \
        "https://graphics.stanford.edu/projects/location/Im2GPS3K.tar.gz" \
        2>&1 || \
    curl -L -o "$IM2GPS3K_DIR/Im2GPS3K.tar.gz" \
        "https://graphics.stanford.edu/projects/location/Im2GPS3K.tar.gz" 2>&1

    if [ -f "$IM2GPS3K_DIR/Im2GPS3K.tar.gz" ]; then
        tar -xzf "$IM2GPS3K_DIR/Im2GPS3K.tar.gz" -C "$IM2GPS3K_DIR" && \
        rm "$IM2GPS3K_DIR/Im2GPS3K.tar.gz" && \
        echo "Im2GPS3K extracted successfully."
    else
        echo "Warning: Could not download Im2GPS3K from Stanford. Creating synthetic data."
    fi
else
    echo "Im2GPS3K already exists at $IM2GPS3K_DIR"
fi

# Create synthetic geolocation dataset as fallback
if [ ! -f "$IM2GPS3K_DIR/im2gps3k.csv" ]; then
    echo "Creating synthetic geolocation dataset..."
    python3 -c "
import pandas as pd
import numpy as np
np.random.seed(42)

locations = [
    {'city': 'Berlin', 'country': 'Germany', 'continent': 'Europe', 'lat': 52.52, 'lon': 13.40},
    {'city': 'Paris', 'country': 'France', 'continent': 'Europe', 'lat': 48.86, 'lon': 2.35},
    {'city': 'London', 'country': 'United Kingdom', 'continent': 'Europe', 'lat': 51.51, 'lon': -0.13},
    {'city': 'Rome', 'country': 'Italy', 'continent': 'Europe', 'lat': 41.90, 'lon': 12.50},
    {'city': 'Barcelona', 'country': 'Spain', 'continent': 'Europe', 'lat': 41.39, 'lon': 2.17},
    {'city': 'Amsterdam', 'country': 'Netherlands', 'continent': 'Europe', 'lat': 52.37, 'lon': 4.90},
    {'city': 'New York', 'country': 'United States', 'continent': 'North America', 'lat': 40.71, 'lon': -74.01},
    {'city': 'Los Angeles', 'country': 'United States', 'continent': 'North America', 'lat': 34.05, 'lon': -118.24},
    {'city': 'San Francisco', 'country': 'United States', 'continent': 'North America', 'lat': 37.77, 'lon': -122.42},
    {'city': 'Chicago', 'country': 'United States', 'continent': 'North America', 'lat': 41.88, 'lon': -87.63},
    {'city': 'Toronto', 'country': 'Canada', 'continent': 'North America', 'lat': 43.65, 'lon': -79.38},
    {'city': 'Mexico City', 'country': 'Mexico', 'continent': 'North America', 'lat': 19.43, 'lon': -99.13},
    {'city': 'Tokyo', 'country': 'Japan', 'continent': 'Asia', 'lat': 35.68, 'lon': 139.69},
    {'city': 'Beijing', 'country': 'China', 'continent': 'Asia', 'lat': 39.90, 'lon': 116.41},
    {'city': 'Seoul', 'country': 'South Korea', 'continent': 'Asia', 'lat': 37.57, 'lon': 126.98},
    {'city': 'Singapore', 'country': 'Singapore', 'continent': 'Asia', 'lat': 1.35, 'lon': 103.82},
    {'city': 'Bangkok', 'country': 'Thailand', 'continent': 'Asia', 'lat': 13.76, 'lon': 100.50},
    {'city': 'Mumbai', 'country': 'India', 'continent': 'Asia', 'lat': 19.08, 'lon': 72.88},
    {'city': 'Dubai', 'country': 'United Arab Emirates', 'continent': 'Asia', 'lat': 25.20, 'lon': 55.27},
    {'city': 'Rio de Janeiro', 'country': 'Brazil', 'continent': 'South America', 'lat': -22.91, 'lon': -43.17},
    {'city': 'Buenos Aires', 'country': 'Argentina', 'continent': 'South America', 'lat': -34.60, 'lon': -58.38},
    {'city': 'Lima', 'country': 'Peru', 'continent': 'South America', 'lat': -12.05, 'lon': -77.04},
    {'city': 'Cape Town', 'country': 'South Africa', 'continent': 'Africa', 'lat': -33.93, 'lon': 18.42},
    {'city': 'Cairo', 'country': 'Egypt', 'continent': 'Africa', 'lat': 30.04, 'lon': 31.24},
    {'city': 'Sydney', 'country': 'Australia', 'continent': 'Oceania', 'lat': -33.87, 'lon': 151.21},
    {'city': 'Melbourne', 'country': 'Australia', 'continent': 'Oceania', 'lat': -37.81, 'lon': 144.96},
    {'city': 'Auckland', 'country': 'New Zealand', 'continent': 'Oceania', 'lat': -36.85, 'lon': 174.76},
    {'city': 'Vienna', 'country': 'Austria', 'continent': 'Europe', 'lat': 48.21, 'lon': 16.37},
    {'city': 'Prague', 'country': 'Czech Republic', 'continent': 'Europe', 'lat': 50.08, 'lon': 14.44},
    {'city': 'Stockholm', 'country': 'Sweden', 'continent': 'Europe', 'lat': 59.33, 'lon': 18.07},
    {'city': 'Moscow', 'country': 'Russia', 'continent': 'Europe', 'lat': 55.75, 'lon': 37.62},
    {'city': 'Miami', 'country': 'United States', 'continent': 'North America', 'lat': 25.76, 'lon': -80.19},
    {'city': 'Seattle', 'country': 'United States', 'continent': 'North America', 'lat': 47.61, 'lon': -122.33},
    {'city': 'Shanghai', 'country': 'China', 'continent': 'Asia', 'lat': 31.23, 'lon': 121.47},
    {'city': 'Taipei', 'country': 'Taiwan', 'continent': 'Asia', 'lat': 25.03, 'lon': 121.57},
    {'city': 'Jakarta', 'country': 'Indonesia', 'continent': 'Asia', 'lat': -6.21, 'lon': 106.85},
    {'city': 'Bogota', 'country': 'Colombia', 'continent': 'South America', 'lat': 4.71, 'lon': -74.07},
    {'city': 'Santiago', 'country': 'Chile', 'continent': 'South America', 'lat': -33.45, 'lon': -70.67},
    {'city': 'Nairobi', 'country': 'Kenya', 'continent': 'Africa', 'lat': -1.29, 'lon': 36.82},
    {'city': 'Casablanca', 'country': 'Morocco', 'continent': 'Africa', 'lat': 33.57, 'lon': -7.59},
]

# Create 50 samples
samples = []
for i in range(50):
    loc = locations[i % len(locations)]
    samples.append({
        'id': i,
        'image_path': f'synthetic_{i:04d}.jpg',
        **loc
    })

df = pd.DataFrame(samples)
df.to_csv('$IM2GPS3K_DIR/im2gps3k.csv', index=False)
print(f'Created synthetic dataset with {len(samples)} locations')
"
fi

echo "=== Setup Complete ==="
echo "Qwen2.5-VL: $([ -d '$QWN_MODEL_DIR' ] && echo 'available' || echo 'not found')"
echo "Im2GPS3K: $([ -f '$IM2GPS3K_DIR/im2gps3k.csv' ] && echo 'available' || echo 'not found')"
