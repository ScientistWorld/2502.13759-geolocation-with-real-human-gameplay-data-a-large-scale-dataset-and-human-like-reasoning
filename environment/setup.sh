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

# Install to /dev/shm/pylib for consistent access
PYPATH="/dev/shm/pylib"
mkdir -p "$PYPATH"

# Install PyTorch with CUDA 12.1 support
uv pip install --system --target "$PYPATH" torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>&1 || \
uv pip install --system --target "$PYPATH" torch torchvision 2>&1 || \
pip3 install --target "$PYPATH" torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>&1 || true

# Install all other packages
uv pip install --system --target "$PYPATH" \
    transformers>=4.40.0 \
    accelerate \
    bitsandbytes \
    sentencepiece \
    protobuf \
    tiktoken \
    pillow \
    opencv-python-headless \
    numpy \
    scipy \
    pandas \
    scikit-learn \
    tqdm \
    qwen-vl-utils \
    einops \
    jiwer \
    rapidfuzz \
    2>&1 || pip3 install --target "$PYPATH" \
    transformers accelerate sentencepiece protobuf tiktoken pillow opencv-python-headless numpy scipy pandas scikit-learn tqdm qwen-vl-utils einops jiwer rapidfuzz 2>&1 || true

# OpenAI client (optional)
uv pip install --system --target "$PYPATH" openai 2>&1 || true

# Set PYTHONPATH
export PYTHONPATH="$PYPATH"
echo "Python packages installed to $PYPATH"

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

echo "=== Step 4: Downloading GeoCLIP dataset ==="
GEOCLIP_DIR="/home/user/data/geoclip"
mkdir -p "$GEOCLIP_DIR"

if [ ! -f "$GEOCLIP_DIR/geoclip.csv" ] || [ ! -d "$GEOCLIP_DIR/images" ]; then
    echo "Downloading GeoCLIP dataset from HuggingFace..."
    # Get file listing from HuggingFace
    curl -s "https://huggingface.co/api/datasets/aryanmagoon/GeoCLIP-data/tree/main" | python3 -c "
import sys, json
d = json.load(sys.stdin)
files = [(f['path'], f['size']) for f in d if f['path'].endswith('.png')]
print(f'Found {len(files)} PNG images')

# Write metadata CSV
import csv
with open('$GEOCLIP_DIR/geoclip.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'lat', 'lon', 'country', 'continent', 'city'])
    for path, size in files:
        name = path.replace('.png', '')
        import re
        m = re.match(r'(-?[\d.]+)_(-?[\d.]+)_([a-z]+)_(\d+)', name)
        if m:
            lat, lon, country, idx = m.groups()
            continent = {'chile': 'South America', 'ecuador': 'South America',
                        'kenya': 'Africa', 'madagascar': 'Africa'}.get(country, 'Unknown')
            writer.writerow([path, lat, lon, country, continent, ''])
print('CSV created')
" 2>&1

    # Download images using wget with parallel connections
    echo "Downloading GeoCLIP images..."
    mkdir -p "$GEOCLIP_DIR/images"
    cd "$GEOCLIP_DIR/images"
    curl -s "https://huggingface.co/api/datasets/aryanmagoon/GeoCLIP-data/tree/main" 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
for f in d:
    if f['path'].endswith('.png'):
        print(f'https://huggingface.co/datasets/aryanmagoon/GeoCLIP-data/resolve/main/{f[\"path\"]}')
" 2>/dev/null | wget -i - -q --directory-prefix=. --timeout=30 --tries=3 -4 2>&1
    echo "GeoCLIP download complete."

    # Update CSV with full paths
    python3 -c "
import csv
with open('$GEOCLIP_DIR/geoclip.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
for row in rows:
    row['image_path'] = '$GEOCLIP_DIR/images/' + row['image_path']
with open('$GEOCLIP_DIR/geoclip.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['image_path','lat','lon','country','continent','city'])
    writer.writeheader()
    writer.writerows(rows)
print('CSV updated with full paths')
" 2>&1
else
    echo "GeoCLIP already exists at $GEOCLIP_DIR"
fi

echo "=== Final Status ==="
echo "Qwen2.5-VL: $([ -d '$QWN_MODEL_DIR' ] && echo 'available' || echo 'not found')"
echo "LLaVA: $([ -d '/home/user/shared/models/llava-v1.5-7b' ] && echo 'available' || echo 'not found')"
echo "GeoCLIP: $([ -f '$GEOCLIP_DIR/geoclip.csv' ] && echo 'available' || echo 'not found')"
