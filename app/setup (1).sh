#!/usr/bin/env bash
# setup.sh – run ONCE to prepare the environment before launching app.py
set -e

echo "=== 1. Installing Python deps ==="
pip install -r requirements.txt

echo ""
echo "=== 2. Cloning GenD ==="
mkdir -p repos
if [ ! -d "repos/GenD/.git" ]; then
  git clone https://github.com/yermandy/GenD repos/GenD
else
  echo "GenD already cloned – skipping."
fi

echo ""
echo "=== 3. Cloning NSG-VD ==="
if [ ! -d "repos/NSG-VD/.git" ]; then
  git clone https://github.com/ZSHsh98/NSG-VD repos/NSG-VD
else
  echo "NSG-VD already cloned – skipping."
fi

# Install any extra deps listed by each repo
echo ""
echo "=== 4. Installing repo-specific deps ==="
pip install -r repos/NSG-VD/requirements.txt || true

echo ""
echo "=== 5. Downloading NSG-VD diffusion backbone ==="
mkdir -p weights
DIFF_PT="weights/256x256_diffusion_uncond.pt"
if [ ! -f "$DIFF_PT" ]; then
  echo "Fetching 256x256_diffusion_uncond.pt from OpenAI …"
  wget -q -O "$DIFF_PT" \
    "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
  echo "Done."
else
  echo "Diffusion backbone already present – skipping."
fi

echo ""
echo "=== 6. Creating reference_bank/ directory ==="
mkdir -p reference_bank
echo ">>> Place 50–100 real (non-AI) .mp4 videos in ./reference_bank/ <<<"
echo "    NSG-VD computes MMD between your test videos and this bank."

echo ""
echo "=== Setup complete. Launch with: python app.py ==="
