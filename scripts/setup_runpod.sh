#!/usr/bin/env bash
set -euo pipefail

TARGET_IMAGE="runpod/pytorch:1.0.7-cu1290-torch291-ubuntu2404"

echo "[setup] Target RunPod image: ${TARGET_IMAGE}"
echo "[setup] Preserving image-provided torch/torchaudio 2.9.1."

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates \
  ffmpeg \
  git \
  libsndfile1 \
  wget
rm -rf /var/lib/apt/lists/*

python scripts/check_env.py --pre-install
python -m pip install --upgrade pip setuptools wheel
python -m pip install -c constraints-runpod.txt -r requirements.txt
python -m pip install --no-deps -e .

mkdir -p /workspace/.cache/huggingface

cat <<'ENV_HINT'

[setup] Recommended environment variables:
  export HF_HOME=/workspace/.cache/huggingface
  export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
  export PYTORCH_ALLOC_CONF=expandable_segments:True

Note: PyTorch 2.9 deprecates PYTORCH_CUDA_ALLOC_CONF in favor of PYTORCH_ALLOC_CONF.

Hardware note for native BF16 full-audio evaluation:
  MiniCPM-o 4.5's official PyTorch deployment guidance requires at least 28 GB VRAM.
  For ~10-minute LongSpeech samples, 40/48 GB+ is the safer choice.

Optional for gated/private mirrors:
  export HF_TOKEN=...

ENV_HINT

python scripts/check_env.py

echo "[setup] Done. Next: python scripts/download_metadata.py"
