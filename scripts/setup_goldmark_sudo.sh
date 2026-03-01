#!/usr/bin/env bash
set -euo pipefail

# System-wide setup for users with sudo access (Ubuntu/Debian).
# Installs OpenSlide + Python build deps, then creates a venv and installs requirements.

REPO_ROOT="${REPO_ROOT:-/data1/vanderbc/vanderbc/GOLDMARK}"
cd "${REPO_ROOT}"

sudo apt-get update
sudo apt-get install -y \
  python3-venv \
  python3-dev \
  build-essential \
  git \
  openslide-tools \
  libopenslide0 \
  libopenslide-dev \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxrender1 \
  libxext6

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-wsi.txt -r requirements-encoders.txt

echo "[info] venv ready at ${REPO_ROOT}/.venv"
