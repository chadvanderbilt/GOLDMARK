#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data1/vanderbc/vanderbc/GOLDMARK}"
cd "${REPO_ROOT}"

# Ensure conda is available in this shell.
if [[ -f /home/vanderbc/.bashrc ]]; then
  set +u
  # shellcheck disable=SC1091
  source /home/vanderbc/.bashrc
  set -u
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found after sourcing /home/vanderbc/.bashrc" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "goldmark"; then
  conda env update -f environment.yml
else
  conda env create -f environment.yml
fi

conda activate goldmark

python -m pip install -r requirements.txt -r requirements-wsi.txt -r requirements-encoders.txt

echo "[info] Conda env 'goldmark' ready."
