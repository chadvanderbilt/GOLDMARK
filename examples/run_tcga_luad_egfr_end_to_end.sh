#!/usr/bin/env bash
set -euo pipefail

# End-to-end TCGA-LUAD → external cohort (EGFR) example.
# This runs in the foreground (not nohup); use RUN_MODE=resume to pick up partial runs.

REPO_ROOT="${REPO_ROOT:-/data1/vanderbc/vanderbc/GOLDMARK}"
cd "${REPO_ROOT}"

# Optional env setup (default on). Disable with SKIP_ENV_SETUP=1.
if [[ "${SKIP_ENV_SETUP:-0}" != "1" ]]; then
  set +u
  # shellcheck disable=SC1091
  source /home/vanderbc/.bashrc
  set -u
  CONDA_ENV="${CONDA_ENV:-goldmark}"
  conda activate "${CONDA_ENV}"
fi

# Load tokens if present.
if [[ -f "${REPO_ROOT}/configs/secrets.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/configs/secrets.env"
  set +a
fi

PROJECT_ID="${PROJECT_ID:-TCGA-LUAD}"
GENE="${GENE:-EGFR}"
ENCODER="${ENCODER:-h-optimus-0}"
DEVICE="${DEVICE:-cuda}"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"
RUN_NAME="${RUN_NAME:-${PROJECT_ID}}"

# 20x + 40x tiling by default.
TARGET_MPP="${TARGET_MPP:-0.5}"
EXTRA_TARGET_MPP="${EXTRA_TARGET_MPP:-0.25}"

PER_CLASS="${PER_CLASS:-0}"
EXTERNAL_PER_CLASS="${EXTERNAL_PER_CLASS:-0}"
EXTERNAL_MANIFEST="${EXTERNAL_MANIFEST:-}"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-}"
LIMIT_TILES="${LIMIT_TILES:-0}"
EPOCHS="${EPOCHS:-25}"
VAL_PER_CLASS="${VAL_PER_CLASS:-0}"
PATIENCE="${PATIENCE:-50}"
RUN_MODE="${RUN_MODE:-force}"  # force|resume|rebuild

EXTRA_ARGS=()
case "${RUN_MODE}" in
  force)   EXTRA_ARGS+=(--force) ;;
  resume)  EXTRA_ARGS+=(--resume) ;;
  rebuild) EXTRA_ARGS+=(--rebuild) ;;
  *) echo "ERROR: Unknown RUN_MODE='${RUN_MODE}'. Use force|resume|rebuild." >&2; exit 3 ;;
esac

TILING_ARGS=(--target-mpp "${TARGET_MPP}")
if [[ -n "${EXTRA_TARGET_MPP}" ]]; then
  TILING_ARGS+=(--extra-target-mpp "${EXTRA_TARGET_MPP}")
fi
EXTERNAL_ARGS=()
if [[ -n "${EXTERNAL_MANIFEST}" ]]; then
  EXTERNAL_ARGS+=(--external-manifest "${EXTERNAL_MANIFEST}")
fi
if [[ -n "${EXTERNAL_ROOT}" ]]; then
  EXTERNAL_ARGS+=(--external-root "${EXTERNAL_ROOT}")
fi
EXTERNAL_ARGS+=(--external-per-class "${EXTERNAL_PER_CLASS}")
LIMIT_ARGS=()
if [[ "${LIMIT_TILES}" -gt 0 ]]; then
  LIMIT_ARGS+=(--limit-tiles "${LIMIT_TILES}")
fi

PYTHON_BIN="${PYTHON_BIN:-/data1/vanderbc/vanderbc/anaconda3/envs/goldmark/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "ERROR: No python interpreter found. Set PYTHON_BIN or load a conda env." >&2
    exit 1
  fi
fi

"${PYTHON_BIN}" scripts/tcga_cv_to_external_full_run.py \
  --output "${RUNS_ROOT}" \
  --run-name "${RUN_NAME}" \
  --project-id "${PROJECT_ID}" \
  --gene "${GENE}" \
  --per-class "${PER_CLASS}" \
  "${EXTERNAL_ARGS[@]}" \
  --encoder "${ENCODER}" \
  --device "${DEVICE}" \
  "${TILING_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  --epochs "${EPOCHS}" \
  --val-per-class "${VAL_PER_CLASS}" \
  --patience "${PATIENCE}" \
  "${EXTRA_ARGS[@]}"
