#!/usr/bin/env bash
#SBATCH -J tcga_cv_to_external
#SBATCH -p preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 3-00:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

# Generic TCGA → external cohort submission (project/gene/encoder are configurable).
# Override any variable below via environment (e.g., PROJECT_ID=TCGA-LUAD GENE=EGFR).

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "${REPO_ROOT}"

export PYTHONPATH=.

PYTHON_BIN="${PYTHON_BIN:-/data1/vanderbc/vanderbc/anaconda3/envs/running_ft/bin/python}"
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

# Recommended: keep caches out of $HOME on HPC.
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

# Ensure tokens from configs/secrets.env are available to downstream tools.
if [[ -f "${REPO_ROOT}/configs/secrets.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/configs/secrets.env"
  set +a
fi

# Hugging Face auth preflight (fail fast).
# Override (not recommended): SKIP_HF_AUTH_CHECK=1
if [[ "${SKIP_HF_AUTH_CHECK:-0}" != "1" ]]; then
  if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: Missing Hugging Face token. Set HUGGINGFACE_HUB_TOKEN or HF_TOKEN in configs/secrets.env." >&2
    echo "Override (not recommended): SKIP_HF_AUTH_CHECK=1" >&2
    exit 2
  fi
fi

# Output configuration
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"

# Run mode:
# - force   : wipe existing run dir and start over (default)
# - resume  : reuse existing run dir + manifests; continue from existing outputs
# - rebuild : preserve smoke_data, rebuild tiling/features/training/inference
RUN_MODE="${RUN_MODE:-force}"
EXTRA_ARGS=()
case "${RUN_MODE}" in
  force)
    EXTRA_ARGS+=(--force)
    ;;
  resume)
    EXTRA_ARGS+=(--resume)
    ;;
  rebuild)
    EXTRA_ARGS+=(--rebuild)
    ;;
  *)
    echo "ERROR: Unknown RUN_MODE='${RUN_MODE}'. Use force|resume|rebuild." >&2
    exit 3
    ;;
esac

# Pipeline knobs (override via env)
PROJECT_ID="${PROJECT_ID:-TCGA-LUAD}"
RUN_NAME="${RUN_NAME:-${PROJECT_ID}}"
GENE="${GENE:-KRAS}"
ENCODER="${ENCODER:-h-optimus-0}"
DEVICE="${DEVICE:-cuda}"
TARGET_MPP="${TARGET_MPP:-0.5}"
EXTRA_TARGET_MPP="${EXTRA_TARGET_MPP:-}"
PER_CLASS="${PER_CLASS:-0}"
EXTERNAL_PER_CLASS="${EXTERNAL_PER_CLASS:-0}"
EXTERNAL_MANIFEST="${EXTERNAL_MANIFEST:-}"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-}"
LIMIT_TILES="${LIMIT_TILES:-0}"
EPOCHS="${EPOCHS:-25}"
PATIENCE="${PATIENCE:-50}"

TILING_ARGS=("--target-mpp" "${TARGET_MPP}")
if [[ -n "${EXTRA_TARGET_MPP}" ]]; then
  TILING_ARGS+=("--extra-target-mpp" "${EXTRA_TARGET_MPP}")
fi
EXTERNAL_ARGS=()
if [[ -n "${EXTERNAL_MANIFEST}" ]]; then
  EXTERNAL_ARGS+=("--external-manifest" "${EXTERNAL_MANIFEST}")
fi
if [[ -n "${EXTERNAL_ROOT}" ]]; then
  EXTERNAL_ARGS+=("--external-root" "${EXTERNAL_ROOT}")
fi
EXTERNAL_ARGS+=("--external-per-class" "${EXTERNAL_PER_CLASS}")
LIMIT_ARGS=()
if [[ "${LIMIT_TILES}" -gt 0 ]]; then
  LIMIT_ARGS+=("--limit-tiles" "${LIMIT_TILES}")
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
  --patience "${PATIENCE}" \
  "${EXTRA_ARGS[@]}"
