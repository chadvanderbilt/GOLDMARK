#!/usr/bin/env bash
set -euo pipefail

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
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-50}"

TILING_ARGS=("--target-mpp" "${TARGET_MPP}")
if [[ -n "${EXTRA_TARGET_MPP}" ]]; then
  TILING_ARGS+=("--extra-target-mpp" "${EXTRA_TARGET_MPP}")
fi
LIMIT_ARGS=()
if [[ "${LIMIT_TILES}" -gt 0 ]]; then
  LIMIT_ARGS+=(--limit-tiles "${LIMIT_TILES}")
fi
EXTERNAL_ARGS=()
if [[ -n "${EXTERNAL_MANIFEST}" ]]; then
  EXTERNAL_ARGS+=(--external-manifest "${EXTERNAL_MANIFEST}")
fi
if [[ -n "${EXTERNAL_ROOT}" ]]; then
  EXTERNAL_ARGS+=(--external-root "${EXTERNAL_ROOT}")
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

RUN_DIR="${RUNS_ROOT}/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
LOG_FILE="${LOG_DIR}/nohup.out"
PID_FILE="${LOG_DIR}/nohup.pid"

mkdir -p "${LOG_DIR}"

if [[ -f "${LOG_FILE}" ]]; then
  STAMP="$(date +%Y%m%dT%H%M%S)"
  mv "${LOG_FILE}" "${LOG_DIR}/nohup.prev_${STAMP}.out"
fi

PYTHONUNBUFFERED=1 nohup "${PYTHON_BIN}" scripts/tcga_cv_to_external_full_run.py \
  --output "${RUNS_ROOT}" \
  --run-name "${RUN_NAME}" \
  --project-id "${PROJECT_ID}" \
  --gene "${GENE}" \
  --per-class "${PER_CLASS}" \
  --external-per-class "${EXTERNAL_PER_CLASS}" \
  --encoder "${ENCODER}" \
  --device "${DEVICE}" \
  "${TILING_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  "${EXTERNAL_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "[info] Started PID=$(cat "${PID_FILE}")"
echo "[info] Run dir: ${RUN_DIR}"
echo "[info] Log: ${LOG_FILE}"
echo "[info] Tail: tail -f \"${LOG_FILE}\""
