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
  conda activate running_ft
fi

# Output configuration
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"
RUN_NAME="${RUN_NAME:-tcga_cv_to_impact}"

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
GENE="${GENE:-KRAS}"
ENCODER="${ENCODER:-h-optimus-0}"
DEVICE="${DEVICE:-cuda}"
PER_CLASS="${PER_CLASS:-0}"
IMPACT_PER_CLASS="${IMPACT_PER_CLASS:-0}"
LIMIT_TILES="${LIMIT_TILES:-0}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-50}"

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

STAMP="$(date +%Y%m%dT%H%M%S)"
LOG_FILE="${RUNS_ROOT}/${RUN_NAME}_nohup_${STAMP}.out"
PID_FILE="${RUNS_ROOT}/${RUN_NAME}_nohup.pid"

mkdir -p "${RUNS_ROOT}"

PYTHONUNBUFFERED=1 nohup "${PYTHON_BIN}" scripts/tcga_luad_kras_cv_to_impact_smoke_test.py \
  --output "${RUNS_ROOT}" \
  --run-name "${RUN_NAME}" \
  --project-id "${PROJECT_ID}" \
  --gene "${GENE}" \
  --per-class "${PER_CLASS}" \
  --impact-per-class "${IMPACT_PER_CLASS}" \
  --encoder "${ENCODER}" \
  --device "${DEVICE}" \
  --limit-tiles "${LIMIT_TILES}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  "${EXTRA_ARGS[@]}" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "[info] Started PID=$(cat "${PID_FILE}")"
echo "[info] Log: ${LOG_FILE}"
