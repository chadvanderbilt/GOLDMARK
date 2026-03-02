#!/usr/bin/env bash
#SBATCH -J luad_egfr_tcga_to_external
#SBATCH -p preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 3-00:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

# This SLURM example runs the end-to-end TCGA→external pipeline *from scratch*:
#   GDC download → mutation labels (EGFR) → tiling → features → 5-split CV training → per-split test inference (attention) → external cohort inference
#
# Notes:
# - Tokens are loaded from `configs/secrets.env` by default.
# - This will download real TCGA SVS files. Even a modest `--per-class` can be many GB.
# - Use `--allow-non-dx` ONLY if you want non-diagnostic slides; default filters to `-00-DX`.
#
# Example environment activation (edit for your cluster):
#   module load cuda
#   source ~/miniconda3/etc/profile.d/conda.sh
#   conda activate <cuda-enabled-env>

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "${REPO_ROOT}"

export PYTHONPATH=.

# Recommended: keep caches out of $HOME on HPC.
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

# Output configuration
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"
RUN_NAME="${RUN_NAME:-tcga_luad_egfr_cv_to_external_full}"

# Pipeline knobs
ENCODER="${ENCODER:-h-optimus-0}"
GENE="${GENE:-EGFR}"

# Balanced slide subset size:
# - per-class=5 -> 10 total (smoke-test scale)
# - increase for a larger run (must be <= number of positives available in the cohort)
# - per-class=0 labels *all* available cases (full cohort; very large)
PER_CLASS="${PER_CLASS:-5}"

# External cohort inference selection:
# - external-per-class=5 -> 10 total (smoke-test scale)
# - external-per-class=0 runs external inference on *all* labeled external cases (very large)
EXTERNAL_PER_CLASS="${EXTERNAL_PER_CLASS:-5}"
EXTERNAL_MANIFEST="${EXTERNAL_MANIFEST:-}"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-}"

# limit-tiles:
# - 64 is fast but NOT “full slide”
# - 0 disables limiting (tiles the full tissue mask; can be very large)
LIMIT_TILES="${LIMIT_TILES:-0}"

EPOCHS="${EPOCHS:-25}"
PATIENCE="${PATIENCE:-50}"

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

python scripts/tcga_cv_to_external_full_run.py \
  --output "${RUNS_ROOT}" \
  --run-name "${RUN_NAME}" \
  --project-id "TCGA-LUAD" \
  --gene "${GENE}" \
  --per-class "${PER_CLASS}" \
  "${EXTERNAL_ARGS[@]}" \
  --encoder "${ENCODER}" \
  --device "cuda" \
  "${LIMIT_ARGS[@]}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --force
