#!/usr/bin/env bash
#SBATCH -J luad_egfr_tcga_to_impact
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH -t 3-00:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

# This SLURM example runs the end-to-end TCGA→IMPACT pipeline *from scratch*:
#   GDC download → mutation labels (EGFR) → tiling → features → 5-split CV training → per-split test inference (attention) → external IMPACT inference
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
RUN_NAME="${RUN_NAME:-tcga_luad_egfr_cv_to_impact_full}"

# Pipeline knobs
ENCODER="${ENCODER:-h-optimus-0}"
GENE="${GENE:-EGFR}"

# Balanced slide subset size:
# - per-class=5 -> 10 total (smoke-test scale)
# - increase for a larger run (must be <= number of positives available in the cohort)
PER_CLASS="${PER_CLASS:-5}"

# limit-tiles:
# - 64 is fast but NOT “full slide”
# - 0 disables limiting (tiles the full tissue mask; can be very large)
LIMIT_TILES="${LIMIT_TILES:-0}"

EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-50}"

python scripts/tcga_luad_kras_cv_to_impact_smoke_test.py \
  --output "${RUNS_ROOT}" \
  --run-name "${RUN_NAME}" \
  --project-id "TCGA-LUAD" \
  --gene "${GENE}" \
  --per-class "${PER_CLASS}" \
  --encoder "${ENCODER}" \
  --device "cuda" \
  --limit-tiles "${LIMIT_TILES}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --force
