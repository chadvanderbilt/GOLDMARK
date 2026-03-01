#!/usr/bin/env bash
#SBATCH -J attn_external_example
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH -t 2-00:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

# Example environment activation (edit for your cluster):
#   module load cuda
#   source ~/miniconda3/etc/profile.d/conda.sh
#   conda activate <env>

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "${REPO_ROOT}"

PYTHONPATH=. python scripts/run_inference_from_plan.py \
  --plan "examples/plans/attn_external_plan_example.csv" \
  --foundation-root "${MIL_DATA_ROOT:-data/foundation_model_training_images}" \
  --device "cuda" \
  --tcga-checkpoints "best,120" \
  --continue-on-error \
  --error-log "logs/attn_external_example_failures.log" \
  --filter-cohort "external" \
  --filter-tumor "BLCA" \
  --filter-target "ERBB2"
