#!/usr/bin/env bash
set -euo pipefail

# Local (Vanderbilt) environment example for TCGA-LUAD EGFR with external inference.
# This script is intentionally not referenced in the public README.

set +u
source /home/vanderbc/.bashrc
set -u
conda activate goldmark

export PROJECT_ID=TCGA-LUAD
export GENE=EGFR
export ENCODER=h-optimus-0
export RUN_NAME=TCGA-LUAD
export RUN_MODE=resume
export PER_CLASS=0
export EXTERNAL_PER_CLASS=0
export TARGET_MPP=0.5
export EXTRA_TARGET_MPP=0.25

# External cohort manifest + root (Vanderbilt layout).
export EXTERNAL_MANIFEST=/data1/vanderbc/foundation_model_training_images/IMPACT/manifests/mutations/final_gene_binary_manifest_latest.csv
export EXTERNAL_ROOT=/data1/vanderbc/foundation_model_training_images/IMPACT

bash /data1/vanderbc/vanderbc/GOLDMARK/scripts/nohup_tcga_cv_to_external.sh
