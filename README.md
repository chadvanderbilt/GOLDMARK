# GOLDMARK — Manuscript Reference Pipeline

This repository is in support of the manuscript **GOLDMARK: Governed Outcome-Linked Diagnostic Model Assessment Reference Kit** and is composed of the end-to-end benchmarking pipeline:

1) **Target construction** (TCGA via GDC download + variant labeling / OncoKB annotation)
2) **Tiling** (20x/40x coordinate generation)
3) **Feature extraction** (canonical pathology foundation models + custom encoders) with **QC metadata**
4) **Training** (gated-attention MIL reference head) on **predefined patient-level splits**
5) **Inference + external inference** and **attention export** (best checkpoint + fixed late epoch)

The design goal is reproducibility and clarity: **cross-validation alone is not sufficient**—the pipeline
is built around **reciprocal external testing** (e.g., TCGA→IMPACT and IMPACT→TCGA) under identical
preprocessing and evaluation criteria.

## Start here (real GDC download + end-to-end pipeline)

Downloads a tiny subset of **real TCGA SVS slides** via **GDC** and runs:

`download → tiling → features → training → inference`

This is a real-data run (WSI deps + OpenSlide required). Even with a small subset, expect **multiple GB**
of downloads and non-trivial runtime.

```bash
git clone https://github.com/chadvanderbilt/GOLDMARK.git
cd GOLDMARK

# Put tokens in one place (never commit the filled file).
# GOLDMARK commands will auto-load `configs/secrets.env` if present.
cp configs/secrets.env.example configs/secrets.env
# edit configs/secrets.env (GDC token file path, ONCOKB_TOKEN, HF_TOKEN, ...)

# Option A (recommended on HPC): conda
conda env create -f environment.yml
conda activate goldmark

# Option B: venv (requires Python 3.10+ and native OpenSlide installed on your system)
# python3 -m venv .venv
# source .venv/bin/activate

python -m pip install -r requirements.txt -r requirements-wsi.txt

# Installs gdc-client into bin/ (ignored by git)
python scripts/install_gdc_client.py --dest bin/gdc-client
# Note: some HPC nodes ship older glibc; if gdc-client fails to run, the smoke
# test will fall back to downloading via the GDC API (sufficient for small runs).

# Downloads 2 tumor + 2 normal slides (smallest by file size) and runs the full pipeline on CPU.
python scripts/gdc_smoke_test_tcga.py --project-id TCGA-COAD --per-class 2 --device cpu --force
```

Outputs land in `runs/gdc_smoke_test/` (including `runs/gdc_smoke_test/inference/inference/inference_results.csv`).

## Reciprocal TCGA→IMPACT smoke tests (mutation labels + external inference)

This repo ships two end-to-end smoke tests that exercise the **full mutation-label → MIL training → attention export**
pathway and validate **reciprocal external inference** (TCGA→IMPACT):

1) `scripts/tcga_to_impact_smoke_test.py` (fast, minimal; good for “does anything run?”)
2) `scripts/tcga_luad_kras_cv_to_impact_smoke_test.py` (**recommended**; 10 TCGA slides, 5-fold CV, attention exports, IMPACT external inference)

### Recommended: TCGA-LUAD KRAS (5×70/30 splits) → IMPACT LUAD external inference

What it does:
- Downloads **10 TCGA-LUAD diagnostic** slides via GDC (filters to `-00-DX` unless `--allow-non-dx`)
- Labels each patient as KRAS **positive/negative** from the GDC **Masked Somatic Mutation** MAF
- Builds **5 independent** 70/30 splits with per-split `val` assignments (train/val/test stored as columns)
- Trains for a small number of epochs (default: 10) and writes a `cv_summary.csv`
- Runs **held-out test inference per split** and exports **attention vectors**
- Runs **external inference on IMPACT LUAD** using the best split checkpoint (and links it from all split dirs)

```bash
python scripts/tcga_luad_kras_cv_to_impact_smoke_test.py \
  --run-name gdc_smoke_test_luad_kras_cv_to_impact \
  --device cpu \
  --limit-tiles 64 \
  --epochs 10 \
  --patience 50 \
  --force
```

#### Output layout (important files and where to find them)

All outputs land under `runs/<run-name>/`.

**A) Smoke-test inputs (download + derived labels)**
- `runs/<run-name>/smoke_data/TCGA-LUAD_svs_manifest.tsv` — full GDC SVS manifest for the project
- `runs/<run-name>/smoke_data/TCGA-LUAD_svs_manifest_subset.tsv` — the 10 selected diagnostic SVS rows
- `runs/<run-name>/smoke_data/gdc_download_svs/**/**.svs` — downloaded TCGA SVS files (GDC UUID folders)
- `runs/<run-name>/smoke_data/gdc_download_maf/**/**.maf.gz` — downloaded GDC mutation calls used for labels
- `runs/<run-name>/smoke_data/tcga_tcga-luad_kras_slides.csv` — selected slide list + labels (schema below)
- `runs/<run-name>/smoke_data/impact_external_KRAS_<encoder>.csv` — external IMPACT subset manifest (schema below)

**B) Tiling**
- `runs/<run-name>/tiling/tiles/manifests/<slide_id>_tiles.csv` — per-slide tile coordinate manifest (schema below)

**C) Features + QC**
- `runs/<run-name>/features/<encoder>/features_<slide_id>.pt` — per-slide feature tensor (N tiles × D)
- `runs/<run-name>/features/<encoder>/features_<slide_id>.json` — QC metadata + checksums (schema below)
- If feature extraction fails QC, tensors are renamed:
  - `features_<slide_id>.FAILED_tile_count_mismatch.pt`
  - `features_<slide_id>.FAILED_degenerate_embeddings.pt`

**D) Training (5-fold CV)**
- `runs/<run-name>/training/checkpoints/<GENE>/versioned_split_manifest/<GENE>_all_splits_latest.csv` — the split manifest used for tiling/features/training (schema below)
- `runs/<run-name>/training/checkpoints/classification_report/cv_summary.csv` — per-split best epoch + metrics (schema below)
- `runs/<run-name>/training/checkpoints/split_1_set/checkpoint/checkpoint_best.pt` — best checkpoint for that split (and similarly for split_2_set..split_5_set)

**E) Per-split held-out test inference (attention export + ROC/PR plots)**

Written under each split so split context is self-contained:
- `runs/<run-name>/training/checkpoints/split_1_set/inference/test/inference_results.csv`
- `runs/<run-name>/training/checkpoints/split_1_set/inference/test/attention/<slide_id>_attention.csv`
- `runs/<run-name>/training/checkpoints/split_1_set/inference/test/plots/roc_pr_curves.png`

Note: the training stage also writes probability exports under the same directory (e.g. `probabilities_test_set.csv`).

**F) External inference (IMPACT LUAD)**

External inference is executed **once** using the **best split** checkpoint and written under that split:
- `runs/<run-name>/training/checkpoints/<best_split>/external_inference/IMPACT/inference_results.csv`
- `runs/<run-name>/training/checkpoints/<best_split>/external_inference/IMPACT/attention/<slide_id>_attention.csv`
- `runs/<run-name>/training/checkpoints/<best_split>/external_inference/IMPACT/plots/roc_pr_curves.png`

For convenience, each non-best split directory contains an `external_inference/IMPACT` entry that links to (or points at)
the best-split external inference results.

#### File schemas (column names)

**TCGA slide selection manifest** (`smoke_data/tcga_tcga-luad_kras_slides.csv`)
- `slide_id` (e.g. `TCGA-05-4244-01Z-00-DX1`)
- `slide_path` (downloaded `.svs` path)
- `label_index` (`0`/`1` mutation label for the chosen gene)
- `patient_id` (TCGA case/patient barcode prefix)

**Versioned split manifest** (`training/checkpoints/<GENE>/versioned_split_manifest/<GENE>_all_splits_latest.csv`)
- `slide_path`, `slide_id`, `label_index`
- `target` (legacy column retained for compatibility)
- `split_1_set` … `split_5_set` with values in `{train,val,test}`

**Tile manifest** (`tiling/tiles/manifests/<slide_id>_tiles.csv`)
- `slide_id`
- `tile_id` (stable tile identifier)
- `x`, `y` (level-0 pixel coordinates)
- `level` (OpenSlide level used for extraction / overlays)
- `width`, `height` (tile size in pixels at `level`)
- `tissue_fraction` (fraction of tile kept after tissue mask filtering)
- `tile_path` (optional; blank when only coordinates are generated)

**Feature metadata JSON** (`features/<encoder>/features_<slide_id>.json`)
- `slide_id`, `encoder`
- `tile_manifest`
- `num_tiles`, `num_features`, `feature_dim`
- `embedding_stats` (degenerate/variance checks)
- `feature_sha256`, `feature_bytes`
- `status` in `{ok,failed}` and `failure_reason` when failed

**CV summary** (`training/checkpoints/classification_report/cv_summary.csv`)
- `split` (e.g. `split_1_set`)
- `best_epoch`
- `val_*` and `test_*` metrics, including `*_roc_auc`, `*_accuracy`, `*_precision`, `*_recall`, `*_f1`, `*_balanced_error_rate`

**Inference results** (`.../inference_results.csv`)
- `slide_id`
- `probability` (predicted P(class=1))
- `prediction` (thresholded at 0.5 by default)
- `target` (ground-truth label, from `label_index`)

**Attention exports** (`.../attention/<slide_id>_attention.csv`)
- `slide_id`
- `tile_index` (0-based)
- `tile_id`, `x`, `y`, `level` (when tile manifest metadata is available)
- `attention` (raw attention weight)
- `probability` (slide-level probability repeated for convenience)

**External inference manifest** (`smoke_data/impact_external_<GENE>_<encoder>.csv`)

Minimum required columns for external inference with `InferenceRunner`:
- `slide_id` (string id used to resolve features / label rows)
- `label_index` (0/1 ground-truth label; used as `target_column`)

Strongly recommended columns (enables overlays + clearer provenance):
- `slide_path` (absolute `.svs` path for overlays; optional if you do not generate overlays)
- `feature_path` (absolute `.pt` path; optional if you pass `--feature-dir` and follow `features_<slide_id>.pt` naming)

Optional columns:
- `split` (only needed if you want to run inference on a subset via `InferenceConfig.split_column/split_value`)

Docs:
- `docs/targets.md`
- `docs/pipeline.md`
- SLURM (GPU) example: `examples/slurm/submit_tcga_luad_EGFR_cv_to_impact.sh`

## Repository layout

| Path | What it contains |
| --- | --- |
| `targets/` | Public, token-safe scripts for GDC download + mutation target tables |
| `goldmark/` | Minimal Python package: tiling, feature extraction, training, inference |
| `scripts/` | Manuscript-scale runners (CV scanning, inference plans, attention export) |
| `examples/` | Example manifest headers + plan templates |
| `paper/` | Select manuscript figures/tables for reference |

## Installation

This repo targets **Python 3.10+**.

```bash
python -m pip install -r requirements.txt
```

Notes:
- For **whole-slide image (WSI)** tiling and attention overlays, install the optional dependencies:
  ```bash
  python -m pip install -r requirements-wsi.txt
  ```
  `openslide-python` also requires the native OpenSlide library (platform-specific).
- For **canonical foundation-model encoders** (timm / transformers / HF hub), install:
  ```bash
  python -m pip install -r requirements-encoders.txt
  ```
  Some encoders download weights and/or require gated access; for offline runs, pre-cache weights or use `--custom-encoder`.

## Quickstart (typical flow)

Set a repo root (recommended):

```bash
export REPO_ROOT="$PWD"
```

### 1) Targets (TCGA example)

Generate a GDC manifest (SVS example):

```bash
python -m goldmark gdc-manifest svs \
  --project-id TCGA-COAD \
  --out tcga_coad_svs_manifest.tsv
```

Download data from GDC:

```bash
targets/tcga/gdc_download.sh --manifest tcga_coad_svs_manifest.tsv --out data/gdc_download
# If needed for controlled-access data, set GDC_TOKEN_FILE in configs/secrets.env
# (or pass --token /path/to/gdc_token.txt).
```

Annotate MAF with OncoKB (token via env var):

```bash
# export ONCOKB_TOKEN="..."  # or set in configs/secrets.env
python targets/variants/annotate_maf_oncokb_by_hgvsg.py \
  --maf-glob "data/gdc_download/**/**.maf.gz" \
  --output data/oncokb/oncokb_annotations.csv
```

Summarize a single gene’s patient-level mutation status:

```bash
python targets/variants/summarize_gene_status.py \
  --annotations data/oncokb/oncokb_annotations.csv \
  --gene PTEN \
  --output data/targets/PTEN_patient_labels.csv
```

Build a slide-level manifest by joining SVS paths to patient labels:

```bash
python targets/tcga/build_slide_manifest_from_svs_and_mutations.py \
  --svs-root data/gdc_download \
  --labels data/targets/PTEN_patient_labels.csv \
  --output data/manifests/TCGA_PTEN_slide_manifest.csv
```

Generate the manuscript-style split manifest:

```bash
PYTHONPATH=. python scripts/generate_versioned_split_manifest.py \
  --manifest data/manifests/TCGA_PTEN_slide_manifest.csv \
  --target PTEN \
  --label-column label_index \
  --target-dir data/checkpoints/PTEN
```

See manifest header examples:
- TCGA split manifest schema: `examples/manifests/tcga_split_manifest_header.csv`
- IMPACT manifest schema (header only): `examples/manifests/impact_LUAD_manifest_header.csv`

### 2) Tiling (20x/40x coordinates)

The canonical tiler is `goldmark/tiling/extractor.py`. For convenience, this repo also exposes a
small CLI:

```bash
PYTHONPATH=. python -m goldmark tiling data/manifests/TCGA_PTEN_slide_manifest.csv \
  --output runs \
  --run-name tcga_pten_demo \
  --tile-size 224 \
  --stride 224 \
  --target-mpp 0.5
```

Tile manifests are written under `runs/<run-name>/tiling/tiles/manifests/` as `<slide_id>_tiles.csv`.

### 3) Feature extraction + QC metadata

Feature extraction writes:
- `features_<slide_id>.pt` (per-slide tensor)
- `features_<slide_id>.json` (QC metadata + checksums)

If **tile/feature counts mismatch** or embeddings are **degenerate (low variance)**, the tensor is renamed:
`features_<slide_id>.FAILED_<reason>.pt` and training/inference will treat the slide as missing.

Example:

```bash
PYTHONPATH=. python -m goldmark features data/manifests/TCGA_PTEN_slide_manifest.csv \
  --tile-manifests runs/tcga_pten_demo/tiling/tiles \
  --output runs \
  --run-name tcga_pten_demo \
  --encoder prov-gigapath \
  --device cuda \
  --num-workers 4
```

Optional integrity audit (tile counts vs feature lengths):

```bash
python scripts/check_tile_feature_counts.py \
  --tile-manifest-dir runs/tcga_pten_demo/tiling/tiles/manifests \
  --feature-dir runs/tcga_pten_demo/features/prov-gigapath
```

### 4) Training (reference gated-attention MIL head)

Manuscript-scale training is orchestrated by:
- `scripts/run_training_scan_target.py` (scan encoders + submit missing runs)
- `scripts/train_task_v2.py` (train one target across 5 splits)

### 5) Inference + external inference + attention export

The manuscript pipeline exports attention vectors for **(i) best checkpoint** and **(ii) a fixed late epoch**
to show how checkpoint selection affects results and to support consistent downstream overlays.

See:
- `scripts/run_inference_from_plan.py`
- `scripts/gma_inference_pipeline.py`
- SLURM template: `examples/slurm/submit_attn_impact_BLCA_ERBB2.sh`

## Reference figures/tables

- Workflow: `paper/figures/workflow_schematic.pdf`
- Tile/feature QC: `paper/figures/qc_tile_feature_integrity.pdf`
- Table 1 (counts): `paper/tables/table1_gene_mutation_counts.tex`

## Manuscript tasks (Table 1)

<details>
<summary>Show tumor/target tasks and cohort counts</summary>

| Tumor Code | Tumor Type | Gene Mutation | MSKCC Total | MSKCC Positive | MSKCC Negative | TCGA Total | TCGA Positive | TCGA Negative |
| --- | --- | --- | ---:| ---:| ---:| ---:| ---:| ---:|
| BLCA | Bladder Urothelial Carcinoma | PIK3CA | 2031 | 337 | 1694 | 386 | 77 | 309 |
| BLCA | Bladder Urothelial Carcinoma | FGFR3 | 2031 | 393 | 1638 | 386 | 34 | 352 |
| BLCA | Bladder Urothelial Carcinoma | ERBB2 | 2031 | 207 | 1824 | 386 | 37 | 349 |
| BLCA | Bladder Urothelial Carcinoma | TSC1 | 2031 | 131 | 1900 | 386 | 27 | 359 |
| BLCA | Bladder Urothelial Carcinoma | ERCC2 | 2031 | 141 | 1890 | 386 | 28 | 358 |
| BRCA | Breast Carcinoma | PIK3CA | 2735 | 966 | 1769 | 1000 | 354 | 646 |
| CESC | Cervical Squamous Cell Carcinoma | PIK3CA | 171 | 62 | 109 | 266 | 82 | 184 |
| COAD | Colon Adenocarcinoma | PIK3CA | 2959 | 605 | 2354 | 550 | 130 | 420 |
| COAD | Colon Adenocarcinoma | BRAF | 2959 | 318 | 2641 | 550 | 59 | 491 |
| COAD | Colon Adenocarcinoma | ATM | 2959 | 153 | 2806 | 550 | 63 | 487 |
| COAD | Colon Adenocarcinoma | PTEN | 2959 | 183 | 2776 | 550 | 38 | 512 |
| COAD | Colon Adenocarcinoma | MSI | 2959 | 350 | 2609 | 405 | 70 | 335 |
| GBM | Glioblastoma Multiforme | PTEN | 816 | 263 | 553 | 244 | 77 | 167 |
| UCEC | Endometrial Cancer | PTEN | 2062 | 972 | 1090 | 499 | 319 | 180 |
| UCEC | Endometrial Cancer | PIK3CA | 2062 | 904 | 1158 | 499 | 254 | 245 |
| UCEC | Endometrial Cancer | FBXW7 | 2062 | 279 | 1783 | 499 | 94 | 405 |
| UCEC | Endometrial Cancer | ATM | 2062 | 142 | 1920 | 499 | 73 | 426 |
| UCEC | Endometrial Cancer | POLE | 2062 | 114 | 1948 | 499 | 49 | 450 |
| LGG | Glioma | IDH1 | 449 | 216 | 233 | 491 | 382 | 109 |
| LGG | Glioma | PIK3CA | 449 | 41 | 408 | 491 | 42 | 449 |
| HNSC | Head and Neck Carcinoma | PIK3CA | 485 | 84 | 401 | 431 | 66 | 365 |
| HNSC | Head and Neck Carcinoma | HRAS | 485 | 11 | 474 | 431 | 25 | 406 |
| LUAD | Lung Adenocarcinoma | KRAS | 923 | 273 | 650 | 465 | 171 | 294 |
| LUAD | Lung Adenocarcinoma | EGFR | 923 | 273 | 650 | 465 | 51 | 414 |
| SKCM | Melanoma | BRAF | 886 | 213 | 673 | 432 | 233 | 199 |
| SKCM | Melanoma | NRAS | 886 | 143 | 743 | 432 | 112 | 320 |
| SKCM | Melanoma | PTEN | 886 | 63 | 823 | 432 | 50 | 382 |
| SKCM | Melanoma | MAP2K1 | 886 | 38 | 848 | 432 | 26 | 406 |
| PAAD | Pancreatic Adenocarcinoma | KRAS | 3018 | 2674 | 344 | 181 | 136 | 45 |
| PCPG | Pheochromocytoma/Paraganglioma | HRAS | 18 | 2 | 16 | 176 | 19 | 157 |
| STAD | Stomach Adenocarcinoma | PIK3CA | 742 | 69 | 673 | 374 | 60 | 314 |
| THCA | Thyroid Cancer | BRAF | 920 | 399 | 521 | 495 | 296 | 199 |
| THCA | Thyroid Cancer | NRAS | 920 | 130 | 790 | 495 | 41 | 454 |

</details>

## Launch all manuscript tasks (optional)

This repo includes a convenience launcher that loops over the manuscript task list and calls
`scripts/run_training_scan_target.py` for each tumor/target.

```bash
export MIL_DATA_ROOT="/path/to/foundation_model_training_images"

# dry-run (prints commands)
python scripts/launch_manuscript_tasks.py

# execute (runs training scans)
python scripts/launch_manuscript_tasks.py --execute
```

Note: this wrapper runs commands **sequentially**. On HPC clusters you will typically wrap each printed command
in `sbatch` (or your scheduler of choice).

## Run a full project (no subsampling)

The `scripts/tcga_luad_kras_cv_to_impact_smoke_test.py` runner defaults to a **balanced subsample** (via `--per-class`)
so it can finish quickly. To validate the pipeline end-to-end on a **full TCGA project**, set:

- `--per-class 0` to label and include **all** available cases in the TCGA project (one slide per patient).
- `--impact-per-class 0` to run external inference on **all** labeled IMPACT slides (optional; can be large).

Important:
- This can download **hundreds of SVS files** (many 10s–100s of GB). Use a scratch filesystem.
- By default the TCGA slide filter keeps diagnostic FFPE slides (`-00-DX`). Use `--allow-non-dx` to disable this.
- Foundation-model encoders may require authentication/approval (e.g. gated HF repos). Put tokens in `configs/secrets.env`.

Example (single full task: TCGA-LUAD EGFR → external IMPACT LUAD):

```bash
export RUNS_ROOT="runs"
export RUN_NAME="tcga_luad_egfr_full"

python scripts/tcga_luad_kras_cv_to_impact_smoke_test.py \
  --output "${RUNS_ROOT}" \
  --run-name "${RUN_NAME}" \
  --project-id "TCGA-LUAD" \
  --gene "EGFR" \
  --per-class 0 \
  --impact-per-class 0 \
  --encoder "h-optimus-0" \
  --device "cuda" \
  --limit-tiles 0 \
  --epochs 120
```

HPC:
- See `examples/slurm/submit_tcga_luad_EGFR_cv_to_impact.sh` and set `PER_CLASS=0` (and optionally `IMPACT_PER_CLASS=0`).

## Notes

- This repo intentionally **does not** ship raw WSIs or protected clinical data.
- OncoKB and GDC access are governed by their respective terms; keep tokens out of git history.
- The code is structured for clarity and parity with the manuscript pipeline, not as a polished SDK.
