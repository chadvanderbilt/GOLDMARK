# GOLDMARK — Manuscript Reference Pipeline

This repository is in support of the manuscript **GOLDMARK: Governed Outcome-Linked Diagnostic Model Assessment Reference Kit** and is composed of the end-to-end benchmarking pipeline:

1) **Target construction** (TCGA via GDC download + variant labeling / OncoKB annotation)
2) **Tiling** (20x/40x coordinate generation)
3) **Feature extraction** (canonical pathology foundation models + custom encoders) with **QC metadata**
4) **Training** (gated-attention MIL reference head) on **predefined patient-level splits**
5) **Inference + external inference** and **attention export** (best checkpoint + fixed late epoch)

The design goal is reproducibility and clarity: **cross-validation alone is not sufficient**—the pipeline
is built around **reciprocal external testing** (e.g., TCGA→external and external→TCGA) under identical
preprocessing and evaluation criteria.

## Start here (real GDC download + end-to-end pipeline)

Downloads and labels **real TCGA SVS slides** via **GDC** and runs:

`download → tiling → features → training → inference`

This is a real-data run (WSI deps + OpenSlide required). Expect **multiple TB**
of downloads and non-trivial runtime.

**Preflight checklist (avoid the common failure modes)**
- Most encoders are hosted on Hugging Face; **approval + a valid HF token are required to run this pipeline**. Without access you will hit `401 Unauthorized` during feature extraction.
- Ensure `configs/secrets.env` exists and is **sourced for non-interactive runs** (SLURM/nohup). Example: `set -a; source configs/secrets.env; set +a`
- If you use `h-optimus-0`, your `HUGGINGFACE_HUB_TOKEN` (or `HF_TOKEN`) must have **explicit access** to `bioptimus/H-optimus-0`. A `401 Unauthorized` means the token is missing or lacks access.
- `gdc-client` requires newer glibc on some clusters. If it fails, GOLDMARK falls back to the GDC API (slower). For full cohorts, prefer a compatible `gdc-client` or a newer node.
- Some clusters do **not** provide `python`. Use `python3` or set `PYTHON_BIN` in the SLURM script.
- The LUAD KRAS end-to-end run defaults to the **full cohort** (`--per-class 0`). Set `--per-class` to a small value for a quick sanity check.

```bash
git clone https://github.com/chadvanderbilt/GOLDMARK.git
cd GOLDMARK

# Put tokens in one place (never commit the filled file).
# GOLDMARK commands will auto-load `configs/secrets.env` if present.
cp configs/secrets.env.example configs/secrets.env
# edit configs/secrets.env (GDC token file path, ONCOKB_TOKEN, HF_TOKEN, ...)

# Option A (recommended on HPC): conda
source /home/vanderbc/.bashrc
conda env create -f environment.yml
conda activate goldmark
python -m pip install -r requirements.txt -r requirements-wsi.txt -r requirements-encoders.txt

# Option B: venv (requires Python 3.10+ and native OpenSlide installed on your system)
# python3 -m venv .venv
# source .venv/bin/activate

python -m pip install -r requirements.txt -r requirements-wsi.txt

# Installs gdc-client into bin/ (ignored by git)
python scripts/install_gdc_client.py --dest bin/gdc-client
# Note: some HPC nodes ship older glibc; if gdc-client fails to run, the pipeline
# will fall back to downloading via the GDC API (sufficient for small runs).

# Downloads 2 tumor + 2 normal slides (smallest by file size) and runs the full pipeline on CPU.
python scripts/gdc_smoke_test_tcga.py --project-id TCGA-COAD --per-class 2 --device cpu --force
```

Outputs land in `runs/gdc_smoke_test/` (including `runs/gdc_smoke_test/inference/inference/inference_results.csv`).

## Reciprocal TCGA→external end-to-end runs (mutation labels + external inference)

This repo ships two end-to-end scripts that exercise the **full mutation-label → MIL training → attention export**
pathway and validate **reciprocal external inference** (TCGA→external):

1) `scripts/tcga_to_external_smoke_test.py` (fast, minimal; good for “does anything run?”)
2) `scripts/tcga_cv_to_external_full_run.py` (**full run by default**; 5-fold CV, attention exports, external inference)

### Recommended: TCGA-LUAD KRAS (5×70/30 splits) → external LUAD inference

What it does:
- Downloads **TCGA-LUAD diagnostic** slides via GDC (filters to `-00-DX` unless `--allow-non-dx`)
- **Only downloads slides with targets**: the script labels cases using MAFs, then builds a subset manifest and downloads only those labeled slides.
- Labels each patient as KRAS **positive/negative** from the GDC **Masked Somatic Mutation** MAF
- Builds **5 independent** 70/30 splits with per-split `val` assignments (train/val/test stored as columns)
- Trains for a full cohort by default and writes a `cv_summary.csv`
- Runs **held-out test inference per split** and exports **attention vectors**
- Runs **external inference on the external LUAD cohort** using the best split checkpoint (and links it from all split dirs)

```bash
python scripts/tcga_cv_to_external_full_run.py \
  --run-name TCGA-LUAD \
  --per-class 0 \
  --device cuda \
  --epochs 10 \
  --patience 50 \
  --force
```

### Generic SLURM + nohup launchers (configurable project/gene/encoder)

For GitHub users, we provide **generic** launchers that accept options via environment variables.
These are not hard-coded to a specific project or gene.

<details>
<summary><strong>SLURM (GPU example)</strong> — <code>examples/slurm/submit_tcga_cv_to_external.sh</code></summary>

```bash
# Example: TCGA-LUAD / EGFR / h-optimus-0, resume in-place
PROJECT_ID=TCGA-LUAD \
GENE=EGFR \
ENCODER=h-optimus-0 \
RUN_MODE=resume \
sbatch examples/slurm/submit_tcga_cv_to_external.sh
```

<strong>SLURM submission example (LUAD EGFR + external inference)</strong> — <code>examples/slurm/submit_tcga_luad_EGFR_cv_to_external.sh</code>

```bash
export PROJECT_ID=TCGA-LUAD
export GENE=EGFR
export ENCODER=h-optimus-0
export RUN_NAME=TCGA-LUAD
export RUN_MODE=resume
export PER_CLASS=0
export EXTERNAL_PER_CLASS=0
export TARGET_MPP=0.5
export EXTRA_TARGET_MPP=0.25
export EXTERNAL_MANIFEST=/path/to/external_manifest.csv
export EXTERNAL_ROOT=/path/to/foundation_model_training_images/EXTERNAL

sbatch examples/slurm/submit_tcga_luad_EGFR_cv_to_external.sh
```

Notes for SLURM users:
- You must set a **GPU-capable partition** in the script (`#SBATCH -p ...`) and match your cluster’s GPU request syntax (e.g., `--gres=gpu:1`).
- Some clusters require different SBATCH fields (account/QoS/time/memory). Adjust the header in `examples/slurm/submit_tcga_cv_to_external.sh` to match local policy.
</details>

<details>
<summary><strong>nohup (interactive node)</strong> — <code>scripts/nohup_tcga_cv_to_external.sh</code></summary>

```bash
PROJECT_ID=TCGA-LUAD \
GENE=EGFR \
ENCODER=h-optimus-0 \
RUN_MODE=resume \
bash scripts/nohup_tcga_cv_to_external.sh

# watch
tail -f runs/<run-name>/logs/nohup.out
```
</details>

Full example (LUAD EGFR, resume, 20x+40x) with explicit external manifest/root:

```bash
export PROJECT_ID=TCGA-LUAD
export GENE=EGFR
export ENCODER=h-optimus-0
export RUN_NAME=TCGA-LUAD
export RUN_MODE=resume
export PER_CLASS=0
export EXTERNAL_PER_CLASS=0
export TARGET_MPP=0.5
export EXTRA_TARGET_MPP=0.25
export EXTERNAL_MANIFEST=/path/to/external_manifest.csv
export EXTERNAL_ROOT=/path/to/foundation_model_training_images/EXTERNAL
bash /data1/vanderbc/vanderbc/GOLDMARK/scripts/nohup_tcga_cv_to_external.sh
```

Note: external inference only runs if `EXTERNAL_MANIFEST` (and optionally `EXTERNAL_ROOT`) are set.

**End-to-end example (foreground)** — `examples/run_tcga_luad_egfr_end_to_end.sh`

```bash
RUN_MODE=force \
TARGET_MPP=0.5 \
EXTRA_TARGET_MPP=0.25 \
bash examples/run_tcga_luad_egfr_end_to_end.sh
```

**Options (env vars)**
- `PROJECT_ID` (default: `TCGA-LUAD`)
- `GENE` (default: `KRAS`)
- `ENCODER` (default: `h-optimus-0`)
- `DEVICE` (default: `cuda`)
- `TARGET_MPP` (default: `0.5`)
- `EXTRA_TARGET_MPP` (default: `0.25`; comma-separated for additional MPPs)
- `RUN_NAME` (default: `${PROJECT_ID}`)
- `RUN_MODE` in `{force|resume|rebuild}` (default: `force`)
- `PER_CLASS`, `EXTERNAL_PER_CLASS`, `LIMIT_TILES`, `EPOCHS`, `PATIENCE`

By default, `RUN_NAME` is set to the project id so all outputs live under `runs/<project-id>/`.

#### Output layout (important files and where to find them)

All outputs land under `runs/<project-id>/` (the default `RUN_NAME` is the project id).

**A) Project inputs (download + derived labels)**
- `runs/<project-id>/gdc_downloads/<PROJECT>_svs_manifest.tsv` — full GDC SVS manifest for the project
- `runs/<project-id>/gdc_downloads/svs/**/**.svs` — downloaded TCGA SVS files (GDC UUID folders)
- `runs/<project-id>/gdc_downloads/maf/**/**.maf.gz` — downloaded GDC mutation calls used for labels
- `runs/<project-id>/checkpoints/<TARGET>/manifests/<PROJECT>_<TARGET>_svs_manifest_subset.tsv` — selected SVS rows
- `runs/<project-id>/checkpoints/<TARGET>/manifests/<PROJECT>_<TARGET>_slides.csv` — selected slide list + labels (schema below)
- `runs/<project-id>/checkpoints/<TARGET>/manifests/external_manifest_<TARGET>_<encoder>.csv` — external subset manifest (schema below)

**B) Tiling**
- `runs/<project-id>/tiling/tiles_20x/manifests/<slide_id>_tiles.csv` — per-slide tile coordinate manifest (schema below)
- `runs/<project-id>/tiling/tiles_20x/cases/<case_id>_tiles.csv` — per-case tile coordinates aggregated across slides
- `runs/<project-id>/tiling/tiles_20x/cases/cases_index.csv` — case_id → case manifest path index
- `runs/<project-id>/tiling/tiles` — alias pointing at `tiles_20x` for compatibility
- `runs/<project-id>/tiling/tile_coords_20x.csv` — aggregated tile coords (x,y,slide,sample_id,target)
- `runs/<project-id>/tiling/tiling_manifest.csv` — slide inventory (file_name,sample_id,target,slide_path)

When `EXTRA_TARGET_MPP` is set, the additional tiling pass writes to:
- `runs/<project-id>/tiling/tiles_40x/manifests/<slide_id>_tiles.csv` (for `EXTRA_TARGET_MPP=0.25`)
- `runs/<project-id>/tiling/tiles_40x/cases/<case_id>_tiles.csv`
- `runs/<project-id>/tiling/tile_coords_40x.csv` (for `EXTRA_TARGET_MPP=0.25`)
- `runs/<project-id>/tiling/tiles_mpp_<value>/...` for any other MPP

**C) Features + QC**
- `runs/<project-id>/features/<encoder>/features_<slide_id>.pt` — per-slide **20x** feature tensor (N tiles × D)
- `runs/<project-id>/features/<encoder>/features_<slide_id>.json` — QC metadata + checksums (schema below)
- If feature extraction fails QC, tensors are renamed:
  - `features_<slide_id>.FAILED_tile_count_mismatch.pt`
  - `features_<slide_id>.FAILED_degenerate_embeddings.pt`

When `EXTRA_TARGET_MPP` is set, the additional feature pass writes **into the same encoder directory**
using a filename suffix:
- `runs/<project-id>/features/<encoder>/features_<slide_id>_40x.pt` (for `EXTRA_TARGET_MPP=0.25`)
- `runs/<project-id>/features/<encoder>/features_<slide_id>_40x.json`
- `features_<slide_id>_mpp_<value>.*` for any other MPP

**D) Training (5-fold CV)**
- `runs/<project-id>/training/checkpoints/<GENE>/versioned_split_manifest/<GENE>_all_splits_latest.csv` — the split manifest used for tiling/features/training (schema below)
- `runs/<project-id>/training/checkpoints/<GENE>/classification_report/cv_summary.csv` — per-split best epoch + metrics (schema below)
- `runs/<project-id>/training/checkpoints/<GENE>/split_1_set/checkpoint/checkpoint_best.pt` — best checkpoint for that split (and similarly for split_2_set..split_5_set)

**E) Per-split held-out test inference (attention export + ROC/PR plots)**

Written under each split so split context is self-contained:
- `runs/<project-id>/training/checkpoints/<GENE>/split_1_set/inference/test/inference_results.csv`
- `runs/<project-id>/training/checkpoints/<GENE>/split_1_set/inference/test/attention/<slide_id>_attention.csv`
- `runs/<project-id>/training/checkpoints/<GENE>/split_1_set/inference/test/plots/roc_pr_curves.png`

Note: the training stage also writes probability exports under the same directory (e.g. `probabilities_test_set.csv`).

**F) External inference (external LUAD)**

External inference is executed **once** using the **best split** checkpoint and written under that split:
- `runs/<project-id>/training/checkpoints/<GENE>/<best_split>/external_inference/external/inference_results.csv`
- `runs/<project-id>/training/checkpoints/<GENE>/<best_split>/external_inference/external/attention/<slide_id>_attention.csv`
- `runs/<project-id>/training/checkpoints/<GENE>/<best_split>/external_inference/external/plots/roc_pr_curves.png`

For convenience, each non-best split directory contains an `external_inference/external` entry that links to (or points at)
the best-split external inference results.

#### File schemas (column names)

**TCGA slide selection manifest** (`training/checkpoints/<GENE>/manifests/<PROJECT>_<GENE>_slides.csv`)
- `slide_id` (e.g. `TCGA-05-4244-01Z-00-DX1`)
- `slide_path` (downloaded `.svs` path)
- `label_index` (`0`/`1` mutation label for the chosen gene)
- `patient_id` (TCGA case/patient barcode prefix)

**Versioned split manifest** (`training/checkpoints/<GENE>/versioned_split_manifest/<GENE>_all_splits_latest.csv`)
- `slide_path`, `slide_id`, `label_index`
- `target` (legacy column retained for compatibility)
- `split_1_set` … `split_5_set` with values in `{train,val,test}`

**Tile manifest** (`tiling/tiles_20x/manifests/<slide_id>_tiles.csv`)
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
For extra MPPs, the same schema is written with a suffix (e.g., `features_<slide_id>_40x.json`).

**CV summary** (`training/checkpoints/<GENE>/classification_report/cv_summary.csv`)
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

**External inference manifest** (`training/checkpoints/<GENE>/manifests/external_manifest_<GENE>_<encoder>.csv`)

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
- SLURM (generic): `examples/slurm/submit_tcga_cv_to_external.sh`
- SLURM (EGFR example): `examples/slurm/submit_tcga_luad_EGFR_cv_to_external.sh`

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
- External manifest schema (header only): `examples/manifests/external_LUAD_manifest_header.csv`

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
- SLURM template: `examples/slurm/submit_attn_external_BLCA_ERBB2.sh`

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

## Run a full project (default)

The `scripts/tcga_cv_to_external_full_run.py` runner defaults to the **full cohort** (`--per-class 0`).
For a quick sanity check, set `--per-class` to a small value and optionally set `--limit-tiles`.

Important:
- This can download **hundreds of SVS files** (many 10s–100s of GB). Use a scratch filesystem.
- By default the TCGA slide filter keeps diagnostic FFPE slides (`-00-DX`). Use `--allow-non-dx` to disable this.
- Foundation-model encoders may require authentication/approval (e.g. gated HF repos). Put tokens in `configs/secrets.env`.

Example (single full task: TCGA-LUAD EGFR → external LUAD):

```bash
export RUNS_ROOT="runs"
export RUN_NAME="tcga_luad_egfr_full"

python scripts/tcga_cv_to_external_full_run.py \
  --output "${RUNS_ROOT}" \
  --run-name "${RUN_NAME}" \
  --project-id "TCGA-LUAD" \
  --gene "EGFR" \
  --external-per-class 0 \
  --encoder "h-optimus-0" \
  --device "cuda" \
  --epochs 120
```

HPC:
- See `examples/slurm/submit_tcga_luad_EGFR_cv_to_external.sh` and set `PER_CLASS=0` (and optionally `EXTERNAL_PER_CLASS=0`).

## Notes

- This repo intentionally **does not** ship raw WSIs or protected clinical data.
- OncoKB and GDC access are governed by their respective terms; keep tokens out of git history.
- The code is structured for clarity and parity with the manuscript pipeline, not as a polished SDK.
