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

## Start here (60-second smoke test)

Runs end-to-end on **CPU** using **synthetic raster images** (no TCGA/IMPACT data, no OpenSlide required).
Uses the built-in `toy` encoder (no weight downloads).

```bash
git clone https://github.com/chadvanderbilt/GOLDMARK.git
cd GOLDMARK

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python scripts/demo_smoke_test.py
```

Outputs land in `runs/smoke_test/` (including `runs/smoke_test/inference/inference/inference_results.csv`).

Docs:
- `docs/targets.md`
- `docs/pipeline.md`

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
targets/tcga/gdc_download.sh --manifest tcga_coad_svs_manifest.tsv --token gdc_token.txt --out data/gdc_download
```

Annotate MAF with OncoKB (token via env var):

```bash
export ONCOKB_TOKEN="..."
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

## Notes

- This repo intentionally **does not** ship raw WSIs or protected clinical data.
- OncoKB and GDC access are governed by their respective terms; keep tokens out of git history.
- The code is structured for clarity and parity with the manuscript pipeline, not as a polished SDK.
