# GOLDMARK — Manuscript Reference Pipeline

This repository is a **minimal, GitHub-ready snapshot** of the code paths used for the manuscript’s
end-to-end benchmarking pipeline:

1) **Target construction** (TCGA via GDC download + variant labeling / OncoKB annotation)
2) **Tiling** (20x/40x coordinate generation)
3) **Feature extraction** (canonical pathology foundation models + custom encoders) with **QC metadata**
4) **Training** (gated-attention MIL reference head) on **predefined patient-level splits**
5) **Inference + external inference** and **attention export** (best checkpoint + fixed late epoch)

The design goal is reproducibility and clarity: **cross-validation alone is not sufficient**—the pipeline
is built around **reciprocal external testing** (e.g., TCGA→IMPACT and IMPACT→TCGA) under identical
preprocessing and evaluation criteria.

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

This repo targets **Python 3.10+**. A minimal dependency list is provided in `requirements.txt`.

Typical (pip) install:

```bash
python -m pip install -r requirements.txt
```

Notes:
- `openslide-python` requires the system OpenSlide library (platform-specific).
- Some canonical encoders download weights via Hugging Face / timm; for offline runs, pre-cache weights or use `--custom-encoder`.

## Quickstart (typical flow)

Set a repo root (recommended):

```bash
export REPO_ROOT="$PWD"
```

### 1) Targets (TCGA example)

Generate a GDC manifest (SVS example):

```bash
python targets/tcga/gdc_generate_manifest.py svs \
  --project-id TCGA-COAD \
  --out tcga_coad_svs_manifest.tsv
```

Download data from GDC:

```bash
targets/tcga/gdc_download.sh --manifest tcga_svs_manifest.tsv --token gdc_token.txt --out data/gdc_download
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

Tile manifests are written under `runs/<run-name>/tiling/tiles/` as `<slide_id>_tiles.csv`.

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
  --tile-manifest-dir runs/tcga_pten_demo/tiling/tiles \
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

## Notes

- This repo intentionally **does not** ship raw WSIs or protected clinical data.
- OncoKB and GDC access are governed by their respective terms; keep tokens out of git history.
- The code is structured for clarity and parity with the manuscript pipeline, not as a polished SDK.
