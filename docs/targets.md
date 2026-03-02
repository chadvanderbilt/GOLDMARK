# Target construction (TCGA + external)

This repository separates **label generation** (“targets”) from WSI preprocessing and model training.
That separation is intentional: target definitions can change (e.g., stricter clinical criteria), and the
rest of the pipeline should remain identical.

## TCGA (GDC-based)

### Generate GDC manifests via API (SVS + WGS VCF)

If you don’t already have a GDC manifest, you can generate one directly from the GDC API.

SVS (whole-slide images):

```bash
python -m goldmark gdc-manifest svs \
  --project-id TCGA-COAD \
  --out tcga_coad_svs_manifest.tsv
```

WGS VCF (variant calls):

```bash
python -m goldmark gdc-manifest wgs-vcf \
  --project-id TCGA-COAD \
  --out tcga_coad_wgs_vcf_manifest.tsv
```

If your project uses a different variant calling workflow, pass filters like `--workflow-type` and
`--data-type` (and optionally disable the WGS constraint with `--experimental-strategy ""`).

The underlying script entry point is also available at:
`targets/tcga/gdc_generate_manifest.py`

### Download raw files

Use the GDC Data Portal or API to generate a **manifest** (`.tsv`) for the files you want (SVS, MAF, …),
then download via `gdc-client`:

```bash
# Optional but recommended: set GDC_TOKEN_FILE in `configs/secrets.env`
# (copy from `configs/secrets.env.example`).
# `gdc_download.sh` will auto-use $GDC_TOKEN_FILE when present.

python scripts/install_gdc_client.py --dest bin/gdc-client
targets/tcga/gdc_download.sh --manifest tcga_svs_manifest.tsv --out data/gdc_download --gdc-client bin/gdc-client
```

### Variant labeling + OncoKB

This repo includes a token-safe OncoKB annotator:

```bash
# Set ONCOKB_TOKEN in `configs/secrets.env` (recommended) or export ONCOKB_TOKEN in your shell.
python targets/variants/annotate_maf_oncokb_by_hgvsg.py \
  --maf-glob "data/gdc_download/**/**.maf.gz" \
  --output data/oncokb/oncokb_annotations.csv
```

From the annotated mutation table, derive a simple **patient-level binary label** for a target gene:

```bash
python targets/variants/summarize_gene_status.py \
  --annotations data/oncokb/oncokb_annotations.csv \
  --gene PTEN \
  --output data/targets/PTEN_patient_labels.csv
```

The output now includes per-patient **p. changes** and **OncoKB level** summaries:
- `p_changes` (joined with `|` for multiple mutations; `no_gene_changes` if none)
- `oncokb_levels` (joined with `|`)
- `oncokb_positive` / `has_gene_mutation` / `label_reason`

`label_index` is **positive if any mutation is OncoKB level 1/2/3** for the target gene.

Finally, join SVS paths to those patient labels:

```bash
python targets/tcga/build_slide_manifest_from_svs_and_mutations.py \
  --svs-root data/gdc_download \
  --labels data/targets/<GENE>_patient_labels.csv \
  --output data/manifests/TCGA_<GENE>_slide_manifest.csv
```

## External cohort (minimal public example)

This repo does **not** ship clinical external data. For documentation and schema parity only, we include
the **header** of a cohort manifest:

- `examples/manifests/external_LUAD_manifest_header.csv`

## Split manifests (versioned, patient-level)

Training and evaluation use **five repeated patient-level splits** stored in a “versioned split manifest”.

Schema example:
- `examples/manifests/tcga_split_manifest_header.csv`

Generator:

```bash
python scripts/generate_versioned_split_manifest.py \
  --manifest data/manifests/TCGA_<GENE>_slide_manifest.csv \
  --target <GENE> \
  --label-column label_index \
  --target-dir data/checkpoints/<GENE>
```

Output:
`data/checkpoints/<GENE>/versioned_split_manifest/<GENE>_all_splits_latest.csv`
