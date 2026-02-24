# Pipeline: tiling → features → QC → training → inference

The manuscript pipeline is intentionally modular. Each stage emits **structured intermediate artifacts**
so failures are observable, reproducible, and debuggable.

## Tiling (coordinate maps)

Tiling produces a per-slide CSV of tile coordinates and dimensions:

- `<slide_id>_tiles.csv` with columns including `x`, `y`, `level`, `width`, `height`

The tiler implementation lives in `goldmark/tiling/extractor.py`.

**20x and 40x coordinate maps**

If you want both 20x-like and 40x-like sampling, run tiling twice with different `--target-mpp`
values (dataset-dependent):

- ~20x: `--target-mpp 0.5`
- ~40x: `--target-mpp 0.25`

Keep the resulting tile manifests in separate directories (or separate `--run-name`s).

## Feature extraction (PFM embeddings)

Feature extraction writes per-slide tensors and a JSON metadata file:

- `features_<slide_id>.pt`
- `features_<slide_id>.json`

The extraction code is in `goldmark/features/encoder.py` and supports:
- canonical foundation models (see `goldmark/features/canonical_sources.py`)
- custom encoders via `--custom-encoder-*` CLI options

**Weights and configuration**

Some canonical encoders load weights from local paths (set `MIL_WEIGHTS_DIR`) and some load via
Hugging Face / timm hubs (requires network access or a pre-populated cache).

## QC: tile/feature integrity + degenerate embeddings

This pipeline enforces two critical sanity checks:

1) **Tile/feature count consistency**
   - The number of rows in the tile manifest must match the first dimension of the saved tensor.
2) **Degenerate embeddings**
   - Slides with near-zero embedding variance are flagged (common failure mode for truncated extraction).

If a slide fails QC, its tensor is renamed to:
`features_<slide_id>.FAILED_<reason>.pt`

This design ensures training cannot “silently” proceed on partial or corrupted features.

## Training (reference MIL head)

Training is performed on **predefined split columns** (`split_1_set` … `split_5_set`) and writes
one output directory per split with checkpoints, metrics, and classification reports.

The manuscript orchestration scripts are:
- `scripts/run_training_scan_target.py`
- `scripts/train_task_v2.py`

## Inference + external inference + attention export

The manuscript reports both internal (cross-validation) and external performance, and exports attention
vectors for interpretability and downstream overlay generation.

Manuscript inference entry points:
- `scripts/run_inference_from_plan.py`
- `scripts/gma_inference_pipeline.py`

Plan files let you standardize inference over:
- split (`split_1_set` … `split_5_set`)
- checkpoint choice (e.g., `best` and a fixed late epoch)
