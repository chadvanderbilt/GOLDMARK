from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from goldmark.data import ManifestNormalizer
from goldmark.features.progress import FeatureProgressTracker
from goldmark.utils.encoder_naming import derive_encoder_dir_name
from goldmark.utils.paths import PipelinePaths
from goldmark.utils.slide_ids import canonicalize_slide_id


def normalize_manifest(args: argparse.Namespace) -> None:
    normalizer = ManifestNormalizer(
        manifest_path=Path(args.input),
        dataset_type=args.dataset_type,
        slide_root=Path(args.slide_root) if args.slide_root else None,
        target_columns=args.target,
    )
    normalized = normalizer.load()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.data.to_csv(output_path, index=False)
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(normalized.metadata.__dict__, indent=2, default=str))
    print(f"Normalized manifest written to {output_path}")


def run_gdc_manifest(args: argparse.Namespace) -> None:
    from goldmark.targets.gdc_manifest import generate_svs_manifest, generate_wgs_vcf_manifest

    kind = getattr(args, "gdc_kind", "") or getattr(args, "subcommand", "")
    kind = str(kind).strip().lower()
    project_id = str(args.project_id).strip()
    out_path = Path(args.out).expanduser()
    page_size = int(getattr(args, "page_size", 2000) or 2000)
    print_summary = bool(getattr(args, "print_summary", False))

    if kind == "svs":
        count = generate_svs_manifest(
            project_id,
            out_path,
            page_size=page_size,
            print_summary=print_summary,
        )
        print(f"Wrote SVS manifest for {project_id} (files={count}) -> {out_path}")
        return

    if kind == "wgs-vcf":
        count = generate_wgs_vcf_manifest(
            project_id,
            out_path,
            data_category=str(getattr(args, "data_category", "") or ""),
            data_types=getattr(args, "data_type", None),
            workflow_types=getattr(args, "workflow_type", None),
            reference_genomes=getattr(args, "reference_genome", None),
            experimental_strategy=str(getattr(args, "experimental_strategy", "") or ""),
            page_size=page_size,
            print_summary=print_summary,
        )
        print(f"Wrote WGS VCF manifest for {project_id} (files={count}) -> {out_path}")
        return

    raise ValueError(f"Unknown gdc-manifest subcommand '{kind}'. Expected svs or wgs-vcf.")


def run_tiling(args: argparse.Namespace) -> None:
    from goldmark.tiling import SlideTiler, TilingConfig

    manifest = pd.read_csv(args.manifest)
    paths = PipelinePaths(args.output, args.run_name, stage="tiling")
    paths.ensure()

    tiles_dir = Path(args.tiles_dir).expanduser().resolve() if getattr(args, "tiles_dir", None) else paths.tiles_dir
    tiles_dir.mkdir(parents=True, exist_ok=True)

    tiler = SlideTiler(
        TilingConfig(
            tile_size=args.tile_size,
            stride=args.stride,
            target_mpp=args.target_mpp,
            save_tiles=args.save_tiles,
            limit_tiles=args.limit_tiles,
        ),
        output_dir=tiles_dir,
        log_level=args.log_level,
    )

    for row in manifest.itertuples():
        slide_path = getattr(row, args.slide_path_column)
        raw_slide_id = getattr(row, args.slide_id_column)
        tile_slide_id = canonicalize_slide_id(raw_slide_id)
        tiler.tile_slide(slide_path, tile_slide_id)
    print(f"Tile manifests saved under {tiles_dir}")


def run_features(args: argparse.Namespace) -> None:
    from goldmark.features import EncoderConfig, FeatureExtractor

    manifest = pd.read_csv(args.manifest)
    paths = PipelinePaths(args.output, args.run_name, stage="features")
    paths.ensure()
    scope = str(getattr(args, "scope", "all") or "all").lower()

    encoder_output_name = derive_encoder_dir_name(
        preferred=getattr(args, "encoder_output_name", None),
        custom_encoder=args.custom_encoder,
        encoder=args.encoder,
    )
    encoder_output_dir = paths.features_dir / encoder_output_name

    custom_kwargs = {}
    if args.custom_encoder_kwargs:
        try:
            custom_kwargs = json.loads(args.custom_encoder_kwargs)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON for --custom-encoder-kwargs: {exc}") from exc

    tracker = FeatureProgressTracker(
        output_dir=encoder_output_dir,
        encoder_name=encoder_output_name,
        display_name=args.encoder,
        total_slides=len(manifest),
    )

    extractor = FeatureExtractor(
        EncoderConfig(
            name=args.encoder,
            batch_size=args.batch_size,
            precision=args.precision,
            num_workers=getattr(args, "num_workers", 4),
            custom_encoder=args.custom_encoder,
            custom_encoder_script=args.custom_encoder_script,
            custom_encoder_module=args.custom_encoder_module,
            custom_encoder_kwargs=custom_kwargs,
            device=args.device,
            gpu_min_free_gb=args.gpu_min_free_gb,
            tile_size=args.tile_size,
            feature_variant=getattr(args, "feature_variant", "cls_post"),
        ),
        output_dir=encoder_output_dir,
        log_level=args.log_level,
    )

    tile_manifest_dir = Path(args.tile_manifests)

    def _discover_tile_coord_sources(base_dir: Path, max_depth: int = 2) -> list[Path]:
        roots: list[Path] = []
        current = base_dir
        depth = 0
        while current and current not in roots and depth <= max_depth:
            roots.append(current)
            if current.parent == current:
                break
            current = current.parent
            depth += 1

        sources: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.glob("tile_coords*.csv")):
                if path in seen:
                    continue
                sources.append(path)
                seen.add(path)
        return sources

    tile_coord_sources = _discover_tile_coord_sources(tile_manifest_dir)
    tile_manifest_dir.mkdir(parents=True, exist_ok=True)
    tile_manifest_cache_dir = tile_manifest_dir / "manifests"
    tile_manifest_cache_dir.mkdir(parents=True, exist_ok=True)
    tile_coord_frames: Dict[Path, pd.DataFrame] = {}

    def _prepare_tile_manifest(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure generated tile manifests include the columns expected by FeatureExtractor."""

        prepared = df.copy().reset_index(drop=True)
        if "tile_id" not in prepared.columns:
            prepared.insert(0, "tile_id", prepared.index.astype(int))
        if "level" not in prepared.columns:
            prepared["level"] = 0
        tile_size_value = int(getattr(extractor.config, "tile_size", 224) or 224)
        if "tile_size" not in prepared.columns:
            prepared["tile_size"] = tile_size_value
        if "width" not in prepared.columns:
            prepared["width"] = tile_size_value
        if "height" not in prepared.columns:
            prepared["height"] = tile_size_value
        return prepared

    tcga_mode = "tcga" in (args.run_name or "").lower()

    def _match_tile_rows(
        df: pd.DataFrame,
        sample_id: str,
        artifact_id: str,
        slide_basename: str,
        *,
        tcga: bool = False,
    ) -> pd.DataFrame:
        if df.empty:
            return df
        sample_candidates = {value.strip() for value in [sample_id, artifact_id] if value}
        if tcga:
            tcga_extras = set()
            for value in list(sample_candidates):
                if "_" in value:
                    tcga_extras.add(value.split("_", 1)[-1])
                    tcga_extras.add(value.rsplit("_", 1)[-1])
            sample_candidates.update({item for item in tcga_extras if item})
        for column in ("sample_id", "slide_id", "case_id", "slide", "slide_name"):
            if column not in df.columns:
                continue
            series = df[column].astype(str).str.strip()
            mask = series.isin(sample_candidates) if sample_candidates else pd.Series(False, index=series.index)
            if tcga and column == "sample_id" and sample_candidates:
                suffix_first = series.str.split("_", n=1).str[-1]
                suffix_last = series.str.rsplit("_", n=1).str[-1]
                mask = mask | suffix_first.isin(sample_candidates) | suffix_last.isin(sample_candidates)
            if not mask.any() and slide_basename:
                if column == "slide":
                    mask = series.str.endswith(slide_basename)
                else:
                    mask = series == slide_basename
            if mask.any():
                return df[mask]
        return df.iloc[0:0]

    def _ensure_tile_manifest_from_coords(
        sample_id: str,
        artifact_id: str,
        slide_path: str,
        *,
        tcga: bool = False,
    ) -> Optional[Path]:
        if not tile_coord_sources:
            return None
        target_path = tile_manifest_cache_dir / f"{artifact_id}_tiles.csv"
        if target_path.exists():
            return target_path
        slide_basename = Path(slide_path).name if slide_path else ""
        for source in tile_coord_sources:
            df = tile_coord_frames.get(source)
            if df is None:
                try:
                    df = pd.read_csv(source)
                except Exception:
                    continue
                tile_coord_frames[source] = df
            matches = _match_tile_rows(df, sample_id, artifact_id, slide_basename, tcga=tcga)
            if not matches.empty:
                prepared = _prepare_tile_manifest(matches)
                prepared.to_csv(target_path, index=False)
                print(f"[feature-extractor] Generated tile manifest for {sample_id} from {source.name} -> {target_path}")
                return target_path
        return None
    feature_name_suffix = str(getattr(args, "feature_name_suffix", "") or "")
    missing_slides: List[str] = []
    processed_slides = 0
    for row in manifest.itertuples():
        raw_slide_id = getattr(row, args.slide_id_column)
        artifact_slide_id = canonicalize_slide_id(raw_slide_id)
        slide_path = getattr(row, args.slide_path_column)
        feature_slide_id = f"{artifact_slide_id}{feature_name_suffix}" if feature_name_suffix else artifact_slide_id
        feature_path = encoder_output_dir / f"features_{feature_slide_id}.pt"
        if scope == "missing":
            try:
                if feature_path.exists() and feature_path.stat().st_size > 0:
                    if tracker:
                        tracker.skip_slide(
                            feature_slide_id,
                            0,
                            0.0,
                            feature_path,
                            reason="existing_features",
                        )
                    print(f"[feature-extractor] Skipping {raw_slide_id} (features already exist).")
                    processed_slides += 1
                    continue
            except OSError:
                pass
        manifest_candidates = [
            tile_manifest_dir / f"{artifact_slide_id}_tiles.csv",
            tile_manifest_dir / f"{raw_slide_id}_tiles.csv",
            tile_manifest_dir / "manifests" / f"{artifact_slide_id}_tiles.csv",
            tile_manifest_dir / "manifests" / f"{raw_slide_id}_tiles.csv",
        ]
        tile_manifest = next((candidate for candidate in manifest_candidates if candidate.exists()), None)
        if not tile_manifest:
            tile_manifest = _ensure_tile_manifest_from_coords(
                str(raw_slide_id),
                artifact_slide_id,
                slide_path,
                tcga=tcga_mode,
            )
        if not tile_manifest:
            if tracker:
                tracker.fail_slide(
                    feature_slide_id,
                    0,
                    0.0,
                    f"Tile manifest for {raw_slide_id} not found in {tile_manifest_dir}",
                )
            print(
                f"[feature-extractor] Warning: tile manifest for {raw_slide_id} "
                f"not found in {tile_manifest_dir}; skipping this slide."
            )
            missing_slides.append(str(raw_slide_id))
            continue
        try:
            if tile_manifest.stat().st_size == 0:
                message = f"Tile manifest {tile_manifest} is empty; skipping {raw_slide_id}"
                if tracker:
                    tracker.fail_slide(feature_slide_id, 0, 0.0, message)
                print(f"[feature-extractor] Warning: {message}")
                missing_slides.append(str(raw_slide_id))
                continue
        except OSError:
            pass
        try:
            extractor.extract(
                slide_path,
                tile_manifest,
                slide_id=feature_slide_id,
                progress=tracker,
            )
            processed_slides += 1
        except Exception as exc:
            print(
                f"[feature-extractor] Error extracting features for {raw_slide_id}: {exc}. "
                "Continuing with remaining slides."
            )
            continue
    if missing_slides:
        print(
            "[feature-extractor] Skipped slides with missing tile manifests: "
            + ", ".join(missing_slides)
        )
    if processed_slides == 0:
        raise FileNotFoundError(
            f"No tile manifests were found in {tile_manifest_dir}; "
            "cannot extract features for any slides."
        )
    print(f"Features saved under {encoder_output_dir}")


def run_training(args: argparse.Namespace) -> None:
    from goldmark.training import MILTrainer, TrainerConfig
    from goldmark.training.cv import run_cross_validation

    manifest = pd.read_csv(args.manifest)
    paths = PipelinePaths(args.output, args.run_name, stage="training")
    paths.ensure()

    test_value = None if args.test_value and args.test_value.lower() == "none" else args.test_value
    base_config = TrainerConfig(
        aggregator=args.aggregator,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        split_column=args.split_column,
        train_split_value=args.train_value,
        val_split_value=args.val_value,
        test_split_value=test_value,
        dropout=not args.no_dropout,
        device=args.device,
        num_workers=args.num_workers,
        class_weight_positive=args.class_weight_positive,
        encoder_name=args.encoder_name,
    )

    feature_dir = Path(args.feature_dir)

    if args.cv_columns:
        run_cross_validation(
            manifest=manifest,
            feature_dir=feature_dir,
            base_output_dir=paths.checkpoints_dir,
            target_column=args.target,
            base_config=base_config,
            split_columns=args.cv_columns,
            log_level=args.log_level,
        )
    else:
        trainer = MILTrainer(
            manifest=manifest,
            feature_dir=feature_dir,
            output_dir=paths.checkpoints_dir,
            target_column=args.target,
            config=base_config,
            log_level=args.log_level,
        )
        trainer.run()

    print(f"Training artifacts saved to {paths.checkpoints_dir}")


def run_inference(args: argparse.Namespace) -> None:
    from goldmark.inference import InferenceConfig, InferenceRunner

    manifest = pd.read_csv(args.manifest)
    paths = PipelinePaths(args.output, args.run_name, stage="inference")
    paths.ensure()

    runner = InferenceRunner(
        manifest=manifest,
        feature_dir=Path(args.feature_dir),
        checkpoint_path=Path(args.checkpoint),
        output_dir=paths.inference_dir,
        target_column=args.target,
        config=InferenceConfig(
            split_column=args.split_column,
            split_value=args.split_value,
            threshold=args.threshold,
            generate_overlays=not args.no_overlays,
            overlay_alpha=args.overlay_alpha,
            export_attention=bool(getattr(args, "export_attention", False)),
        ),
        log_level=args.log_level,
    )
    runner.run()
    print(f"Inference outputs stored in {paths.inference_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="goldmark", description="GOLDMARK reference pipeline controller")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    subparsers = parser.add_subparsers(dest="command", required=True)

    gdc_parser = subparsers.add_parser("gdc-manifest", help="Generate gdc-client manifests via the GDC API")
    gdc_subparsers = gdc_parser.add_subparsers(dest="gdc_kind", required=True)

    gdc_svs = gdc_subparsers.add_parser("svs", help="Manifest for SVS whole-slide images")
    gdc_svs.add_argument("--project-id", required=True, help="e.g., TCGA-COAD")
    gdc_svs.add_argument("--out", required=True, help="Output manifest TSV path")
    gdc_svs.add_argument("--page-size", type=int, default=2000)
    gdc_svs.add_argument("--print-summary", action="store_true", help="Print basic metadata histograms")
    gdc_svs.set_defaults(func=run_gdc_manifest)

    gdc_wgs = gdc_subparsers.add_parser("wgs-vcf", help="Manifest for WGS VCF variant call files")
    gdc_wgs.add_argument("--project-id", required=True, help="e.g., TCGA-COAD")
    gdc_wgs.add_argument("--out", required=True, help="Output manifest TSV path")
    gdc_wgs.add_argument("--page-size", type=int, default=2000)
    gdc_wgs.add_argument("--print-summary", action="store_true", help="Print basic metadata histograms")
    gdc_wgs.add_argument("--data-category", default="Simple Nucleotide Variation")
    gdc_wgs.add_argument(
        "--data-type",
        action="append",
        default=None,
        help="Repeatable. If omitted, uses common defaults.",
    )
    gdc_wgs.add_argument(
        "--workflow-type",
        action="append",
        default=None,
        help="Repeatable. Optional analysis.workflow_type filter (e.g., 'MuTect2 Annotation').",
    )
    gdc_wgs.add_argument(
        "--reference-genome",
        action="append",
        default=None,
        help="Repeatable. Optional reference_genome filter (e.g., GRCh38).",
    )
    gdc_wgs.add_argument(
        "--experimental-strategy",
        default="WGS",
        help="Optional experimental_strategy filter (default: WGS). Use empty string to disable.",
    )
    gdc_wgs.set_defaults(func=run_gdc_manifest)

    manifest_parser = subparsers.add_parser("manifest", help="Normalize heterogeneous manifests")
    manifest_parser.add_argument("input", help="Path to manifest (csv/tsv)")
    manifest_parser.add_argument("output", help="Destination csv for normalized manifest")
    manifest_parser.add_argument("--slide-root", help="Base directory containing SVS files")
    manifest_parser.add_argument("--dataset-type", choices=["external", "cptac", "tcga", "generic"], help="Force manifest type")
    manifest_parser.add_argument("--target", nargs="*", help="Explicit target columns")
    manifest_parser.set_defaults(func=normalize_manifest)

    tiling_parser = subparsers.add_parser("tiling", help="Generate tile manifests")
    tiling_parser.add_argument("manifest", help="Normalized manifest csv")
    tiling_parser.add_argument("--output", required=True, help="Output root directory")
    tiling_parser.add_argument("--run-name", required=True, help="Name for this pipeline run")
    tiling_parser.add_argument("--slide-id-column", default="slide_id")
    tiling_parser.add_argument("--slide-path-column", default="slide_path")
    tiling_parser.add_argument("--tile-size", type=int, default=224)
    tiling_parser.add_argument("--stride", type=int, default=224)
    tiling_parser.add_argument("--target-mpp", type=float, default=0.5)
    tiling_parser.add_argument(
        "--tiles-dir",
        help="Optional override for the tiling output directory (default: <runs>/<run-name>/tiling/tiles).",
    )
    tiling_parser.add_argument("--limit-tiles", type=int)
    tiling_parser.add_argument("--save-tiles", action="store_true")
    tiling_parser.set_defaults(func=run_tiling)

    features_parser = subparsers.add_parser("features", help="Extract foundation model features")
    features_parser.add_argument("manifest", help="Normalized manifest csv")
    features_parser.add_argument("--tile-manifests", required=True, help="Directory containing tile csv files")
    features_parser.add_argument("--output", required=True, help="Output root directory")
    features_parser.add_argument("--run-name", required=True, help="Name for this pipeline run")
    features_parser.add_argument("--encoder", default="prov-gigapath")
    features_parser.add_argument("--batch-size", type=int, default=256)
    features_parser.add_argument("--precision", default="fp16")
    features_parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="CPU worker processes for decoding tiles (0 = single-process).",
    )
    features_parser.add_argument("--tile-size", type=int, default=224)
    features_parser.add_argument("--slide-id-column", default="slide_id")
    features_parser.add_argument("--slide-path-column", default="slide_path")
    features_parser.add_argument("--custom-encoder")
    features_parser.add_argument("--custom-encoder-script")
    features_parser.add_argument("--custom-encoder-module")
    features_parser.add_argument("--custom-encoder-kwargs")
    features_parser.add_argument(
        "--encoder-output-name",
        help="Directory name (under the features stage) for this encoder's tensors",
    )
    features_parser.add_argument(
        "--feature-name-suffix",
        default="",
        help="Optional suffix appended to per-slide feature filenames (e.g., _40x).",
    )
    features_parser.add_argument("--device", default="auto")
    features_parser.add_argument("--gpu-min-free-gb", type=float, default=2.0)
    features_parser.add_argument(
        "--feature-variant",
        default="cls_post",
        choices=["cls_post", "cls_pre"],
        help="Feature variant for ICML encoders (CLS token post- or pre-final LayerNorm).",
    )
    features_parser.add_argument("--scope", choices=["all", "missing"], default="all")
    features_parser.set_defaults(func=run_features)

    training_parser = subparsers.add_parser("training", help="Train MIL aggregator models")
    training_parser.add_argument("manifest", help="Normalized manifest csv")
    training_parser.add_argument("--feature-dir", required=True, help="Directory with feature tensors")
    training_parser.add_argument("--output", required=True, help="Output root directory")
    training_parser.add_argument("--run-name", required=True, help="Name for this pipeline run")
    training_parser.add_argument("--target", required=True, help="Target column to train")
    training_parser.add_argument("--aggregator", default="gma")
    training_parser.add_argument("--epochs", type=int, default=50)
    training_parser.add_argument("--batch-size", type=int, default=4)
    training_parser.add_argument("--learning-rate", type=float, default=1e-4)
    training_parser.add_argument("--weight-decay", type=float, default=1e-4)
    training_parser.add_argument("--patience", type=int, default=8)
    training_parser.add_argument("--split-column", default="split")
    training_parser.add_argument("--train-value", default="train")
    training_parser.add_argument("--val-value", default="val")
    training_parser.add_argument("--test-value", default="test")
    training_parser.add_argument("--cv-columns", nargs="*", help="Optional list of split columns for cross-validation")
    training_parser.add_argument("--device", default="cuda")
    training_parser.add_argument("--num-workers", type=int, default=0)
    training_parser.add_argument("--class-weight-positive", type=float)
    training_parser.add_argument("--no-dropout", action="store_true")
    training_parser.add_argument("--encoder-name", help="Encoder/feature directory name for progress tracking")
    training_parser.set_defaults(func=run_training)

    inference_parser = subparsers.add_parser("inference", help="Run inference and overlays")
    inference_parser.add_argument("manifest", help="Normalized manifest csv")
    inference_parser.add_argument("--feature-dir", required=True, help="Directory with feature tensors")
    inference_parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    inference_parser.add_argument("--output", required=True, help="Output root directory")
    inference_parser.add_argument("--run-name", required=True, help="Name for this pipeline run")
    inference_parser.add_argument("--target", help="Ground truth column")
    inference_parser.add_argument("--split-column", default="split")
    inference_parser.add_argument("--split-value", default="test")
    inference_parser.add_argument("--threshold", type=float, default=0.5)
    inference_parser.add_argument("--overlay-alpha", type=float, default=0.6)
    inference_parser.add_argument("--no-overlays", action="store_true")
    inference_parser.add_argument("--export-attention", action="store_true", help="Export per-tile attention weights")
    inference_parser.set_defaults(func=run_inference)

    return parser


def main(argv: list[str] | None = None) -> None:
    from goldmark.utils.secrets import load_secrets_env

    load_secrets_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
