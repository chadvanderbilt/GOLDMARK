#!/usr/bin/env python3
"""Run CV + external inference for missing attention rows in a plan CSV."""

from pathlib import Path
import argparse
import csv
import json
import os
import subprocess
from typing import Dict, Optional

import pandas as pd


def _context_from_root(gma_root: Path) -> Dict[str, str]:
    gma_root = gma_root.resolve()
    encoder_dir = gma_root.parent
    target_dir = encoder_dir.parent
    checkpoints_dir = target_dir.parent
    project_dir = checkpoints_dir.parent
    cohort_dir = project_dir.parent
    tumor = project_dir.name
    if tumor.startswith("TCGA-") and tumor.endswith("_svs"):
        tumor = tumor[len("TCGA-") : -len("_svs")]
    return {
        "gma_root": str(gma_root),
        "encoder": encoder_dir.name,
        "target": target_dir.name,
        "project": project_dir.name,
        "cohort": cohort_dir.name.lower(),
        "tumor": tumor,
    }


def _resolve_external_cfg(gma_root: Path, cohort: str) -> str:
    cohort = (cohort or "").lower()
    if cohort == "external":
        subdir = "tcga_inference"
        cfg_name = "tcga_run_config.json"
    else:
        subdir = "external_inference"
        cfg_name = "external_run_config.json"

    name = gma_root.name
    if name.endswith("_gma_pub"):
        tapfm_root = gma_root.with_name(name[: -len("_gma_pub")] + "_tapfm")
    elif name.endswith("_gma"):
        tapfm_root = gma_root.with_name(name[: -len("_gma")] + "_tapfm")
    else:
        tapfm_root = gma_root.with_name(name + "_tapfm")

    candidate = tapfm_root / "split_1_set" / subdir / cfg_name
    if candidate.exists():
        return str(candidate)

    target_dir = gma_root.parent.parent
    fallback = list(target_dir.glob(f"*/*_tapfm/split_1_set/{subdir}/{cfg_name}"))
    if fallback:
        return str(fallback[0])
    return ""


def _find_latest_manifest(manifest_dir: Path, target: str) -> Optional[Path]:
    if not manifest_dir.exists():
        return None
    candidates = sorted(
        manifest_dir.glob(f"{target}_all_splits*.csv"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _build_external_manifest_from_splits(
    split_manifest: Path,
    target: str,
    out_csv: Path,
) -> Optional[Path]:
    if not split_manifest.exists():
        return None
    df = pd.read_csv(split_manifest)
    if "slide_path" not in df.columns:
        return None
    if target in df.columns:
        target_col = target
    elif "target" in df.columns:
        target_col = "target"
    else:
        return None
    slide_id_col = None
    for candidate in ("DMP_ASSAY_ID", "slide_id", "slideid", "slideid_mnumber"):
        if candidate in df.columns:
            slide_id_col = candidate
            break
    if slide_id_col is None:
        return None
    out_df = pd.DataFrame(
        {
            "DMP_ASSAY_ID": df[slide_id_col].astype(str),
            "slide_id": df[slide_id_col].astype(str),
            "slide_path": df["slide_path"].astype(str),
            target: df[target_col],
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_csv


def _maybe_write_external_config(
    foundation_root: Path,
    tumor: str,
    target: str,
    encoder: str,
    gma_root: Path,
) -> Optional[Path]:
    external_root = Path(foundation_root) / "EXTERNAL" / tumor
    if not external_root.exists():
        return None
    target_dir = external_root / "checkpoints" / target
    if not target_dir.exists():
        return None
    manifest_path = _find_latest_manifest(target_dir / "versioned_split_manifest", target)
    if manifest_path is None or not manifest_path.exists():
        return None
    out_dir = Path("tmp/external_manifests")
    safe_encoder = encoder.replace("/", "_")
    out_csv = out_dir / f"external_{tumor}_{target}_{safe_encoder}.csv"
    built = _build_external_manifest_from_splits(manifest_path, target, out_csv)
    if built is None:
        return None
    tile_size = _infer_tile_size_from_features(external_root, encoder)
    cfg_path = out_dir / f"external_{tumor}_{target}_{safe_encoder}_config.json"
    cfg = {
        "external_manifest": str(built),
        "external_root": str(external_root),
        "tiling_dir": str(external_root / "tiling"),
        "tile_size": tile_size,
        "mode": "external",
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return cfg_path


def _infer_tile_size_from_features(root_dir: Path, encoder: str) -> int:
    feature_dir = root_dir / "features" / encoder
    if not feature_dir.exists():
        return 224
    for meta_path in feature_dir.glob("features_*.json"):
        try:
            data = json.loads(meta_path.read_text())
            tile_size = int(data.get("tile_size") or 224)
            return tile_size
        except Exception:
            continue
    return 224


def _detect_feature_prefix(feature_dir: Path) -> str:
    for entry in feature_dir.iterdir():
        if entry.name.startswith("features_img"):
            return "img"
    return ""


def _feature_id_for_slide_id(slide_id: str, prefix: str) -> str:
    if prefix and not slide_id.startswith(prefix):
        return f"{prefix}{slide_id}"
    return slide_id


def _manifest_has_header(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            header_line = handle.readline()
        return bool(header_line.strip())
    except Exception:
        return False


def _collect_missing_manifest_ids(
    slide_ids: list,
    feature_dir: Path,
    prefix: str,
    manifest_dir: Path,
) -> list:
    missing_ids = []
    for slide_id in slide_ids:
        feature_id = _feature_id_for_slide_id(slide_id, prefix)
        meta_path = feature_dir / f"features_{feature_id}.json"
        manifest_path_guess = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                manifest_path_guess = Path(meta.get("tile_manifest") or "")
            except Exception:
                manifest_path_guess = None
        if not manifest_path_guess:
            manifest_path_guess = manifest_dir / f"{feature_id}_tiles.csv"
        if not manifest_path_guess.exists() or not _manifest_has_header(manifest_path_guess):
            missing_ids.append(slide_id)
    return missing_ids


def _ensure_tile_manifests_from_config(cfg_path: Path, encoder: str) -> None:
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return
    manifest_path = Path(cfg.get("external_manifest") or cfg.get("manifest") or "")
    root_dir = Path(cfg.get("external_root") or cfg.get("root") or "")
    tiling_dir = Path(cfg.get("tiling_dir") or (root_dir / "tiling"))
    if not manifest_path.exists() or manifest_path.is_dir() or not root_dir.exists() or not tiling_dir.exists():
        return
    tile_size = cfg.get("tile_size")
    if tile_size is None:
        tile_size = _infer_tile_size_from_features(root_dir, encoder)
    try:
        tile_size = int(tile_size)
    except Exception:
        tile_size = 224
    coords_name = "tile_coords_40x.csv" if tile_size >= 400 else "tile_coords_20x.csv"
    coords_path = tiling_dir / coords_name
    if not coords_path.exists():
        print(f"[warn] Missing tile coords for manifest generation: {coords_path}", flush=True)
        return

    whitelist_path = Path("tmp/external_manifests") / f"whitelist_{root_dir.name}_{encoder}.txt"
    whitelist_path.parent.mkdir(parents=True, exist_ok=True)
    slide_ids = []
    with manifest_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return
        slide_key = "DMP_ASSAY_ID" if "DMP_ASSAY_ID" in reader.fieldnames else "slide_id"
        for row in reader:
            slide_id = (row.get(slide_key) or "").strip()
            if slide_id:
                slide_ids.append(slide_id)
    if not slide_ids:
        return

    feature_dir = root_dir / "features" / encoder
    prefix = _detect_feature_prefix(feature_dir) if feature_dir.exists() else "img"
    manifest_dir = tiling_dir / "tiles" / "manifests"
    missing_ids = _collect_missing_manifest_ids(slide_ids, feature_dir, prefix, manifest_dir)

    if not missing_ids:
        return

    def _run_materialize(coords: Path, size: int, ids: list) -> None:
        whitelist_path.write_text("\n".join(ids) + "\n")
        cmd = [
            "python",
            "scripts/materialize_tile_manifests.py",
            str(coords),
            "--output",
            str(tiling_dir / "tiles"),
            "--tile-size",
            str(size),
            "--whitelist",
            str(whitelist_path),
            "--overwrite",
        ]
        print("[run]", " ".join(cmd), flush=True)
        subprocess.check_call(cmd)

    # Primary coords based on tile_size; if still missing, try the other coords file.
    coord_candidates = [(coords_path, int(tile_size))]
    if coords_path.name == "tile_coords_20x.csv":
        alt = tiling_dir / "tile_coords_40x.csv"
        if alt.exists():
            coord_candidates.append((alt, 448))
    else:
        alt = tiling_dir / "tile_coords_20x.csv"
        if alt.exists():
            coord_candidates.append((alt, 224))

    for coords, size in coord_candidates:
        if not missing_ids:
            break
        _run_materialize(coords, size, missing_ids)
        missing_ids = _collect_missing_manifest_ids(slide_ids, feature_dir, prefix, manifest_dir)

    if missing_ids:
        print(
            f"[warn] Still missing {len(missing_ids)} tile manifests after materialization for {root_dir.name}",
            flush=True,
        )


def _attn_index_paths(attn_dir: Path, set_type: str, split_col: str, epoch: int):
    split_tag = split_col or ""
    split_name = f"attn_index_{set_type}_{split_tag}_epoch_{epoch:03d}" if split_tag else ""
    legacy_name = f"attn_index_{set_type}_epoch_{epoch:03d}"
    names = []
    if split_name:
        names.append(split_name)
    if legacy_name not in names:
        names.append(legacy_name)
    paths = []
    for name in names:
        paths.append(attn_dir / f"{name}.parquet")
        paths.append(attn_dir / f"{name}.csv")
    return paths


def _purge_attn_indices(attn_dir: Path, set_type: str, splits: list, epochs: list) -> None:
    if not attn_dir.exists():
        return
    for split_col in splits:
        for epoch in epochs:
            for path in _attn_index_paths(attn_dir, set_type, split_col, epoch):
                if path.exists():
                    path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--foundation-root",
        default=os.environ.get("MIL_DATA_ROOT", "data/foundation_model_training_images"),
        help="Root containing TCGA/ and EXTERNAL/ (or set MIL_DATA_ROOT).",
    )
    parser.add_argument("--tcga-checkpoints", default="best", help="External checkpoints (best or epochs).")
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--pub-only", action="store_true")
    parser.add_argument(
        "--external-from-manifest",
        action="store_true",
        help="If no external config exists for TCGA models, build one from external split manifests.",
    )
    parser.add_argument(
        "--ensure-tiling-manifests",
        action="store_true",
        help="Generate per-slide tile manifests for external inference before running.",
    )
    parser.add_argument("--overwrite-attn", action="store_true")
    parser.add_argument(
        "--purge-attn-index",
        action="store_true",
        help="Remove existing attention index files for the plan epochs before rerun.",
    )
    parser.add_argument("--filter-cohort", default="")
    parser.add_argument("--filter-tumor", default="")
    parser.add_argument("--filter-target", default="")
    parser.add_argument("--filter-encoder", default="")
    parser.add_argument("--filter-project", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to next gma_root if a command fails.",
    )
    parser.add_argument(
        "--error-log",
        default="",
        help="Optional path to append failures (gma_root, exit_code, command).",
    )
    args = parser.parse_args()
    foundation_root = Path(args.foundation_root).expanduser()

    plan = pd.read_csv(args.plan)
    if plan.empty:
        print("No rows in plan; nothing to run.")
        return 0

    filter_cohort = args.filter_cohort.strip().lower()
    filter_tumor = args.filter_tumor.strip()
    filter_target = args.filter_target.strip()
    filter_encoder = args.filter_encoder.strip()
    filter_project = args.filter_project.strip()

    for gma_root_str, grp in plan.groupby("gma_root"):
        gma_root = Path(str(gma_root_str))
        if args.pub_only and not gma_root.name.endswith("_gma_pub"):
            continue
        ctx = _context_from_root(gma_root)
        if filter_cohort and ctx["cohort"] != filter_cohort:
            continue
        if filter_project and ctx["project"] != filter_project:
            continue
        if filter_tumor and ctx["tumor"] != filter_tumor:
            continue
        if filter_target and ctx["target"] != filter_target:
            continue
        if filter_encoder and ctx["encoder"] != filter_encoder:
            continue

        splits = sorted(set(grp["split"].astype(str)))
        epochs = sorted(set(int(e) for e in grp["epoch"].tolist()))
        if args.purge_attn_index:
            cv_attn_dir = gma_root / "tile_attn" / "inference"
            _purge_attn_indices(cv_attn_dir, "cv", splits, epochs)
            external_label = "tcga_inference" if ctx["cohort"] == "external" else "external_inference"
            ext_attn_dir = gma_root / "tile_attn" / external_label
            _purge_attn_indices(ext_attn_dir, external_label, splits, epochs)
        cmd = [
            "python",
            "scripts/gma_inference_pipeline.py",
            "--gma-root",
            str(gma_root),
            "--splits",
            ",".join(splits),
            "--cv-checkpoints",
            ",".join(str(e) for e in epochs),
            "--tcga-checkpoints",
            args.tcga_checkpoints,
            "--device",
            args.device,
        ]
        if args.skip_external:
            cmd.append("--skip-external")
        else:
            external_cfg = _resolve_external_cfg(gma_root, ctx["cohort"])
            if not external_cfg and args.external_from_manifest and ctx["cohort"] == "tcga":
                cfg_path = _maybe_write_external_config(
                    foundation_root,
                    ctx["tumor"],
                    ctx["target"],
                    ctx["encoder"],
                    gma_root,
                )
                external_cfg = str(cfg_path) if cfg_path else ""
                if not external_cfg:
                    print(
                        f"[warn] Missing external manifest for external config: tumor={ctx['tumor']} "
                        f"target={ctx['target']} encoder={ctx['encoder']}",
                        flush=True,
                    )
            if external_cfg:
                if args.ensure_tiling_manifests:
                    _ensure_tile_manifests_from_config(Path(external_cfg), ctx["encoder"])
                cmd.extend(["--external-config", external_cfg])
            elif ctx["cohort"] == "external":
                print(f"[warn] Missing external config for {gma_root}; external inference may be skipped.")
        if args.skip_cv:
            cmd.append("--skip-cv")
        if args.overwrite_attn:
            cmd.append("--overwrite-attn")

        print("[run]", " ".join(cmd))
        if args.dry_run:
            continue
        result = subprocess.run(cmd)
        if result.returncode != 0:
            msg = f"[warn] Command failed ({result.returncode}) for {gma_root}"
            print(msg, flush=True)
            if args.error_log:
                try:
                    Path(args.error_log).parent.mkdir(parents=True, exist_ok=True)
                    with Path(args.error_log).open("a", encoding="utf-8") as handle:
                        handle.write(f"{gma_root}\t{result.returncode}\t{' '.join(cmd)}\n")
                except Exception:
                    pass
            if not args.continue_on_error:
                return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
