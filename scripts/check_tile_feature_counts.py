#!/usr/bin/env python3
"""Compare feature metadata num_tiles vs tile_coords counts (20x/40x)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch


def _derive_sample_id(slide_id: str) -> str:
    slide_id = str(slide_id)
    if slide_id.startswith("img"):
        return slide_id[len("img") :]
    return slide_id


def _collect_feature_metadata(feature_dir: Path) -> Dict[str, Dict[str, object]]:
    metadata: Dict[str, Dict[str, object]] = {}
    for meta_path in sorted(feature_dir.glob("features_*.json")):
        try:
            payload = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        slide_id = payload.get("slide_id") or meta_path.stem.replace("features_", "")
        sample_id = _derive_sample_id(slide_id)
        metadata[sample_id] = {
            "slide_id": slide_id,
            "sample_id": sample_id,
            "num_tiles": payload.get("num_tiles"),
            "json_path": str(meta_path),
        }
    return metadata


def _feature_id_from_path(feature_path: Path) -> str:
    stem = feature_path.stem
    if stem.startswith("features_"):
        stem = stem[len("features_") :]
    return stem


def _load_feature_tensor(feature_path: Path) -> torch.Tensor:
    try:
        payload = torch.load(feature_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(feature_path, map_location="cpu")
    if torch.is_tensor(payload):
        tensor = payload
    elif isinstance(payload, dict):
        tensor = None
        for key in ("features", "embeddings", "data", "values"):
            value = payload.get(key)
            if torch.is_tensor(value):
                tensor = value
                break
        if tensor is None:
            raise ValueError(f"Unsupported feature payload at {feature_path}")
    elif isinstance(payload, (list, tuple)) and payload and torch.is_tensor(payload[0]):
        tensor = payload[0]
    else:
        raise ValueError(f"Unsupported feature payload at {feature_path}")
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.dim() != 2:
        raise ValueError(f"Unexpected feature tensor shape {tuple(tensor.shape)} from {feature_path}")
    return tensor


def _select_tile_coords_path(
    tiling_dir: Path,
    sample_ids: List[str],
    tile_size: int,
    chunksize: int = 500_000,
) -> Path:
    candidates: List[Path] = []
    tile_20x = tiling_dir / "tile_coords_20x.csv"
    tile_40x = tiling_dir / "tile_coords_40x.csv"
    if tile_size >= 448:
        candidates = [tile_40x, tile_20x]
    else:
        candidates = [tile_20x, tile_40x]
    candidates = [path for path in candidates if path.exists()]
    if not candidates:
        raise FileNotFoundError(f"No tile coords found in {tiling_dir}")
    if not sample_ids:
        return candidates[0]
    target_set = set(sample_ids)
    for path in candidates:
        for chunk in pd.read_csv(path, usecols=["sample_id"], chunksize=chunksize):
            if target_set.intersection(set(chunk["sample_id"])):
                return path
    return candidates[0]


def _count_tiles_for_samples(
    tile_coords: Path,
    sample_ids: List[str],
    chunksize: int = 500_000,
) -> Dict[str, int]:
    counts = {sample_id: 0 for sample_id in sample_ids}
    if not tile_coords.exists():
        return counts
    target_set = set(sample_ids)
    for chunk in pd.read_csv(tile_coords, usecols=["sample_id"], chunksize=chunksize):
        subset = chunk[chunk["sample_id"].isin(target_set)]
        if subset.empty:
            continue
        for sample_id, count in subset["sample_id"].value_counts().items():
            counts[sample_id] += int(count)
    return counts


def _ensure_tile_manifests(
    sample_ids: List[str],
    tiling_dir: Path,
    tile_size: int,
    tcga_root: Path,
    tile_coords_path: Path,
    chunksize: int = 500_000,
) -> Path:
    manifest_dir = tiling_dir / "tiles" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pending = [sample_id for sample_id in sample_ids if not (manifest_dir / f"img{sample_id}_tiles.csv").exists()]
    if not pending:
        return manifest_dir
    pending_set = set(pending)
    tile_counts = {sample_id: 0 for sample_id in pending_set}
    for chunk in pd.read_csv(
        tile_coords_path,
        usecols=["x", "y", "slide", "sample_id", "target"],
        chunksize=chunksize,
    ):
        chunk = chunk[chunk["sample_id"].isin(pending_set)]
        if chunk.empty:
            continue
        chunk["slide"] = chunk["slide"].astype(str).apply(
            lambda value: str((tcga_root / value.lstrip("./")).resolve()) if value.startswith("./") else value
        )
        for sample_id, group in chunk.groupby("sample_id"):
            out_path = manifest_dir / f"img{sample_id}_tiles.csv"
            start = tile_counts[sample_id]
            group = group.reset_index(drop=True)
            group.insert(0, "tile_id", range(start, start + len(group)))
            tile_counts[sample_id] = start + len(group)
            group["level"] = 0
            group["tile_size"] = tile_size
            group["width"] = tile_size
            group["height"] = tile_size
            header = not out_path.exists()
            group.to_csv(out_path, mode="a", header=header, index=False)
    return manifest_dir


def _generate_missing_metadata(
    feature_dir: Path,
    tiling_dir: Path,
    tcga_root: Path,
    tile_size: int,
    tile_coords_path: Optional[Path],
    chunksize: int,
) -> None:
    feature_paths = sorted(feature_dir.glob("features_*.pt"))
    if not feature_paths:
        raise SystemExit(f"No features_*.pt files found in {feature_dir}")
    missing = [path for path in feature_paths if not (feature_dir / f"{path.stem}.json").exists()]
    if not missing:
        return
    sample_ids = []
    for path in missing:
        feature_id = _feature_id_from_path(path)
        sample_ids.append(_derive_sample_id(feature_id))
    coords_path = tile_coords_path or _select_tile_coords_path(tiling_dir, sample_ids, tile_size, chunksize)
    manifest_dir = _ensure_tile_manifests(sample_ids, tiling_dir, tile_size, tcga_root, coords_path, chunksize)
    encoder = feature_dir.name
    for path in missing:
        feature_id = _feature_id_from_path(path)
        sample_id = _derive_sample_id(feature_id)
        tile_manifest = manifest_dir / f"img{sample_id}_tiles.csv"
        if not tile_manifest.exists():
            continue
        feature_tensor = _load_feature_tensor(path)
        num_tiles = sum(1 for _ in tile_manifest.open("r", encoding="utf-8")) - 1
        payload = {
            "slide_id": feature_id,
            "encoder": encoder,
            "tile_manifest": str(tile_manifest),
            "num_tiles": int(max(num_tiles, 0)),
            "feature_dim": int(feature_tensor.shape[1]),
            "tile_size": int(tile_size),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        meta_path = feature_dir / f"features_{feature_id}.json"
        meta_path.write_text(json.dumps(payload, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check tile_coords counts against feature num_tiles.")
    parser.add_argument("--feature-dir", required=True, help="Directory with features_*.json metadata.")
    parser.add_argument(
        "--tiling-dir",
        required=True,
        help="Directory containing tile_coords_20x.csv and tile_coords_40x.csv.",
    )
    parser.add_argument("--generate-json", action="store_true", help="Create missing feature JSONs first.")
    parser.add_argument("--tcga-root", help="Root directory for resolving TCGA slide paths.")
    parser.add_argument("--tile-size", type=int, default=224, help="Tile size used during tiling.")
    parser.add_argument("--tile-coords", help="Optional tile_coords CSV path to use.")
    parser.add_argument("--output", help="Output CSV path.")
    parser.add_argument("--chunksize", type=int, default=500_000, help="CSV chunk size.")
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    tiling_dir = Path(args.tiling_dir)
    tile_coords_path = Path(args.tile_coords) if args.tile_coords else None

    if args.generate_json:
        if not args.tcga_root:
            raise SystemExit("--tcga-root is required with --generate-json")
        _generate_missing_metadata(
            feature_dir,
            tiling_dir,
            Path(args.tcga_root),
            args.tile_size,
            tile_coords_path,
            args.chunksize,
        )

    metadata = _collect_feature_metadata(feature_dir)
    if not metadata:
        raise SystemExit(f"No metadata JSONs found in {feature_dir}")

    sample_ids = list(metadata.keys())

    coords_20x = tiling_dir / "tile_coords_20x.csv"
    coords_40x = tiling_dir / "tile_coords_40x.csv"
    counts_20x = _count_tiles_for_samples(coords_20x, sample_ids, args.chunksize)
    counts_40x = _count_tiles_for_samples(coords_40x, sample_ids, args.chunksize)

    rows = []
    for sample_id, info in metadata.items():
        num_tiles = info.get("num_tiles")
        row = {
            "sample_id": sample_id,
            "slide_id": info.get("slide_id"),
            "num_tiles_json": num_tiles,
            "tiles_20x": counts_20x.get(sample_id, 0),
            "tiles_40x": counts_40x.get(sample_id, 0),
            "json_path": info.get("json_path"),
        }
        row["match_20x"] = num_tiles is not None and row["tiles_20x"] == num_tiles
        row["match_40x"] = num_tiles is not None and row["tiles_40x"] == num_tiles
        if num_tiles is None:
            row["issue"] = "missing_num_tiles"
        elif row["tiles_20x"] == 0 and row["tiles_40x"] == 0:
            row["issue"] = "missing_in_tile_coords"
        elif row["match_20x"] or row["match_40x"]:
            row["issue"] = ""
        else:
            row["issue"] = "tile_feature_mismatch"
        rows.append(row)

    df = pd.DataFrame(rows)
    output = Path(args.output) if args.output else feature_dir / "tile_feature_count_check.csv"
    df.to_csv(output, index=False)

    total = len(df)
    mismatch = (df["issue"] == "tile_feature_mismatch").sum()
    missing = (df["issue"] == "missing_in_tile_coords").sum()
    missing_num = (df["issue"] == "missing_num_tiles").sum()
    match_20x = df["match_20x"].sum()
    match_40x = df["match_40x"].sum()
    print(f"Wrote {output}")
    print(
        f"total={total} match_20x={match_20x} match_40x={match_40x} "
        f"mismatch={mismatch} missing_tile_coords={missing} missing_num_tiles={missing_num}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
