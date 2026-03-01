#!/usr/bin/env python
"""
Convert aggregated tile coordinate tables into per-slide manifest CSVs.

Historical external runs sometimes shipped a single ``tile_coords_XXx.csv``
file instead of the per-slide ``*_tiles.csv`` manifests that the modern
dashboard expects. This utility streams the coordinate table, groups rows
by slide ID, and writes canonical manifests so feature extraction can run.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Set, TextIO

from goldmark.utils.slide_ids import canonicalize_slide_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize per-slide tile manifests from coordinate CSVs.")
    parser.add_argument("coords", help="CSV containing tile coordinates (columns: x,y,<sample_id>, ...).")
    parser.add_argument(
        "--output",
        required=True,
        help="Destination tiling/tiles directory (the script will create a manifests/ subdirectory).",
    )
    parser.add_argument("--tile-size", type=int, default=224, help="Tile width/height recorded in the manifest.")
    parser.add_argument("--level", type=int, default=0, help="Whole-slide level to record for each tile.")
    parser.add_argument("--sample-column", default="sample_id", help="Column containing slide identifiers.")
    parser.add_argument("--x-column", default="x", help="Column containing tile X coordinates (in pixels).")
    parser.add_argument("--y-column", default="y", help="Column containing tile Y coordinates (in pixels).")
    parser.add_argument(
        "--whitelist",
        help="Optional text file listing slide IDs to include (one per line). "
        "If omitted, the script processes every sample present in the coordinate CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild manifests even if destination files already exist.",
    )
    return parser.parse_args()


def _load_whitelist(path: Path | None) -> Set[str]:
    if path is None:
        return set()
    allowed: Set[str] = set()
    for line in path.read_text().splitlines():
        token = line.strip()
        if not token:
            continue
        allowed.add(canonicalize_slide_id(token))
    return allowed


def _existing_slides(manifest_dir: Path) -> Set[str]:
    slides: Set[str] = set()
    if not manifest_dir.exists():
        return slides
    for candidate in manifest_dir.glob("*_tiles.csv"):
        slides.add(candidate.name.replace("_tiles.csv", ""))
    return slides


def _require_columns(fieldnames: Iterable[str], required: Set[str]) -> None:
    missing = required.difference(set(fieldnames or []))
    if missing:
        raise ValueError(f"Coordinate file is missing required columns: {sorted(missing)}")


def main() -> None:
    args = _parse_args()
    coords_path = Path(args.coords).expanduser().resolve()
    if not coords_path.exists():
        raise FileNotFoundError(f"Coordinate file not found: {coords_path}")

    output_root = Path(args.output).expanduser().resolve()
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    whitelist = _load_whitelist(Path(args.whitelist).expanduser()) if args.whitelist else set()
    existing = _existing_slides(manifest_dir)
    skipped_existing = 0

    fieldnames = [
        "slide_id",
        "tile_id",
        "x",
        "y",
        "level",
        "width",
        "height",
        "tissue_fraction",
        "tile_path",
    ]

    open_files: Dict[str, TextIO] = {}
    writers: Dict[str, csv.DictWriter] = {}
    tile_counts = defaultdict(int)

    try:
        with coords_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            _require_columns(reader.fieldnames, {args.sample_column, args.x_column, args.y_column})
            for row in reader:
                raw_id = (row.get(args.sample_column) or "").strip()
                if not raw_id:
                    continue
                slide_id = canonicalize_slide_id(raw_id)
                if whitelist and slide_id not in whitelist:
                    continue
                if slide_id in existing and not args.overwrite:
                    skipped_existing += 1
                    continue

                writer = writers.get(slide_id)
                if writer is None:
                    dest = manifest_dir / f"{slide_id}_tiles.csv"
                    if dest.exists() and args.overwrite:
                        dest.unlink()
                    elif dest.exists():
                        # This can happen if files appeared after we computed `existing`.
                        skipped_existing += 1
                        existing.add(slide_id)
                        continue
                    file_handle = dest.open("w", newline="")
                    open_files[slide_id] = file_handle
                    writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writers[slide_id] = writer

                tile_index = tile_counts[slide_id]
                tile_counts[slide_id] = tile_index + 1
                tile_id = f"{slide_id}_{tile_index:05d}"

                try:
                    x_coord = int(float(row[args.x_column]))
                    y_coord = int(float(row[args.y_column]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid coordinates for slide {slide_id}: {row}") from exc

                writer.writerow(
                    {
                        "slide_id": slide_id,
                        "tile_id": tile_id,
                        "x": x_coord,
                        "y": y_coord,
                        "level": args.level,
                        "width": args.tile_size,
                        "height": args.tile_size,
                        "tissue_fraction": 1.0,
                        "tile_path": "",
                    }
                )
    finally:
        for file_handle in open_files.values():
            file_handle.close()

    for slide_id, count in tile_counts.items():
        print(f"[tile-manifests] Wrote {count:,} tiles for {slide_id}")
    if skipped_existing and not args.overwrite:
        print(f"[tile-manifests] Skipped {skipped_existing:,} rows because manifests already existed (use --overwrite).")


if __name__ == "__main__":
    main()
