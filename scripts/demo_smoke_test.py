#!/usr/bin/env python3
"""End-to-end smoke test for GOLDMARK (no TCGA/IMPACT data required).

This script generates a tiny synthetic dataset of raster "slides" and then runs:

  tiling -> features (toy encoder) -> training -> inference

Outputs are written under:
  <output>/<run-name>/

The smoke test is designed to run on CPU and does not require OpenSlide.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from goldmark.cli import main as goldmark_main  # noqa: E402


def _assign_splits(
    labels: list[int],
    rng: np.random.Generator,
    *,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> list[str]:
    indices_by_class: dict[int, list[int]] = {0: [], 1: []}
    for idx, label in enumerate(labels):
        indices_by_class[int(label)].append(idx)

    splits = [""] * len(labels)
    for label_value, idxs in indices_by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = max(1, int(round(train_frac * n))) if n >= 3 else max(1, n - 2)
        n_val = max(1, int(round(val_frac * n))) if n >= 3 else 1
        n_train = min(n_train, n - 2) if n >= 3 else max(1, n_train)
        n_val = min(n_val, n - n_train - 1) if n >= 3 else min(n_val, max(0, n - n_train))

        train_idx = idxs[:n_train]
        val_idx = idxs[n_train : n_train + n_val]
        test_idx = idxs[n_train + n_val :]

        for i in train_idx:
            splits[i] = "train"
        for i in val_idx:
            splits[i] = "val"
        for i in test_idx:
            splits[i] = "test"

    if any(s == "" for s in splits):
        raise RuntimeError("Split assignment failed; some entries were not labeled.")
    return splits


def _write_synthetic_slide(path: Path, label: int, rng: np.random.Generator, *, size: int) -> None:
    base = np.array([190, 70, 70], dtype=np.float32) if int(label) == 1 else np.array([70, 70, 190], dtype=np.float32)
    noise = rng.normal(loc=0.0, scale=25.0, size=(size, size, 3)).astype(np.float32)
    canvas = np.clip(base + noise, 0, 255).astype(np.uint8)

    stripe = max(8, size // 28)
    canvas[::stripe, :, 1] = 255 - canvas[::stripe, :, 1]
    canvas[:, ::stripe, 2] = 255 - canvas[:, ::stripe, 2]

    Image.fromarray(canvas, mode="RGB").save(path)


def _run_goldmark(argv: list[str]) -> None:
    print("\n==> goldmark " + " ".join(argv))
    goldmark_main(argv)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs", help="Output root directory (default: runs)")
    parser.add_argument("--run-name", default="smoke_test", help="Run name under the output root")
    parser.add_argument("--num-slides", type=int, default=20, help="Number of synthetic slides to generate")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5)")
    parser.add_argument("--tile-size", type=int, default=224, help="Tile size (default: 224)")
    parser.add_argument("--stride", type=int, default=224, help="Stride (default: 224)")
    parser.add_argument("--image-size", type=int, default=896, help="Synthetic slide edge length in pixels")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing run directory")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    output_root = Path(args.output).expanduser().resolve()
    run_dir = output_root / args.run_name
    if run_dir.exists():
        if not args.force:
            raise FileExistsError(f"Run directory already exists: {run_dir} (use --force to overwrite)")
        shutil.rmtree(run_dir)

    data_dir = run_dir / "smoke_data"
    slides_dir = data_dir / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    num_slides = max(4, int(args.num_slides))
    labels = ([0] * (num_slides // 2)) + ([1] * (num_slides - (num_slides // 2)))
    rng.shuffle(labels)
    splits = _assign_splits(labels, rng)

    rows = []
    for idx, (label, split) in enumerate(zip(labels, splits)):
        slide_id = f"demo_slide_{idx:03d}"
        slide_path = slides_dir / f"{slide_id}.png"
        _write_synthetic_slide(slide_path, int(label), rng, size=int(args.image_size))
        rows.append(
            {
                "slide_id": slide_id,
                "slide_path": str(slide_path),
                "label_index": int(label),
                "split": str(split),
            }
        )

    manifest_path = data_dir / "smoke_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"Wrote synthetic manifest: {manifest_path}")

    _run_goldmark(
        [
            "tiling",
            str(manifest_path),
            "--output",
            str(output_root),
            "--run-name",
            str(args.run_name),
            "--tile-size",
            str(int(args.tile_size)),
            "--stride",
            str(int(args.stride)),
        ]
    )

    tile_dir = run_dir / "tiling" / "tiles"
    _run_goldmark(
        [
            "features",
            str(manifest_path),
            "--tile-manifests",
            str(tile_dir),
            "--output",
            str(output_root),
            "--run-name",
            str(args.run_name),
            "--encoder",
            "toy",
            "--device",
            "cpu",
            "--batch-size",
            "32",
            "--num-workers",
            "0",
            "--tile-size",
            str(int(args.tile_size)),
        ]
    )

    feature_dir = run_dir / "features" / "toy"
    _run_goldmark(
        [
            "training",
            str(manifest_path),
            "--feature-dir",
            str(feature_dir),
            "--output",
            str(output_root),
            "--run-name",
            str(args.run_name),
            "--target",
            "label_index",
            "--epochs",
            str(int(args.epochs)),
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--split-column",
            "split",
            "--train-value",
            "train",
            "--val-value",
            "val",
            "--test-value",
            "test",
        ]
    )

    ckpt = run_dir / "training" / "checkpoints" / "checkpoint" / "checkpoint_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Expected best checkpoint not found: {ckpt}")

    _run_goldmark(
        [
            "inference",
            str(manifest_path),
            "--feature-dir",
            str(feature_dir),
            "--checkpoint",
            str(ckpt),
            "--output",
            str(output_root),
            "--run-name",
            str(args.run_name),
            "--target",
            "label_index",
            "--split-column",
            "split",
            "--split-value",
            "test",
            "--no-overlays",
        ]
    )

    results_path = run_dir / "inference" / "inference" / "inference_results.csv"
    print("\nSmoke test complete.")
    print(f"- Run directory: {run_dir}")
    print(f"- Inference results: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
