#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path


DEFAULT_ENCODERS = ["h-optimus-0", "gigapath_ft", "virchow", "virchow2", "uni", "prov-gigapath"]


def _project_root(root: Path, cohort: str, tumor: str) -> Path:
    cohort = cohort.lower()
    if cohort == "tcga":
        return root / "TCGA" / f"TCGA-{tumor}_svs"
    return root / "IMPACT" / tumor


def _missing_any_split(gma_root: Path, split_dirs: list, epoch: int) -> bool:
    for split_dir in split_dirs:
        ckpt = gma_root / split_dir / "checkpoint" / f"checkpoint_epoch_{epoch:03d}.pt"
        if not ckpt.exists():
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan encoders for a target and run training where epoch-120 is missing.")
    parser.add_argument("--cohort", required=True, help="tcga or impact")
    parser.add_argument("--tumor", required=True, help="Tumor code (e.g., COAD)")
    parser.add_argument("--target", required=True, help="Target gene (e.g., PTEN)")
    parser.add_argument(
        "--root",
        default=os.environ.get("MIL_DATA_ROOT", "data/foundation_model_training_images"),
        help="Training root containing TCGA/ and IMPACT/ (or set MIL_DATA_ROOT).",
    )
    parser.add_argument(
        "--encoders",
        default=",".join(DEFAULT_ENCODERS),
        help="Comma-separated encoder list to scan.",
    )
    parser.add_argument("--require-gma-pub", action="store_true", help="Only scan existing *_gma_pub dirs.")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs.")
    parser.add_argument("--val-interval", type=int, default=999, help="Validation interval override.")
    parser.add_argument("--epoch", type=int, default=120, help="Epoch to require before skipping.")
    parser.add_argument(
        "--splits",
        default="split_1_set,split_2_set,split_3_set,split_4_set,split_5_set",
        help="Comma-separated split directories to require.",
    )
    parser.add_argument("--config", default="configs/train_task_v2_pub.sh", help="Training config (shell-style).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    split_dirs = [item.strip() for item in args.splits.split(",") if item.strip()]
    encoders = [item.strip() for item in args.encoders.split(",") if item.strip()]

    root = Path(args.root)
    project_dir = _project_root(root, args.cohort, args.tumor)
    target_dir = project_dir / "checkpoints" / args.target
    manifest = target_dir / "versioned_split_manifest" / f"{args.target}_all_splits_latest.csv"
    if not manifest.exists():
        print(f"[skip] missing manifest {manifest}")
        return 0

    for encoder in encoders:
        feature_dir = project_dir / "features" / encoder
        if not feature_dir.exists():
            continue
        gma_root = target_dir / encoder / f"{args.target}_gma_pub"
        if args.require_gma_pub and not gma_root.exists():
            continue
        if not _missing_any_split(gma_root, split_dirs, int(args.epoch)):
            print(f"[skip] checkpoint exists for {gma_root}")
            continue

        cmd = [
            "python",
            "scripts/train_task_v2.py",
            "--config",
            args.config,
            "--manifest",
            str(manifest),
            "--feature-dir",
            str(feature_dir),
            "--target-column",
            "label_index",
            "--output-dir",
            str(gma_root),
            "--encoder-name",
            encoder,
            "--epochs",
            str(int(args.epochs)),
            "--val-interval",
            str(int(args.val_interval)),
            "--log-level",
            "INFO",
        ]
        print("[run]", " ".join(cmd))
        if args.dry_run:
            continue
        subprocess.check_call(cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
