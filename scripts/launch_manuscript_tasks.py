#!/usr/bin/env python3
"""Launch training scans for all manuscript tumor/target tasks.

This is a thin convenience wrapper around:
  scripts/run_training_scan_target.py

Typical usage (dry-run, prints the commands):

  export MIL_DATA_ROOT="/path/to/foundation_model_training_images"
  python scripts/launch_manuscript_tasks.py

To actually execute:
  python scripts/launch_manuscript_tasks.py --execute
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks-csv",
        default=str(REPO_ROOT / "configs" / "manuscript_tasks.csv"),
        help="CSV with tumor_code,target rows (default: configs/manuscript_tasks.csv)",
    )
    parser.add_argument(
        "--cohorts",
        default="tcga,external",
        help="Comma-separated cohorts to scan (default: tcga,external)",
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("MIL_DATA_ROOT", "data/foundation_model_training_images"),
        help="Training root containing TCGA/ and EXTERNAL/ (or set MIL_DATA_ROOT).",
    )
    parser.add_argument(
        "--encoders",
        default="h-optimus-0,gigapath_ft,virchow,virchow2,uni,prov-gigapath",
        help="Comma-separated encoder list to scan (default mirrors scripts/run_training_scan_target.py).",
    )
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs to request (default: 120).")
    parser.add_argument("--val-interval", type=int, default=999, help="Validation interval override.")
    parser.add_argument("--epoch", type=int, default=120, help="Epoch to require before skipping (default: 120).")
    parser.add_argument(
        "--splits",
        default="split_1_set,split_2_set,split_3_set,split_4_set,split_5_set",
        help="Comma-separated split directories to require.",
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "train_task_v2_pub.sh"),
        help="Training config (shell-style).",
    )
    parser.add_argument("--require-gma-pub", action="store_true", help="Only scan existing *_gma_pub dirs.")
    parser.add_argument("--execute", action="store_true", help="Run commands (default: dry-run).")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after failures.")
    parser.add_argument("--limit", type=int, help="Only run the first N tasks (for debugging).")
    args = parser.parse_args()

    tasks_path = Path(args.tasks_csv).expanduser()
    if not tasks_path.exists():
        raise FileNotFoundError(f"Tasks CSV not found: {tasks_path}")

    cohorts = _parse_list(args.cohorts)
    if not cohorts:
        raise ValueError("--cohorts must include at least one cohort (tcga and/or external).")

    tasks = pd.read_csv(tasks_path)
    if "tumor_code" not in tasks.columns or "target" not in tasks.columns:
        raise ValueError("Tasks CSV must include tumor_code and target columns.")

    if args.limit is not None:
        tasks = tasks.head(int(args.limit)).copy()

    failures = 0
    total = 0
    for cohort in cohorts:
        for row in tasks.itertuples(index=False):
            tumor = str(getattr(row, "tumor_code")).strip()
            target = str(getattr(row, "target")).strip()
            if not tumor or not target:
                continue

            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_training_scan_target.py"),
                "--cohort",
                cohort,
                "--tumor",
                tumor,
                "--target",
                target,
                "--root",
                str(args.root),
                "--encoders",
                str(args.encoders),
                "--epochs",
                str(int(args.epochs)),
                "--val-interval",
                str(int(args.val_interval)),
                "--epoch",
                str(int(args.epoch)),
                "--splits",
                str(args.splits),
                "--config",
                str(args.config),
            ]
            if args.require_gma_pub:
                cmd.append("--require-gma-pub")
            if not args.execute:
                cmd.append("--dry-run")

            total += 1
            print("\n[task]", f"cohort={cohort}", f"tumor={tumor}", f"target={target}")
            print("[cmd]", " ".join(cmd))
            if not args.execute:
                continue
            try:
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError as exc:
                failures += 1
                print(f"[error] Command failed (exit={exc.returncode})")
                if not args.continue_on_error:
                    raise

    mode = "executed" if args.execute else "planned"
    print(f"\nDone ({mode}). tasks={total} failures={failures}")
    if failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
