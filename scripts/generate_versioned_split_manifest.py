#!/usr/bin/env python3
"""
Generate a 5-split (or N-split) versioned split manifest for MIL training.

This produces the same on-disk layout used by the manuscript experiments:

  <target_dir>/versioned_split_manifest/<TARGET>_all_splits_latest.csv

Each split column (split_1_set ... split_5_set) contains "train" or "test"
membership, and is generated at the *patient/case* level to avoid leakage.

Example:
  python scripts/generate_versioned_split_manifest.py \
    --manifest /path/to/slide_manifest.csv \
    --target MSI \
    --label-column label_index \
    --target-dir /path/to/checkpoints/MSI
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from goldmark.utils.slide_ids import canonicalize_slide_id


def _derive_group_id(slide_id: str, cohort: str) -> str:
    slide_id = canonicalize_slide_id(slide_id)
    if cohort == "tcga" and slide_id.upper().startswith("TCGA"):
        parts = slide_id.split("-")
        if len(parts) >= 3:
            return "-".join(parts[:3])
        return slide_id
    if cohort == "impact":
        return slide_id.split("-", 2)[0] if "-" in slide_id else slide_id
    return slide_id


def _cohort_guess(series: pd.Series) -> str:
    sample = series.dropna().astype(str).head(50).tolist()
    tcga_hits = sum(1 for value in sample if value.upper().startswith("TCGA"))
    impact_hits = sum(1 for value in sample if value.lower().startswith("imgp-"))
    if tcga_hits >= impact_hits and tcga_hits > 0:
        return "tcga"
    if impact_hits > 0:
        return "impact"
    return "auto"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create versioned split manifest (patient-level CV holdouts).")
    parser.add_argument("--manifest", required=True, help="Input slide-level manifest CSV.")
    parser.add_argument("--target", required=True, help="Target name (used for output filename).")
    parser.add_argument(
        "--label-column",
        default="label_index",
        help="Column containing labels (0/1). Defaults to label_index.",
    )
    parser.add_argument("--slide-id-column", default="slide_id")
    parser.add_argument("--slide-path-column", default="slide_path")
    parser.add_argument(
        "--group-column",
        help="Optional grouping column (e.g., patient_id or case_id). Overrides cohort heuristics.",
    )
    parser.add_argument(
        "--group-regex",
        help="Optional regex with a capture group to derive group id from slide_id (e.g., '^(TCGA-[^-]+-[^-]+)').",
    )
    parser.add_argument(
        "--cohort",
        choices=["tcga", "impact", "auto"],
        default="auto",
        help="Heuristic grouping when --group-column is not provided.",
    )
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--test-frac", type=float, default=0.33, help="Fraction of groups held out per split.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--target-dir",
        required=True,
        help="Directory that contains versioned_split_manifest/ (typically checkpoints/<TARGET>).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser()
    target_dir = Path(args.target_dir).expanduser()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)

    for required in (args.slide_id_column, args.slide_path_column, args.label_column):
        if required not in df.columns:
            raise ValueError(f"Required column '{required}' missing from {manifest_path}")

    label = pd.to_numeric(df[args.label_column], errors="coerce")
    if label.isna().any():
        raise ValueError(f"Label column '{args.label_column}' contains non-numeric entries.")
    label = label.astype(int)
    if not set(label.unique()).issubset({0, 1}):
        raise ValueError(f"Label column '{args.label_column}' must be binary (0/1). Found {sorted(label.unique())}.")

    slide_ids = df[args.slide_id_column].astype(str)

    cohort = args.cohort
    if cohort == "auto":
        cohort = _cohort_guess(slide_ids)
        cohort = "tcga" if cohort == "auto" else cohort

    group_regex: Optional[re.Pattern[str]] = re.compile(args.group_regex) if args.group_regex else None

    def _group_for_row(row_idx: int, raw_slide_id: str) -> str:
        if args.group_column and args.group_column in df.columns:
            value = str(df.at[row_idx, args.group_column])
            value = value.strip()
            if value and value.lower() not in {"nan", "none"}:
                return value
        canonical = canonicalize_slide_id(raw_slide_id)
        if group_regex:
            match = group_regex.search(canonical)
            if match and match.group(1):
                return match.group(1)
        return _derive_group_id(canonical, cohort)

    # Derive group ids (patient/case ids)
    group_ids: list[str] = []
    for row_idx, raw_id in enumerate(slide_ids.tolist()):
        group_ids.append(_group_for_row(row_idx, raw_id))
    df["group_id"] = group_ids
    df[args.label_column] = label
    df["label_index"] = label
    if "target" not in df.columns:
        df["target"] = label
    else:
        df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(label).astype(int)

    groups = (
        df.groupby("group_id", dropna=False)["label_index"]
        .max()
        .reset_index()
        .rename(columns={"label_index": "group_label"})
    )
    if groups.empty:
        raise ValueError("No groups were constructed; check slide_id values and grouping options.")

    splits = max(1, int(args.splits))
    test_frac = float(args.test_frac)
    if not (0.0 < test_frac < 1.0):
        raise ValueError("--test-frac must be between 0 and 1.")

    splitter = StratifiedShuffleSplit(
        n_splits=splits,
        test_size=test_frac,
        random_state=int(args.seed),
    )

    group_labels = groups["group_label"].astype(int).to_numpy()
    split_columns: list[str] = []
    for idx, (_, test_idx) in enumerate(splitter.split(groups, group_labels), start=1):
        column = f"split_{idx}_set"
        split_columns.append(column)
        df[column] = "train"
        test_groups = set(groups.iloc[test_idx]["group_id"].astype(str).tolist())
        df.loc[df["group_id"].astype(str).isin(test_groups), column] = "test"

    out_dir = target_dir / "versioned_split_manifest"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.target}_all_splits_latest.csv"
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {out_path} (use --overwrite to replace)")

    ordered: list[str] = []
    for col in (args.slide_path_column, args.slide_id_column):
        if col in df.columns and col not in ordered:
            ordered.append(col)
    for col in ("DMP_ASSAY_ID", "sample_id", "case_id"):
        if col in df.columns and col not in ordered:
            ordered.append(col)
    for col in ("label_index", "target"):
        if col in df.columns and col not in ordered:
            ordered.append(col)
    ordered.extend(split_columns)

    df[ordered].to_csv(out_path, index=False)
    print(f"Wrote versioned split manifest -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
