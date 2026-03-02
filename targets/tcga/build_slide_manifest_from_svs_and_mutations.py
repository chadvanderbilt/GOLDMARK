#!/usr/bin/env python3
"""
Build a slide-level training manifest by joining TCGA SVS paths with per-patient
mutation labels.

Input:
  - A directory tree containing .svs files downloaded from GDC.
  - A patient-level label table (e.g., derived from MAF/VCF + OncoKB) with a
    patient id column and a binary label column.

Output:
  A CSV with the minimal columns used throughout this repo:
    slide_path, slide_id, DMP_ASSAY_ID, label_index, target
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _tcga_barcode_from_filename(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".svs"):
        name = name[:-4]
    # Many TCGA filenames are "<BARCODE>.<UUID>"
    return name.split(".", 1)[0]


def _tcga_patient_id(barcode: str) -> str:
    return barcode[:12]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build slide manifest from SVS files and patient labels.")
    parser.add_argument("--svs-root", required=True, help="Directory containing SVS files (nested OK).")
    parser.add_argument("--labels", required=True, help="CSV with patient ids + labels.")
    parser.add_argument(
        "--patient-column",
        default="patient_id",
        help="Patient id column in --labels (common: patient_id or patientId).",
    )
    parser.add_argument(
        "--label-column",
        default="label_index",
        help="Binary label column in --labels (0/1).",
    )
    parser.add_argument(
        "--slide-id-format",
        choices=["barcode", "patient_barcode"],
        default="patient_barcode",
        help="slide_id formatting: barcode (TCGA-..-..-..) or patient_barcode (PATIENT_BARCODE).",
    )
    parser.add_argument("--output", required=True, help="Output CSV path.")
    args = parser.parse_args()

    svs_root = Path(args.svs_root).expanduser()
    if not svs_root.exists():
        raise FileNotFoundError(f"SVS root not found: {svs_root}")

    label_path = Path(args.labels).expanduser()
    if not label_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {label_path}")
    labels = pd.read_csv(label_path)
    if args.patient_column not in labels.columns:
        raise ValueError(f"Missing patient column '{args.patient_column}' in {label_path}")
    if args.label_column not in labels.columns:
        raise ValueError(f"Missing label column '{args.label_column}' in {label_path}")
    labels = labels.copy()
    labels[args.patient_column] = labels[args.patient_column].astype(str).str.strip()
    labels[args.label_column] = pd.to_numeric(labels[args.label_column], errors="raise").astype(int)

    svs_paths = sorted([p for p in svs_root.rglob("*.svs") if p.is_file()])
    if not svs_paths:
        raise FileNotFoundError(f"No .svs files found under {svs_root}")

    rows = []
    for path in svs_paths:
        barcode = _tcga_barcode_from_filename(path)
        patient_id = _tcga_patient_id(barcode)
        if args.slide_id_format == "barcode":
            slide_id = barcode
        else:
            slide_id = f"{patient_id}_{barcode}"
        rows.append(
            {
                "slide_path": str(path),
                "slide_id": slide_id,
                "DMP_ASSAY_ID": slide_id,
                "patient_id": patient_id,
            }
        )

    manifest = pd.DataFrame(rows)
    rename_map = {args.patient_column: "patient_id", args.label_column: "label_index"}
    merged = manifest.merge(labels.rename(columns=rename_map), how="left", on="patient_id")
    if merged["label_index"].isna().any():
        missing = merged.loc[merged["label_index"].isna(), "patient_id"].astype(str).unique().tolist()[:10]
        raise ValueError(
            "Some SVS files have no matching label rows. "
            f"Example missing patient_ids: {missing} (showing up to 10)"
        )
    merged["label_index"] = merged["label_index"].astype(int)
    merged["target"] = merged["label_index"]

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    core_cols = ["slide_path", "slide_id", "DMP_ASSAY_ID", "label_index", "target"]
    extra_cols = [col for col in merged.columns if col not in core_cols]
    merged[core_cols + extra_cols].to_csv(out_path, index=False)
    print(f"Wrote slide manifest -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
