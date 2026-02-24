#!/usr/bin/env python3
"""
Summarize patient-level gene mutation status from an annotated mutation table.

Input:
  - CSV from targets/variants/annotate_maf_oncokb_by_hgvsg.py (or any table with
    patient_id + Hugo_Symbol columns).

Output:
  - patient_id,label_index
    where label_index = 1 if the patient has >=1 mutation for the requested gene.

Example:
  python targets/variants/summarize_gene_status.py \
    --annotations data/oncokb/oncokb_annotations.csv \
    --gene MSI \
    --output data/targets/MSI_patient_labels.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize per-patient gene mutation status.")
    parser.add_argument("--annotations", required=True, help="Annotated mutation CSV.")
    parser.add_argument("--gene", required=True, help="Gene symbol to summarize (e.g., PTEN).")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--patient-column", default="patient_id")
    parser.add_argument("--gene-column", default="Hugo_Symbol")
    args = parser.parse_args()

    path = Path(args.annotations).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Annotations not found: {path}")
    df = pd.read_csv(path)
    for required in (args.patient_column, args.gene_column):
        if required not in df.columns:
            raise ValueError(f"Missing column '{required}' in {path}")

    gene = args.gene.strip()
    subset = df[df[args.gene_column].astype(str).str.strip().str.upper() == gene.upper()].copy()
    if subset.empty:
        raise ValueError(f"No rows found for gene '{gene}' in {path}")

    patient_ids = subset[args.patient_column].astype(str).str.strip()
    labels = (
        pd.DataFrame({"patient_id": patient_ids})
        .assign(label_index=1)
        .dropna()
        .drop_duplicates(subset=["patient_id"])
        .sort_values("patient_id")
    )

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_path, index=False)
    print(f"Wrote patient labels -> {out_path} (patients={len(labels)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

