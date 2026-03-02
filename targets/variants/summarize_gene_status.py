#!/usr/bin/env python3
"""
Summarize patient-level gene mutation status from an annotated mutation table.

Input:
  - CSV from targets/variants/annotate_maf_oncokb_by_hgvsg.py (or any table with
    patient_id + Hugo_Symbol columns).

Output:
  - patient_id,label_index,p_changes,oncokb_levels,oncokb_positive,has_gene_mutation
    where label_index = 1 if any mutation is OncoKB level 1/2/3 for the requested gene.

Example:
  python targets/variants/summarize_gene_status.py \
    --annotations data/oncokb/oncokb_annotations.csv \
    --gene MSI \
    --output data/targets/MSI_patient_labels.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def _collect_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value is None:
            continue
        token = str(value).strip()
        if not token or token.lower() == "nan":
            continue
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def _is_actionable_level(level: str) -> bool:
    token = str(level or "").strip().upper().replace(" ", "_")
    return token.startswith("LEVEL_1") or token.startswith("LEVEL_2") or token.startswith("LEVEL_3")


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
    all_patients = df[args.patient_column].astype(str).str.strip()
    subset = df[df[args.gene_column].astype(str).str.strip().str.upper() == gene.upper()].copy()
    if subset.empty:
        raise ValueError(f"No rows found for gene '{gene}' in {path}")

    level_column = None
    for candidate in ("highestSensitiveLevel", "highestFdaLevel"):
        if candidate in df.columns:
            level_column = candidate
            break

    protein_column = None
    for candidate in ("protein_change", "HGVSp_Short", "Protein_Change", "HGVSp", "hgvsg"):
        if candidate in df.columns:
            protein_column = candidate
            break

    grouped = subset.groupby(args.patient_column)
    rows = []
    for patient_id in sorted(all_patients.dropna().unique()):
        if patient_id in grouped.groups:
            patient_rows = grouped.get_group(patient_id)
            p_values = _collect_unique(patient_rows[protein_column].tolist() if protein_column else [])
            level_values = _collect_unique(
                patient_rows[level_column].tolist() if level_column else []
            )
            oncokb_positive = any(_is_actionable_level(level) for level in level_values)
            label_index = 1 if oncokb_positive else 0
            if p_values:
                p_changes = "|".join(p_values)
            else:
                p_changes = "no_gene_changes"
            oncokb_levels = "|".join(level_values)
            has_gene_mutation = True
            label_reason = "oncokb_actionable" if oncokb_positive else "non_actionable_gene_mutation"
        else:
            label_index = 0
            p_changes = "no_gene_changes"
            oncokb_levels = ""
            oncokb_positive = False
            has_gene_mutation = False
            label_reason = "no_gene_changes"
        rows.append(
            {
                "patient_id": patient_id,
                "label_index": int(label_index),
                "p_changes": p_changes,
                "oncokb_levels": oncokb_levels,
                "oncokb_positive": int(oncokb_positive),
                "has_gene_mutation": int(has_gene_mutation),
                "label_reason": label_reason,
            }
        )

    labels = pd.DataFrame(rows)

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_path, index=False)
    print(f"Wrote patient labels -> {out_path} (patients={len(labels)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
