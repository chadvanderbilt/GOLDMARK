#!/usr/bin/env python3
"""
Annotate MAF files with OncoKB using the byHGVSg endpoint.

This is a public, token-safe version of the internal workflow used to generate
mutation target tables for the manuscript. You must provide an OncoKB token via:
  - --oncokb-token, or
  - the ONCOKB_TOKEN environment variable.

Example:
  export ONCOKB_TOKEN="..."
  python targets/variants/annotate_maf_oncokb_by_hgvsg.py \
    --maf-glob "data/gdc_download/**/**.maf.gz" \
    --output data/oncokb/oncokb_annotations.csv \
    --gene-list data/cancerGeneList.tsv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from goldmark.utils.secrets import load_secrets_env  # noqa: E402


def build_hgvsg(chromosome: str, start: int, end: int, ref: str, alt: str) -> str:
    chromosome = str(chromosome).replace("chr", "")
    ref = str(ref)
    alt = str(alt)
    start = int(start)
    end = int(end)

    if len(ref) == 1 and len(alt) == 1 and ref != "-" and alt != "-":
        return f"{chromosome}:g.{start}{ref}>{alt}"

    if alt == "-" and ref != "-":
        new_end = start + len(ref) - 1
        if len(ref) == 1:
            return f"{chromosome}:g.{start}del"
        return f"{chromosome}:g.{start}_{new_end}del"

    if ref == "-" and alt != "-":
        new_end = start + 1
        return f"{chromosome}:g.{start}_{new_end}ins{alt}"

    new_end = start + len(ref) - 1
    return f"{chromosome}:g.{start}_{new_end}del{ref}ins{alt}"


def _read_gene_list(path: Optional[str]) -> Optional[set[str]]:
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Gene list not found: {p}")
    df = pd.read_csv(p, sep="\t")
    for col in ("Hugo Symbol", "Hugo_Symbol", "gene"):
        if col in df.columns:
            return set(df[col].astype(str).str.strip().tolist())
    raise ValueError(f"Gene list must contain a Hugo symbol column; got columns: {list(df.columns)}")


def main() -> int:
    load_secrets_env()

    parser = argparse.ArgumentParser(description="Annotate MAF rows with OncoKB (byHGVSg).")
    parser.add_argument("--maf-glob", required=True, help="Glob for .maf.gz files (recursive supported with **).")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--oncokb-token", help="OncoKB token (or set ONCOKB_TOKEN).")
    parser.add_argument(
        "--gene-list",
        help="Optional TSV listing cancer genes to keep (column: 'Hugo Symbol').",
    )
    parser.add_argument(
        "--tumor-type",
        default="",
        help="Optional tumor type / OncoTree code for OncoKB (e.g., LUAD).",
    )
    parser.add_argument(
        "--oncotree-code",
        default="",
        help="Alias for --tumor-type (preferred to pass OncoTree code).",
    )
    parser.add_argument("--reference-genome", default="GRCh38", choices=["GRCh38", "GRCh37"])
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Sleep between API requests (rate limiting).")
    parser.add_argument("--max-rows", type=int, help="Optional cap for demo/debug runs.")
    args = parser.parse_args()

    token = args.oncokb_token or os.environ.get("ONCOKB_TOKEN")
    if not token:
        raise ValueError("Missing OncoKB token. Provide --oncokb-token or set ONCOKB_TOKEN.")

    maf_files = sorted(glob.glob(args.maf_glob, recursive=True))
    if not maf_files:
        raise FileNotFoundError(f"No MAF files matched: {args.maf_glob}")

    gene_set = _read_gene_list(args.gene_list)

    api_endpoint = "https://www.oncokb.org/api/v1/annotate/mutations/byHGVSg"
    session = requests.Session()
    session.headers.update({"accept": "application/json", "Authorization": f"Bearer {token}"})
    tumor_type = (args.tumor_type or args.oncotree_code or "").strip()

    cache: Dict[str, Dict[str, Any]] = {}
    out_rows: list[dict[str, Any]] = []
    processed = 0

    columns = [
        "Hugo_Symbol",
        "Chromosome",
        "Start_Position",
        "End_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele2",
        "Tumor_Sample_Barcode",
        "Variant_Classification",
        "Variant_Type",
        "Mutation_Status",
    ]
    optional_columns = [
        "HGVSp_Short",
        "Protein_Change",
        "HGVSp",
    ]

    for maf_path in maf_files:
        maf_path = str(maf_path)
        try:
            if not Path(maf_path).exists() or Path(maf_path).stat().st_size == 0:
                print(f"[warn] Skipping missing/empty MAF: {maf_path}")
                continue
            df = pd.read_csv(maf_path, compression="gzip", sep="\t", comment="#", low_memory=False)
        except Exception as exc:
            print(f"[warn] Failed to read MAF {maf_path}: {exc}")
            continue
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"{maf_path} missing required MAF columns: {missing}")
        keep_cols = columns + [c for c in optional_columns if c in df.columns]
        df = df[keep_cols].copy()
        for col in optional_columns:
            if col not in df.columns:
                df[col] = pd.NA
        if gene_set is not None:
            df = df[df["Hugo_Symbol"].astype(str).isin(gene_set)].copy()
        if df.empty:
            continue

        df["patient_id"] = df["Tumor_Sample_Barcode"].astype(str).str.slice(0, 12)
        df["protein_change"] = (
            df["HGVSp_Short"]
            .fillna(df["Protein_Change"])
            .fillna(df["HGVSp"])
            .astype(str)
            .replace({"nan": ""})
        )
        df["hgvsg"] = df.apply(
            lambda row: build_hgvsg(
                chromosome=row["Chromosome"],
                start=row["Start_Position"],
                end=row["End_Position"],
                ref=row["Reference_Allele"],
                alt=row["Tumor_Seq_Allele2"],
            ),
            axis=1,
        )

        for row in df.itertuples(index=False):
            if args.max_rows is not None and processed >= int(args.max_rows):
                break
            hgvsg = getattr(row, "hgvsg")
            if not hgvsg or str(hgvsg).lower() in {"nan", "none"}:
                continue

            if hgvsg in cache:
                anno = cache[hgvsg]
            else:
                params = {"hgvsg": hgvsg, "referenceGenome": args.reference_genome}
                if tumor_type:
                    params["tumorType"] = tumor_type
                resp = session.get(api_endpoint, params=params, timeout=30)
                if resp.status_code == 429:
                    time.sleep(max(1.0, args.sleep_seconds or 0.0))
                    resp = session.get(api_endpoint, params=params, timeout=30)
                if resp.status_code != 200:
                    anno = {"oncokb_error": f"status={resp.status_code}"}
                else:
                    payload = resp.json() or {}
                    mut_effect = payload.get("mutationEffect") or {}
                    anno = {
                        "oncogenic": payload.get("oncogenic"),
                        "knownEffect": mut_effect.get("knownEffect"),
                        "highestSensitiveLevel": payload.get("highestSensitiveLevel"),
                        "highestFdaLevel": payload.get("highestFdaLevel"),
                        "hotspot": payload.get("hotspot"),
                    }
                cache[hgvsg] = anno
                if args.sleep_seconds:
                    time.sleep(float(args.sleep_seconds))

            out_rows.append(
                {
                    "patient_id": getattr(row, "patient_id"),
                    "Tumor_Sample_Barcode": getattr(row, "Tumor_Sample_Barcode"),
                    "Hugo_Symbol": getattr(row, "Hugo_Symbol"),
                    "Variant_Classification": getattr(row, "Variant_Classification"),
                    "Variant_Type": getattr(row, "Variant_Type"),
                    "Mutation_Status": getattr(row, "Mutation_Status"),
                    "protein_change": getattr(row, "protein_change"),
                    "hgvsg": hgvsg,
                    **anno,
                }
            )
            processed += 1
        if args.max_rows is not None and processed >= int(args.max_rows):
            break

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote OncoKB annotations -> {out_path} (rows={len(out_df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
