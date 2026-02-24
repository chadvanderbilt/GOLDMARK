#!/usr/bin/env python3
"""
Generate GDC manifest TSVs for a TCGA project via the GDC API.

This is a minimal, public replacement for internal helpers like
TCGA_api_query_svs.py. The output is a *gdc-client* compatible manifest:

  id <tab> filename <tab> md5 <tab> size <tab> state

Examples
--------
SVS (whole-slide images):
  python targets/tcga/gdc_generate_manifest.py svs \
    --project-id TCGA-COAD \
    --out tcga_coad_svs_manifest.tsv

WGS VCF (variant call files):
  python targets/tcga/gdc_generate_manifest.py wgs-vcf \
    --project-id TCGA-COAD \
    --out tcga_coad_wgs_vcf_manifest.tsv

Notes
-----
- For controlled-access files, you must download with a GDC token (see
  targets/tcga/gdc_download.sh).
- The GDC variant-call schema varies by project and pipeline; use the optional
  flags to adjust filters (data_type, workflow_type, etc.).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests


FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
DEFAULT_FIELDS = (
    "file_id",
    "file_name",
    "md5sum",
    "file_size",
    "file_state",
    "data_category",
    "data_type",
    "data_format",
    "experimental_strategy",
    "analysis.workflow_type",
)


@dataclass(frozen=True)
class QueryConfig:
    project_id: str
    filters: Dict[str, Any]
    fields: tuple[str, ...] = DEFAULT_FIELDS


def _in_filter(field: str, values: Iterable[str]) -> Dict[str, Any]:
    values = [v for v in (str(x).strip() for x in values) if v]
    return {"op": "in", "content": {"field": field, "value": values}}


def _and_filter(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"op": "and", "content": [item for item in items if item]}


def _build_svs_filters(project_id: str) -> Dict[str, Any]:
    return _and_filter(
        [
            _in_filter("cases.project.project_id", [project_id]),
            _in_filter("data_category", ["Biospecimen"]),
            _in_filter("data_type", ["Slide Image"]),
            _in_filter("data_format", ["SVS"]),
        ]
    )


def _build_wgs_vcf_filters(
    project_id: str,
    *,
    data_category: str,
    data_types: List[str],
    workflow_types: List[str],
    reference_genomes: List[str],
    experimental_strategy: str,
) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = [
        _in_filter("cases.project.project_id", [project_id]),
        _in_filter("data_format", ["VCF"]),
    ]
    if data_category:
        items.append(_in_filter("data_category", [data_category]))
    if data_types:
        items.append(_in_filter("data_type", data_types))
    if workflow_types:
        items.append(_in_filter("analysis.workflow_type", workflow_types))
    if reference_genomes:
        items.append(_in_filter("reference_genome", reference_genomes))
    if experimental_strategy:
        items.append(_in_filter("experimental_strategy", [experimental_strategy]))
    return _and_filter(items)


def _request_with_retries(session: requests.Session, params: Dict[str, Any], retries: int = 6) -> requests.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(max(1, retries)):
        try:
            resp = session.get(FILES_ENDPOINT, params=params, timeout=60)
            if resp.status_code in {429, 502, 503, 504}:
                wait = min(60.0, 1.5 ** attempt)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = min(60.0, 1.5 ** attempt)
            time.sleep(wait)
    raise RuntimeError(f"GDC request failed after retries: {last_exc}")


def _query_all(cfg: QueryConfig, *, page_size: int = 2000) -> List[Dict[str, Any]]:
    session = requests.Session()
    hits: List[Dict[str, Any]] = []
    offset = 0

    while True:
        params = {
            "filters": json.dumps(cfg.filters),
            "fields": ",".join(cfg.fields),
            "format": "JSON",
            "size": str(int(page_size)),
            "from": str(int(offset)),
        }
        resp = _request_with_retries(session, params=params)
        payload = resp.json()
        data = payload.get("data") or {}
        page_hits = data.get("hits") or []
        hits.extend(page_hits)

        pagination = data.get("pagination") or {}
        total = int(pagination.get("total") or len(hits))
        offset += int(pagination.get("count") or len(page_hits))
        if offset >= total or not page_hits:
            break
    return hits


def _write_gdc_manifest(hits: List[Dict[str, Any]], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "filename", "md5", "size", "state"])
        for hit in hits:
            writer.writerow(
                [
                    hit.get("file_id") or "",
                    hit.get("file_name") or "",
                    hit.get("md5sum") or "",
                    hit.get("file_size") or "",
                    hit.get("file_state") or "",
                ]
            )


def _summarize(hits: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for hit in hits:
        value = hit.get(key)
        if isinstance(value, list):
            for item in value:
                token = str(item)
                counts[token] = counts.get(token, 0) + 1
        elif value is not None:
            token = str(value)
            counts[token] = counts.get(token, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate gdc-client manifests for TCGA projects.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    svs = subparsers.add_parser("svs", help="Generate manifest for SVS whole-slide images.")
    svs.add_argument("--project-id", required=True, help="e.g., TCGA-COAD")
    svs.add_argument("--out", required=True, help="Output manifest TSV path.")

    wgs = subparsers.add_parser("wgs-vcf", help="Generate manifest for WGS VCF files (variant calls).")
    wgs.add_argument("--project-id", required=True, help="e.g., TCGA-COAD")
    wgs.add_argument("--out", required=True, help="Output manifest TSV path.")
    wgs.add_argument("--data-category", default="Simple Nucleotide Variation")
    wgs.add_argument(
        "--data-type",
        action="append",
        default=["Annotated Somatic Mutation", "Raw Simple Somatic Mutation"],
        help="Repeatable. Defaults cover common GDC variant-call types.",
    )
    wgs.add_argument(
        "--workflow-type",
        action="append",
        default=[],
        help="Repeatable. Optional analysis.workflow_type filter (e.g., 'MuTect2 Annotation').",
    )
    wgs.add_argument(
        "--reference-genome",
        action="append",
        default=[],
        help="Repeatable. Optional reference_genome filter (e.g., GRCh38).",
    )
    wgs.add_argument(
        "--experimental-strategy",
        default="WGS",
        help="Optional experimental_strategy filter (default: WGS). Use empty string to disable.",
    )

    parser.add_argument("--page-size", type=int, default=2000)
    parser.add_argument("--print-summary", action="store_true", help="Print basic metadata histograms.")
    args = parser.parse_args()

    if args.command == "svs":
        filters = _build_svs_filters(args.project_id)
        cfg = QueryConfig(project_id=args.project_id, filters=filters)
    else:
        filters = _build_wgs_vcf_filters(
            args.project_id,
            data_category=args.data_category,
            data_types=list(args.data_type or []),
            workflow_types=list(args.workflow_type or []),
            reference_genomes=list(args.reference_genome or []),
            experimental_strategy=str(args.experimental_strategy or ""),
        )
        cfg = QueryConfig(project_id=args.project_id, filters=filters)

    hits = _query_all(cfg, page_size=int(args.page_size))
    if not hits:
        raise SystemExit(
            f"No files returned for {cfg.project_id} ({args.command}). "
            "Adjust filters (data_type, workflow_type, experimental_strategy)."
        )

    out_path = Path(args.out).expanduser()
    _write_gdc_manifest(hits, out_path)
    print(f"Wrote GDC manifest -> {out_path} (files={len(hits)})")

    if args.print_summary:
        for key in ("data_type", "analysis.workflow_type", "experimental_strategy", "data_category", "data_format"):
            counts = _summarize(hits, key)
            if not counts:
                continue
            top = list(counts.items())[:10]
            print(f"\nSummary: {key} (top 10)")
            for name, count in top:
                print(f"  {name}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

