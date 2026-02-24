#!/usr/bin/env python3
"""
Generate GDC manifest TSVs for a TCGA project via the GDC API.

This is a minimal, public replacement for internal helpers like
`TCGA_api_query_svs.py`.

The output is a *gdc-client* compatible manifest:
  id <tab> filename <tab> md5 <tab> size <tab> state

Usage:
  python targets/tcga/gdc_generate_manifest.py svs --project-id TCGA-COAD --out tcga_coad_svs_manifest.tsv
  python targets/tcga/gdc_generate_manifest.py wgs-vcf --project-id TCGA-COAD --out tcga_coad_wgs_vcf_manifest.tsv
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from goldmark.targets.gdc_manifest import main  # noqa: E402


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

