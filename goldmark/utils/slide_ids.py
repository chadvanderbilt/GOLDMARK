from __future__ import annotations

import re

_DMP_ID_PATTERN = re.compile(r"^P-\d{7}-T\d{2}-IM\d+[A-Z]?$", re.IGNORECASE)


def canonicalize_slide_id(raw_id: str) -> str:
    """
    Produce a canonical identifier for artifact generation.

    IMPACT slides use DMP assay identifiers (P-XXXXXXX-TXX-IMX). Historical
    feature exports prefix these identifiers with ``img`` before writing tile
    manifests or feature tensors (e.g., ``features_imgP-0000123-T01-IM3.pt``).
    Returning the prefixed form keeps new exports aligned with the existing
    convention so downstream tooling (especially pre-generated feature caches)
    behaves consistently.
    """

    if raw_id is None:
        return ""

    slide_id = str(raw_id).strip()
    if not slide_id:
        return slide_id

    lowered = slide_id.lower()
    if lowered.startswith("img"):
        return slide_id

    if _DMP_ID_PATTERN.match(slide_id):
        return f"img{slide_id}"

    return slide_id
