from __future__ import annotations

import re
from typing import Iterable, Optional


RESERVED_ENCODER_NAMES = {
    "features",
    "tiles",
    "checkpoints",
    "training",
    "inference",
    "manifests",
    "thumbnails",
}


def sanitize_encoder_dir_name(raw: Optional[str]) -> Optional[str]:
    """Return a filesystem-safe version of the provided encoder label."""

    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None

    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", value)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_-.")
    if not sanitized:
        return None
    if sanitized.lower() in RESERVED_ENCODER_NAMES:
        sanitized = f"{sanitized}_encoder"
    return sanitized


def _candidate_variants(raw: Optional[str]) -> Iterable[str]:
    if raw is None:
        return []
    value = str(raw).strip()
    if not value:
        return []

    variants = []
    if "·" in value:
        variants.append(value.split("·", 1)[0].strip())
    if "|" in value:
        variants.append(value.split("|", 1)[0].strip())
    if ":" in value:
        before, after = value.split(":", 1)
        if after.strip():
            variants.append(after.strip())
        if before.strip():
            variants.append(before.strip())
    if "@" in value:
        variants.append(value.split("@", 1)[0].strip())
    if value.endswith(".pth"):
        variants.append(value[: -4].strip())
    variants.append(value)
    return [variant for variant in variants if variant]


def derive_encoder_dir_name(
    *,
    preferred: Optional[str] = None,
    display: Optional[str] = None,
    custom_encoder: Optional[str] = None,
    source: Optional[str] = None,
    encoder: Optional[str] = None,
) -> str:
    """Pick a readable directory name for storing features from a specific encoder."""

    candidates: list[str] = []
    for raw in [
        preferred,
        display,
        custom_encoder,
        source,
        encoder,
    ]:
        candidates.extend(_candidate_variants(raw))

    for candidate in candidates:
        sanitized = sanitize_encoder_dir_name(candidate)
        if sanitized:
            return sanitized
    return "features_encoder"
