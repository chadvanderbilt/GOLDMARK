#!/usr/bin/env python3
"""End-to-end TCGA CV → external cohort run for GOLDMARK.

This script runs what the manuscript pipeline needs to prove it works:

1) Download a small set of real TCGA-LUAD diagnostic slides from GDC (SVS)
2) Derive *real* Positive/Negative labels for KRAS using per-case mutation calls
   (Masked Somatic Mutation MAF from GDC)
3) Create 5 unique 70/30 train/test splits saved in a versioned split manifest
   - For best-epoch selection, the script also carves a tiny per-split validation
     subset out of the training partition (1 positive + 1 negative when possible).
4) Tile → extract features → train for N epochs (default: 10) across the 5 splits
5) Run held-out test inference for each split with attention export + ROC/PR plot
6) Pick the best split (by test ROC AUC) and run external inference on a
   user-provided external manifest (attention export + ROC/PR plot).

Notes
-----
- Default TCGA slide filter is diagnostic/FFPE-style barcodes containing `-00-DX`.
- Requires tokens in `configs/secrets.env` (auto-loaded) for controlled GDC calls.
- For encoder consistency with your external feature cache, set `--encoder`
  to match the external feature directory.

Example
-------
/data1/vanderbc/vanderbc/anaconda3/bin/python scripts/tcga_cv_to_external_full_run.py \\
  --run-name TCGA-LUAD \\
  --device cuda
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from goldmark.cli import main as goldmark_main  # noqa: E402
from goldmark.inference import InferenceConfig, InferenceRunner  # noqa: E402
from goldmark.utils.secrets import load_secrets_env  # noqa: E402
from goldmark.utils.slide_ids import canonicalize_slide_id  # noqa: E402


@dataclass(frozen=True)
class _ManifestRow:
    file_id: str
    filename: str
    md5: str
    size: int
    state: str
    barcode: str
    patient_id: str
    sample_type: Optional[int]
    label_index: int


@dataclass(frozen=True)
class _GdcFileRow:
    file_id: str
    filename: str
    md5: str
    size: int
    state: str = ""


def _run_goldmark(argv: List[str]) -> None:
    print("\n==> goldmark " + " ".join(argv))
    goldmark_main(argv)


def _barcode_from_filename(name: str) -> str:
    token = Path(str(name)).name
    if token.lower().endswith(".svs"):
        token = token[:-4]
    return token.split(".", 1)[0]


def _is_dx_slide(barcode: str) -> bool:
    token = str(barcode or "").strip().upper()
    return "-00-DX" in token


def _parse_sample_type(barcode: str) -> Optional[int]:
    parts = str(barcode).split("-")
    if len(parts) < 4:
        return None
    sample_part = parts[3]
    digits = sample_part[:2]
    if not digits.isdigit():
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _load_gdc_manifest(path: Path) -> List[_ManifestRow]:
    rows: List[_ManifestRow] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"id", "filename", "md5", "size", "state"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Unexpected manifest header in {path}. Expected columns: {sorted(required)}")
        for raw in reader:
            file_id = (raw.get("id") or "").strip()
            filename = (raw.get("filename") or "").strip()
            if not file_id or not filename:
                continue
            size_raw = (raw.get("size") or "").strip()
            try:
                size = int(float(size_raw)) if size_raw else 0
            except ValueError:
                size = 0
            barcode = _barcode_from_filename(filename)
            patient_id = barcode[:12]
            sample_type = _parse_sample_type(barcode)
            rows.append(
                _ManifestRow(
                    file_id=file_id,
                    filename=filename,
                    md5=(raw.get("md5") or "").strip(),
                    size=size,
                    state=(raw.get("state") or "").strip(),
                    barcode=barcode,
                    patient_id=patient_id,
                    sample_type=sample_type,
                    label_index=-1,
                )
            )
    return rows


def _gdc_in_filter(field: str, values: Sequence[str]) -> Dict[str, Any]:
    values = [v for v in (str(x).strip() for x in values) if v]
    return {"op": "in", "content": {"field": field, "value": values}}


def _gdc_and_filter(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {"op": "and", "content": [item for item in items if item]}


def _mpp_slug(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _mpp_label(value: float) -> str:
    if abs(float(value) - 0.5) < 1e-6:
        return "20x"
    if abs(float(value) - 0.25) < 1e-6:
        return "40x"
    return f"mpp_{_mpp_slug(value)}"


def _available_mpp_targets(target_mpp: float, extra_target_mpp: str) -> List[float]:
    targets = [float(target_mpp)]
    for value in _parse_extra_target_mpp(extra_target_mpp):
        targets.append(float(value))
    unique: List[float] = []
    for value in targets:
        if not any(abs(value - existing) < 1e-6 for existing in unique):
            unique.append(value)
    return sorted(unique, reverse=True)


def _choose_mpp_bucket(mpp_value: Optional[float], targets: Sequence[float]) -> float:
    if not targets:
        raise ValueError("No target MPP values provided.")
    if mpp_value is None or mpp_value <= 0:
        return max(targets)
    return min(targets, key=lambda target: abs(float(mpp_value) - float(target)))


def _scale_tile_size(base_tile_size: int, base_mpp: float, target_mpp: float) -> int:
    if target_mpp <= 0:
        return int(base_tile_size)
    scaled = int(round(float(base_tile_size) * float(base_mpp) / float(target_mpp)))
    return max(1, scaled)


def _read_slide_mpp(slide_path: Path) -> Tuple[Optional[float], Optional[float]]:
    try:
        import openslide  # type: ignore
    except Exception:
        return None, None
    try:
        slide = openslide.OpenSlide(str(slide_path))
    except Exception:
        return None, None
    try:
        mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X, "")
        mpp_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, "")
    finally:
        slide.close()
    try:
        mpp_x_val = float(mpp_x) if mpp_x not in ("", None) else None
    except (TypeError, ValueError):
        mpp_x_val = None
    try:
        mpp_y_val = float(mpp_y) if mpp_y not in ("", None) else None
    except (TypeError, ValueError):
        mpp_y_val = None
    return mpp_x_val, mpp_y_val


def _ensure_slide_mpp_columns(slide_manifest: Path) -> "pd.DataFrame":
    import pandas as pd

    df = pd.read_csv(slide_manifest)
    if "slide_path" not in df.columns:
        return df
    needs_mpp = "mpp_x" not in df.columns or "mpp_y" not in df.columns
    if not needs_mpp:
        if df["mpp_x"].isna().any() or df["mpp_y"].isna().any():
            needs_mpp = True
    if not needs_mpp:
        return df
    mpp_x_vals: List[Optional[float]] = []
    mpp_y_vals: List[Optional[float]] = []
    for row in df.itertuples():
        slide_path = Path(getattr(row, "slide_path"))
        mpp_x, mpp_y = _read_slide_mpp(slide_path)
        mpp_x_vals.append(mpp_x)
        mpp_y_vals.append(mpp_y)
    df["mpp_x"] = mpp_x_vals
    df["mpp_y"] = mpp_y_vals
    df.to_csv(slide_manifest, index=False)
    return df


def _parse_extra_target_mpp(raw: str) -> List[float]:
    raw = str(raw or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    values: List[float] = []
    for part in parts:
        value = float(part)
        if value <= 0:
            raise ValueError(f"Invalid target-mpp value: {part}")
        values.append(value)
    return values


def _mpp_tiles_dir(run_dir: Path, value: float) -> Path:
    return run_dir / "tiling" / f"tiles_{_mpp_label(value)}"


def _ensure_alias_dir(alias: Path, target: Path, *, label: str) -> None:
    if alias.exists() or alias.is_symlink():
        return
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError:
        alias.mkdir(parents=True, exist_ok=True)
        (alias / "LOCATION.txt").write_text(f"{label} -> {target}\n")


def _ensure_target_checkpoint_root(
    base_root: Path,
    target_root: Path,
    encoder_name: Optional[str] = None,
    aggregator_name: Optional[str] = None,
) -> Path:
    base_root.mkdir(parents=True, exist_ok=True)
    target_root.mkdir(parents=True, exist_ok=True)
    encoder_root = target_root
    if encoder_name:
        safe_encoder = re.sub(r"[^0-9A-Za-z._-]+", "_", str(encoder_name)).strip("_-.")
        if safe_encoder:
            encoder_root = target_root / safe_encoder
            encoder_root.mkdir(parents=True, exist_ok=True)
    aggregator_root = encoder_root
    if aggregator_name:
        safe_aggregator = re.sub(r"[^0-9A-Za-z._-]+", "_", str(aggregator_name)).strip("_-.")
        if safe_aggregator:
            aggregator_root = encoder_root / safe_aggregator
            aggregator_root.mkdir(parents=True, exist_ok=True)

    def _collect_moves(root: Path) -> List[Path]:
        moves: List[Path] = []
        if not root.exists():
            return moves
        for entry in root.iterdir():
            if entry.name in {target_root.name, encoder_root.name, aggregator_root.name}:
                continue
            if entry.name.startswith("split_") or entry.name == "classification_report":
                moves.append(entry)
            elif entry.is_file():
                moves.append(entry)
        return moves

    move_candidates = _collect_moves(base_root) + _collect_moves(encoder_root)
    for entry in move_candidates:
        dest = aggregator_root / entry.name
        if dest.exists():
            continue
        shutil.move(str(entry), str(dest))
    return aggregator_root


def _download_is_complete(file_id: str, filename: str, size: int, download_root: Path) -> bool:
    path = download_root / file_id / filename
    if not path.exists():
        return False
    try:
        existing_size = path.stat().st_size
    except OSError:
        return False
    if existing_size <= 0:
        return False
    if size and existing_size < int(size):
        return False
    return True


def _md5_matches(path: Path, expected_md5: Optional[str]) -> bool:
    expected = str(expected_md5 or "").strip().lower()
    if not expected:
        return True
    md5 = hashlib.md5()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                md5.update(chunk)
    except OSError:
        return False
    digest = md5.hexdigest()
    if digest != expected:
        print(f"[gdc] MD5 mismatch for {path.name}: expected={expected} got={digest}")
        return False
    return True


def _find_last_epoch_checkpoint(checkpoint_dir: Path) -> Optional[int]:
    if not checkpoint_dir.exists():
        return None
    epoch_candidates: List[int] = []
    for suffix in ("pt", "pth"):
        for path in checkpoint_dir.glob(f"checkpoint_epoch_*.{suffix}"):
            match = re.search(r"checkpoint_epoch_(\d+)\." + re.escape(suffix), path.name)
            if match:
                try:
                    epoch_candidates.append(int(match.group(1)))
                except ValueError:
                    continue
    return max(epoch_candidates) if epoch_candidates else None


def _download_is_complete_with_md5(
    file_id: str,
    filename: str,
    size: int,
    download_root: Path,
    expected_md5: Optional[str],
) -> bool:
    path = download_root / file_id / filename
    if not _download_is_complete(file_id, filename, size, download_root):
        return False
    return _md5_matches(path, expected_md5)


def _tile_coords_path(run_dir: Path, value: float) -> Path:
    label = _mpp_label(value)
    if label in {"20x", "40x"}:
        return run_dir / "tiling" / f"tile_coords_{label}.csv"
    return run_dir / "tiling" / f"tile_coords_{label}.csv"


def _rel_slide_path(raw: str, run_dir: Path) -> str:
    if not raw:
        return ""
    value = str(raw).strip()
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        return value
    try:
        rel = path.resolve().relative_to(run_dir.resolve())
    except Exception:
        return value
    return f"./{rel.as_posix()}"


def _load_slide_index(slide_manifest: Path, run_dir: Path, project_id: str) -> Tuple[Dict[str, Dict[str, str]], List[Dict[str, str]]]:
    mapping: Dict[str, Dict[str, str]] = {}
    manifest_rows: List[Dict[str, str]] = []
    with slide_manifest.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_slide_id = (row.get("slide_id") or "").strip()
            if not raw_slide_id:
                continue
            slide_id = canonicalize_slide_id(raw_slide_id)
            patient_id = (row.get("patient_id") or "").strip()
            sample_id = f"{patient_id}_{slide_id}" if patient_id else slide_id
            slide_path = _rel_slide_path(row.get("slide_path") or "", run_dir)
            file_name = Path(row.get("slide_path") or "").name if row.get("slide_path") else ""
            manifest_rows.append(
                {
                    "file_name": file_name,
                    "sample_id": sample_id,
                    "target": str(project_id),
                    "slide_path": slide_path,
                }
            )
            mapping[slide_id] = {"sample_id": sample_id, "slide": slide_path}
    return mapping, manifest_rows


def _write_tiling_manifest(run_dir: Path, rows: List[Dict[str, str]], *, resume: bool) -> Path:
    out_path = run_dir / "tiling" / "tiling_manifest.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing_rows: List[Dict[str, str]] = []
        existing_ids: set[str] = set()
        with out_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                sample_id = (row.get("sample_id") or "").strip()
                if sample_id:
                    existing_ids.add(sample_id)
                existing_rows.append(row)
        added = [row for row in rows if (row.get("sample_id") or "").strip() not in existing_ids]
        if not added:
            return out_path
        with out_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["file_name", "sample_id", "target", "slide_path"])
            writer.writeheader()
            writer.writerows(existing_rows + added)
        return out_path
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_name", "sample_id", "target", "slide_path"])
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def _write_tile_coords(
    tile_dir: Path,
    out_path: Path,
    slide_index: Dict[str, Dict[str, str]],
    *,
    project_id: str,
    resume: bool,
) -> None:
    manifest_dir = tile_dir / "manifests"
    if not manifest_dir.exists():
        return
    if resume and out_path.exists() and out_path.stat().st_size > 0:
        print(f"[resume] Tile coords already present: {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "y", "slide", "sample_id", "target"])
        writer.writeheader()
        for manifest_path in sorted(manifest_dir.glob("*_tiles.csv")):
            slide_id = manifest_path.name.replace("_tiles.csv", "")
            info = slide_index.get(slide_id, {})
            sample_id = info.get("sample_id", slide_id)
            slide_path = info.get("slide", "")
            with manifest_path.open("r", newline="") as in_handle:
                reader = csv.DictReader(in_handle)
                for row in reader:
                    writer.writerow(
                        {
                            "x": row.get("x"),
                            "y": row.get("y"),
                            "slide": slide_path,
                            "sample_id": sample_id,
                            "target": str(project_id),
                        }
                    )


def _write_bucketed_split_manifests(
    split_manifest: Path,
    slide_df: "pd.DataFrame",
    run_dir: Path,
    targets: Sequence[float],
) -> Dict[str, Path]:
    import pandas as pd

    split_df = pd.read_csv(split_manifest)
    if "slide_id" not in split_df.columns:
        return {}
    bucket_map = {
        str(getattr(row, "slide_id")): str(getattr(row, "bucket_label"))
        for row in slide_df.itertuples()
    }
    default_label = _mpp_label(max(targets))
    split_df["bucket_label"] = split_df["slide_id"].map(bucket_map).fillna(default_label)
    out_dir = run_dir / "tiling"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}
    for target_mpp in targets:
        label = _mpp_label(target_mpp)
        subset = split_df[split_df["bucket_label"] == label].drop(columns=["bucket_label"])
        out_path = out_dir / f"split_manifest_{label}.csv"
        subset.to_csv(out_path, index=False)
        outputs[label] = out_path
    return outputs


def _manifest_has_rows(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            header = handle.readline()
            if not header:
                return False
            for line in handle:
                if line.strip():
                    return True
    except OSError:
        return False
    return False


def _tile_manifests_present(tile_dir: Path) -> bool:
    manifest_dir = tile_dir / "manifests"
    if not manifest_dir.exists():
        return False
    return any(manifest_dir.glob("*_tiles.csv"))
def _case_column_from_manifest(df) -> Optional[str]:
    for col in ("patient_id", "case_id", "group_id", "sample_id"):
        if col in df.columns:
            return col
    return None


def _safe_case_slug(case_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(case_id or "").strip())
    return slug.strip("_") or "case"


def _write_case_tile_manifests(tile_dir: Path, slide_manifest: Path, *, resume: bool) -> None:
    import pandas as pd

    manifest_dir = tile_dir / "manifests"
    if not manifest_dir.exists():
        return

    case_dir = tile_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    if resume and any(case_dir.glob("*_tiles.csv")):
        print(f"[resume] Case-level manifests already present: {case_dir}")
        return

    slide_df = pd.read_csv(slide_manifest)
    if "slide_id" not in slide_df.columns:
        raise ValueError(f"slide_manifest missing slide_id column: {slide_manifest}")
    case_col = _case_column_from_manifest(slide_df)
    if not case_col:
        print("[warn] No case column found in slide manifest; falling back to slide_id for case_id.")

    slide_to_case: Dict[str, str] = {}
    for row in slide_df.itertuples():
        raw_slide_id = getattr(row, "slide_id")
        slide_id = canonicalize_slide_id(raw_slide_id)
        case_id = slide_id
        if case_col:
            case_value = getattr(row, case_col)
            if case_value is not None and str(case_value).strip():
                case_id = str(case_value).strip()
        slide_to_case[slide_id] = case_id

    fieldnames = [
        "case_id",
        "slide_id",
        "tile_id",
        "x",
        "y",
        "level",
        "width",
        "height",
        "tissue_fraction",
        "tile_path",
    ]

    case_index: Dict[str, Path] = {}
    for slide_manifest_path in sorted(manifest_dir.glob("*_tiles.csv")):
        slide_id = slide_manifest_path.name.replace("_tiles.csv", "")
        case_id = slide_to_case.get(slide_id, slide_id)
        case_slug = _safe_case_slug(case_id)
        case_manifest_path = case_dir / f"{case_slug}_tiles.csv"
        write_header = not case_manifest_path.exists()
        case_index[case_id] = case_manifest_path

        with slide_manifest_path.open("r", newline="") as handle_in, case_manifest_path.open("a", newline="") as handle_out:
            reader = csv.DictReader(handle_in)
            writer = csv.DictWriter(handle_out, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in reader:
                writer.writerow(
                    {
                        "case_id": case_id,
                        "slide_id": slide_id,
                        "tile_id": row.get("tile_id"),
                        "x": row.get("x"),
                        "y": row.get("y"),
                        "level": row.get("level"),
                        "width": row.get("width"),
                        "height": row.get("height"),
                        "tissue_fraction": row.get("tissue_fraction"),
                        "tile_path": row.get("tile_path"),
                    }
                )

    index_path = case_dir / "cases_index.csv"
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "case_manifest"])
        writer.writeheader()
        for case_id, path in sorted(case_index.items()):
            writer.writerow({"case_id": case_id, "case_manifest": str(path)})
    print(f"[cases] Case-level manifests written under {case_dir}")


def _query_case_maf_file(
    *,
    project_id: str,
    patient_id: str,
    experimental_strategy: str = "WXS",
) -> Optional[Dict[str, Any]]:
    """Return a single best-hit MAF file record for a TCGA case (submitter_id)."""
    import requests

    filters = _gdc_and_filter(
        [
            _gdc_in_filter("cases.project.project_id", [project_id]),
            _gdc_in_filter("cases.submitter_id", [patient_id]),
            _gdc_in_filter("data_category", ["Simple Nucleotide Variation"]),
            _gdc_in_filter("data_type", ["Masked Somatic Mutation"]),
            _gdc_in_filter("data_format", ["MAF"]),
            _gdc_in_filter("experimental_strategy", [experimental_strategy] if experimental_strategy else []),
        ]
    )
    fields = [
        "file_id",
        "file_name",
        "md5sum",
        "file_size",
        "file_state",
        "data_type",
        "data_format",
        "experimental_strategy",
        "analysis.workflow_type",
        "cases.submitter_id",
    ]
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": "50",
    }
    resp = requests.get("https://api.gdc.cancer.gov/files", params=params, timeout=60)
    resp.raise_for_status()
    hits = (resp.json().get("data") or {}).get("hits") or []
    if not hits:
        return None

    def _priority(hit: Dict[str, Any]) -> Tuple[int, int]:
        name = str(hit.get("file_name") or "").lower()
        workflow = (hit.get("analysis") or {}).get("workflow_type") if isinstance(hit.get("analysis"), dict) else None
        workflow = str(workflow or "").lower()
        rank = 10
        if "aliquot ensemble somatic variant merging and masking" in workflow:
            rank = 0
        elif "aliquot_ensemble_masked" in name or "ensemble_masked" in name:
            rank = 1
        size = int(hit.get("file_size") or 0)
        return (rank, -size)

    hits.sort(key=_priority)
    return hits[0]


def _maf_has_gene_variant(path: Path, gene: str) -> bool:
    """Return True if the MAF contains a non-silent variant for `gene`."""
    import gzip

    gene_upper = str(gene).strip().upper()
    if not gene_upper:
        raise ValueError("gene is required")

    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        header = None
        for line in handle:
            if not line or line.startswith("#"):
                continue
            header = line.rstrip("\n").split("\t")
            break
        if not header:
            return False
        col_map = {name: idx for idx, name in enumerate(header)}
        hugo_idx = col_map.get("Hugo_Symbol")
        vc_idx = col_map.get("Variant_Classification")
        if hugo_idx is None:
            return False
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if hugo_idx >= len(parts):
                continue
            if parts[hugo_idx].strip().upper() != gene_upper:
                continue
            if vc_idx is not None and vc_idx < len(parts):
                vc = parts[vc_idx].strip().lower()
                if vc in {"silent", "synonymous", "synonymous_variant"}:
                    continue
            return True
    return False


def _read_gdc_token(token_file: Optional[str]) -> Optional[str]:
    if not token_file:
        return None
    path = Path(str(token_file)).expanduser()
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        for key in ("token", "access_token", "gdc_token"):
            value = payload.get(key)
            if value:
                return str(value).strip() or None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        return line
    return None


def _download_with_gdc_api(
    *,
    rows: List[_GdcFileRow],
    out_dir: Path,
    token_file: Optional[str],
    retry_amount: int = 3,
    force_downloads: bool = False,
) -> None:
    """Download files directly from the GDC API (fallback when gdc-client is unavailable)."""
    import requests

    token = _read_gdc_token(token_file)
    headers = {}
    if token:
        headers["X-Auth-Token"] = token

    out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        target_dir = out_dir / row.file_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / row.filename
        if not force_downloads and _download_is_complete_with_md5(
            row.file_id,
            row.filename,
            row.size,
            out_dir,
            row.md5,
        ):
            continue
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        if force_downloads:
            if target_path.exists():
                target_path.unlink()
            if tmp_path.exists():
                tmp_path.unlink()
        if target_path.exists() and target_path.stat().st_size > 0:
            existing_size = target_path.stat().st_size
            if row.size and existing_size >= row.size and _md5_matches(target_path, row.md5):
                continue
            if row.size and existing_size >= row.size and row.md5 and not _md5_matches(target_path, row.md5):
                target_path.unlink()
                if tmp_path.exists():
                    tmp_path.unlink()
                existing_size = 0
            if not tmp_path.exists():
                target_path.replace(tmp_path)
            else:
                tmp_size = tmp_path.stat().st_size
                if existing_size > tmp_size:
                    tmp_path.unlink()
                    target_path.replace(tmp_path)
                else:
                    target_path.unlink()
        if tmp_path.exists() and row.size and tmp_path.stat().st_size >= row.size:
            tmp_path.replace(target_path)
            continue

        url = f"https://api.gdc.cancer.gov/data/{row.file_id}"
        attempts = max(1, int(retry_amount) if retry_amount else 1)
        for attempt in range(1, attempts + 1):
            resume_from = tmp_path.stat().st_size if tmp_path.exists() else 0
            if row.size and resume_from >= row.size:
                tmp_path.replace(target_path)
                break
            req_headers = dict(headers)
            if resume_from > 0:
                req_headers["Range"] = f"bytes={resume_from}-"
            size_label = f"{row.size / (1024**2):.1f} MB" if row.size else "unknown size"
            print(
                f"[gdc-api] Downloading {row.file_id} -> {target_path.name} "
                f"({size_label})"
                + (f" [resume@{resume_from}]" if resume_from else "")
            )
            md5 = hashlib.md5()
            if resume_from > 0:
                with tmp_path.open("rb") as existing_handle:
                    for chunk in iter(lambda: existing_handle.read(1024 * 1024), b""):
                        md5.update(chunk)
            try:
                while True:
                    with requests.get(url, headers=req_headers, stream=True, timeout=(30, 600)) as resp:
                        if resume_from > 0 and resp.status_code == 200:
                            # Server ignored Range; restart from scratch.
                            resume_from = 0
                            req_headers = dict(headers)
                            md5 = hashlib.md5()
                            if tmp_path.exists():
                                tmp_path.unlink()
                            continue
                        if resp.status_code == 416 and resume_from > 0:
                            break
                        resp.raise_for_status()
                        mode = "ab" if resume_from > 0 else "wb"
                        with tmp_path.open(mode) as handle:
                            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                                if not chunk:
                                    continue
                                handle.write(chunk)
                                md5.update(chunk)
                    break
                if row.size and tmp_path.stat().st_size < row.size:
                    raise IOError(
                        f"Incomplete download for {row.file_id}: "
                        f"{tmp_path.stat().st_size} < {row.size}"
                    )
                tmp_path.replace(target_path)
                if row.md5:
                    digest = md5.hexdigest()
                    if digest != row.md5:
                        raise ValueError(f"MD5 mismatch for {row.file_id}: expected={row.md5} got={digest}")
                break
            except Exception as exc:  # noqa: BLE001
                if attempt >= attempts:
                    raise
                wait = min(60, 5 * attempt)
                print(f"[gdc-api] Retry {attempt}/{attempts} after error: {exc}. Sleeping {wait}s...")
                time.sleep(wait)


def _write_filtered_gdc_manifest(rows: List[_ManifestRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "filename", "md5", "size", "state"])
        for row in rows:
            writer.writerow([row.file_id, row.filename, row.md5, str(int(row.size)), row.state])


def _ensure_gdc_client(path_hint: Optional[str]) -> Path:
    if path_hint:
        candidate = Path(path_hint).expanduser()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"--gdc-client not found: {candidate}")

    local_bin = REPO_ROOT / "bin" / "gdc-client"
    if local_bin.exists():
        return local_bin
    local_bin_exe = REPO_ROOT / "bin" / "gdc-client.exe"
    if local_bin_exe.exists():
        return local_bin_exe

    if shutil.which("gdc-client"):
        return Path("gdc-client")

    installer = REPO_ROOT / "scripts" / "install_gdc_client.py"
    if not installer.exists():
        raise FileNotFoundError(f"Expected installer script not found: {installer}")
    print("[gdc-client] Not found; installing into bin/ ...")
    subprocess.run([sys.executable, str(installer), "--dest", str(local_bin)], check=True)
    if local_bin.exists():
        return local_bin
    if local_bin_exe.exists():
        return local_bin_exe
    raise RuntimeError("gdc-client install completed but binary not found in bin/.")


def _gdc_client_usable(path: Path) -> bool:
    try:
        result = subprocess.run([str(path), "--version"], check=False, capture_output=True, text=True)
    except OSError:
        return False
    if result.returncode == 0:
        return True
    stderr = (result.stderr or "").strip()
    if stderr:
        print(f"[gdc-client] Unusable: {stderr}", file=sys.stderr)
    return False


def _download_with_gdc_client(
    *,
    gdc_client: Path,
    manifest_path: Path,
    out_dir: Path,
    token: Optional[str],
    retry_amount: int,
) -> None:
    cmd = [
        "bash",
        str(REPO_ROOT / "targets" / "tcga" / "gdc_download.sh"),
        "--manifest",
        str(manifest_path),
        "--out",
        str(out_dir),
        "--gdc-client",
        str(gdc_client),
        "--retry-amount",
        str(int(retry_amount)),
    ]
    if token:
        cmd.extend(["--token", token])
    print("\n==> " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _resolve_downloaded_svs(download_root: Path, row: _ManifestRow) -> Path:
    direct = download_root / row.file_id / row.filename
    if direct.exists():
        return direct
    file_dir = download_root / row.file_id
    if file_dir.exists():
        svs = sorted(file_dir.glob("*.svs"))
        if len(svs) == 1:
            return svs[0]
    candidates = sorted(download_root.rglob(row.filename))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Downloaded SVS not found for {row.file_id} ({row.filename}) under {download_root}")


def _validate_svs_with_openslide(path: Path) -> bool:
    try:
        import openslide  # type: ignore
    except Exception:
        return True
    try:
        slide = openslide.OpenSlide(str(path))
        slide.close()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[svs] OpenSlide failed for {path}: {exc}")
        return False


def _ensure_valid_svs(
    row: _ManifestRow,
    *,
    download_root: Path,
    token_file: Optional[str],
    retry_amount: int,
    force_downloads: bool,
) -> Path:
    try:
        path = _resolve_downloaded_svs(download_root, row)
    except FileNotFoundError:
        path = None
    if path is not None:
        if not _md5_matches(path, row.md5):
            reason = "md5 mismatch"
        elif _validate_svs_with_openslide(path):
            return path
        else:
            reason = "OpenSlide failure"
    else:
        reason = "missing file"
    print(f"[svs] Re-downloading {row.file_id} ({row.filename}) due to {reason}.")
    if path is not None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        if path.exists():
            path.unlink()
        if tmp_path.exists():
            tmp_path.unlink()
    _download_with_gdc_api(
        rows=[
            _GdcFileRow(
                file_id=row.file_id,
                filename=row.filename,
                md5=row.md5,
                size=row.size,
                state=row.state,
            )
        ],
        out_dir=download_root,
        token_file=token_file,
        retry_amount=retry_amount,
        force_downloads=force_downloads,
    )
    path = _resolve_downloaded_svs(download_root, row)
    if not _md5_matches(path, row.md5) or not _validate_svs_with_openslide(path):
        raise ValueError(f"SVS download invalid after re-download: {path}")
    return path


def _select_gene_subset(
    rows: List[_ManifestRow],
    *,
    project_id: str,
    gene: str,
    per_class: int,
    maf_download_dir: Path,
    token_file: Optional[str],
    retry_amount: int = 3,
    force_downloads: bool = False,
    experimental_strategy: str = "WXS",
) -> List[_ManifestRow]:
    """Select `per_class` positive/negative patients by downloading/parsing per-case MAFs."""
    per_class = int(per_class)
    gene_upper = str(gene).strip().upper()
    if not gene_upper:
        raise ValueError("--gene is required when deriving mutation labels")
    full_cohort = per_class <= 0
    if not full_cohort:
        per_class = max(1, per_class)

    by_patient: Dict[str, List[_ManifestRow]] = {}
    for row in rows:
        by_patient.setdefault(row.patient_id, []).append(row)

    selected: List[_ManifestRow] = []
    counts = {0: 0, 1: 0}
    checked = 0
    for patient_id in sorted(by_patient.keys()):
        if not full_cohort and counts[0] >= per_class and counts[1] >= per_class:
            break

        maf_hit = _query_case_maf_file(
            project_id=str(project_id),
            patient_id=str(patient_id),
            experimental_strategy=str(experimental_strategy or ""),
        )
        if not maf_hit:
            continue
        file_id = str(maf_hit.get("file_id") or maf_hit.get("id") or "").strip()
        filename = str(maf_hit.get("file_name") or "").strip()
        if not file_id or not filename:
            continue
        maf_row = _GdcFileRow(
            file_id=file_id,
            filename=filename,
            md5=str(maf_hit.get("md5sum") or "").strip(),
            size=int(maf_hit.get("file_size") or 0),
            state=str(maf_hit.get("file_state") or "").strip(),
        )
        if not force_downloads and _download_is_complete_with_md5(
            file_id,
            filename,
            maf_row.size,
            maf_download_dir,
            maf_row.md5,
        ):
            pass
        else:
            _download_with_gdc_api(
                rows=[maf_row],
                out_dir=maf_download_dir,
                token_file=token_file,
                retry_amount=retry_amount,
                force_downloads=force_downloads,
            )
        maf_path = maf_download_dir / file_id / filename
        if not maf_path.exists() or maf_path.stat().st_size == 0:
            continue

        checked += 1
        is_positive = _maf_has_gene_variant(maf_path, gene_upper)
        label = 1 if is_positive else 0
        if not full_cohort and counts[label] >= per_class:
            continue

        candidates = [r for r in by_patient[patient_id] if r.sample_type is not None and int(r.sample_type) < 10]
        if not candidates:
            candidates = list(by_patient[patient_id])
        slide = sorted(candidates, key=lambda r: (r.size, r.filename))[0]

        selected.append(
            _ManifestRow(
                file_id=slide.file_id,
                filename=slide.filename,
                md5=slide.md5,
                size=slide.size,
                state=slide.state,
                barcode=slide.barcode,
                patient_id=slide.patient_id,
                sample_type=slide.sample_type,
                label_index=int(label),
            )
        )
        counts[label] += 1

        if checked % 10 == 0:
            if full_cohort:
                print(f"[labels] scanned cases={checked} labeled pos={counts[1]} neg={counts[0]}")
            else:
                print(
                    f"[labels] scanned cases={checked} selected pos={counts[1]} neg={counts[0]} "
                    f"(target per_class={per_class})"
                )

    if full_cohort:
        if not selected:
            raise ValueError(f"No labeled cases found for {gene_upper} in {project_id}.")
        if counts[0] == 0 or counts[1] == 0:
            raise ValueError(
                f"Full-cohort labeling found only one class for {gene_upper} in {project_id} "
                f"(pos={counts[1]} neg={counts[0]})."
            )
        return sorted(selected, key=lambda r: (r.label_index, r.size, r.filename))

    if counts[0] < per_class or counts[1] < per_class:
        raise ValueError(
            f"Unable to find enough labeled cases for {gene_upper} in {project_id}. "
            f"Selected pos={counts[1]} neg={counts[0]} (requested {per_class} each)."
        )
    return sorted(selected, key=lambda r: (r.label_index, r.size, r.filename))


def _add_val_assignments(
    df,
    *,
    split_columns: Sequence[str],
    label_column: str,
    val_per_class: int,
    seed: int,
) -> None:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(int(seed))
    labels = pd.to_numeric(df[label_column], errors="coerce").fillna(0).astype(int)
    for split_col in split_columns:
        if split_col not in df.columns:
            raise ValueError(f"Split column missing from manifest: {split_col}")
        train_idx = df.index[df[split_col].astype(str) == "train"].tolist()
        if not train_idx:
            continue
        pos_idx = [i for i in train_idx if int(labels.loc[i]) == 1]
        neg_idx = [i for i in train_idx if int(labels.loc[i]) == 0]
        val_idx: List[int] = []
        if len(pos_idx) >= val_per_class and len(neg_idx) >= val_per_class:
            val_idx.extend(rng.choice(pos_idx, size=val_per_class, replace=False).tolist())
            val_idx.extend(rng.choice(neg_idx, size=val_per_class, replace=False).tolist())
        else:
            take = min(len(train_idx), max(1, val_per_class))
            val_idx.extend(rng.choice(train_idx, size=take, replace=False).tolist())
        df.loc[val_idx, split_col] = "val"


def _split_has_both_classes(df, *, split_col: str, split_value: str, label_column: str) -> bool:
    import pandas as pd

    sub = df.loc[df[split_col].astype(str) == str(split_value)].copy()
    if sub.empty:
        return False
    labels = pd.to_numeric(sub[label_column], errors="coerce").fillna(0).astype(int)
    return len(set(labels.tolist())) >= 2


def _maybe_write_roc_pr_plots(results_csv: Path, out_dir: Path, *, title: str) -> None:
    try:
        import pandas as pd
        from sklearn.metrics import auc, precision_recall_curve, roc_curve, roc_auc_score, average_precision_score
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plots] Skipping ROC/PR plots (missing deps): {exc}")
        return

    df = pd.read_csv(results_csv)
    if df.empty or "target" not in df.columns or "probability" not in df.columns:
        print("[plots] Skipping ROC/PR plots (missing columns).")
        return
    y_true = df["target"].astype(int).values
    y_score = df["probability"].astype(float).values
    if len(set(y_true.tolist())) < 2:
        print("[plots] Skipping ROC/PR plots (single-class truth).")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(rec, prec)
    ap = average_precision_score(y_true, y_score)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(fpr, tpr, linewidth=2.2, label=f"AUC={roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1)
    ax1.set_title("ROC")
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.legend(frameon=False)

    ax2.plot(rec, prec, linewidth=2.2, label=f"PR AUC={pr_auc:.3f} (AP={ap:.3f})")
    ax2.set_title("Precision–Recall")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(frameon=False)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = out_dir / "roc_pr_curves.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[plots] Wrote {out_path}")


def _map_binary_status(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"nan", "none"}:
        return None
    if text in {"positive", "pos", "1", "true", "yes"}:
        return 1
    if text in {"negative", "neg", "0", "false", "no"}:
        return 0
    return None


def _coerce_bool(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def _discover_external_feature(feature_dir: Path, slide_id: str) -> Tuple[Path, Optional[Path]]:
    slide_id = str(slide_id).strip()
    if not slide_id:
        raise ValueError("Missing slide_id for external features")
    candidates = [
        feature_dir / f"features_img{slide_id}.pt",
        feature_dir / f"features_{slide_id}.pt",
        feature_dir / f"features_img{slide_id}.pth",
        feature_dir / f"features_{slide_id}.pth",
    ]
    feature_path = next((p for p in candidates if p.exists()), None)
    if feature_path is None:
        raise FileNotFoundError(
            f"External feature tensor not found for {slide_id} under {feature_dir}. Tried: "
            + ", ".join(str(p.name) for p in candidates)
        )
    meta_path = feature_path.with_suffix(".json")
    return feature_path, meta_path if meta_path.exists() else None


def _infer_slide_path_from_tile_manifest(meta_path: Optional[Path]) -> Optional[str]:
    if not meta_path or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(meta, dict):
        return None
    tile_manifest = meta.get("tile_manifest")
    if not tile_manifest:
        return None
    try:
        import pandas as pd

        tile_df = pd.read_csv(tile_manifest)
    except Exception:  # noqa: BLE001
        return None
    if tile_df.empty or "slide" not in tile_df.columns:
        return None
    value = str(tile_df["slide"].iloc[0]).strip()
    return value or None


def _load_external_manifest_balanced(
    path: Path,
    *,
    gene: str,
    external_root: Optional[Path],
    encoder: str,
    per_class: int,
) -> Tuple[List[Dict[str, object]], Optional[Path]]:
    import pandas as pd

    df = pd.read_csv(path)
    gene_col = str(gene).strip().upper()
    label_series = None
    if gene_col in df.columns:
        label_series = df[gene_col].map(_map_binary_status)
    elif "label_index" in df.columns:
        label_series = pd.to_numeric(df["label_index"], errors="coerce")
    elif "label" in df.columns:
        label_series = df["label"].map(_map_binary_status)
    elif "target" in df.columns:
        label_series = df["target"].map(_map_binary_status)
    if label_series is None:
        raise ValueError(
            f"External manifest missing label column. Provide {gene_col!r} or 'label_index'/'label'."
        )

    if "scanned_slides_exist" in df.columns:
        df = df.loc[df["scanned_slides_exist"].map(_coerce_bool)].copy()
    if df.empty:
        raise ValueError("No external rows selected (scanned_slides_exist filtered everything).")

    feature_dir: Optional[Path] = None
    if external_root:
        candidates: List[Path] = []
        if "OncoTree_Code" in df.columns:
            cohort = str(df["OncoTree_Code"].iloc[0]).strip()
            if cohort:
                candidates.append(external_root / cohort / "features" / str(encoder))
        candidates.append(external_root / "features" / str(encoder))
        candidates.append(external_root / str(encoder))
        for candidate in candidates:
            if candidate.exists():
                feature_dir = candidate
                break

    labeled = df.copy()
    labeled["label_index"] = label_series
    labeled = labeled.dropna(subset=["label_index"]).copy()
    labeled["label_index"] = labeled["label_index"].astype(int)
    if labeled.empty:
        raise ValueError(f"No usable external labels found for {gene_col} (need Positive/Negative).")

    per_class = int(per_class)
    full_cohort = per_class <= 0
    if full_cohort:
        picked = labeled.sample(frac=1.0, random_state=1337).reset_index(drop=True)
    else:
        pos = labeled.loc[labeled["label_index"] == 1]
        neg = labeled.loc[labeled["label_index"] == 0]
        take_pos = min(len(pos), int(per_class))
        take_neg = min(len(neg), int(per_class))
        if take_pos == 0 or take_neg == 0:
            raise ValueError(
                f"External selection needs both classes; found pos={len(pos)} neg={len(neg)} for {gene_col}."
            )
        picked = pd.concat([pos.head(take_pos), neg.head(take_neg)], ignore_index=True)
        picked = picked.sample(frac=1.0, random_state=1337).reset_index(drop=True)

    slide_id_col = None
    for candidate in ("slide_id", "DMP_ASSAY_ID", "sample_id", "case_id", "slide"):
        if candidate in picked.columns:
            slide_id_col = candidate
            break
    if slide_id_col is None:
        raise ValueError("External manifest missing slide identifier column (need slide_id).")

    rows: List[Dict[str, object]] = []
    for _, row in picked.iterrows():
        slide_id = str(row[slide_id_col]).strip()
        if not slide_id:
            continue
        label = int(row["label_index"])
        feature_path_value = str(row.get("feature_path") or "").strip()
        feature_path: Optional[Path] = None
        meta_path: Optional[Path] = None
        if feature_path_value:
            feature_path = Path(feature_path_value).expanduser()
        elif feature_dir is not None:
            feature_path, meta_path = _discover_external_feature(feature_dir, slide_id)
        if feature_path is not None and meta_path is None:
            meta_path = feature_path.with_suffix(".json")
        slide_path = str(row.get("slide_path") or "").strip() or None
        if not slide_path and meta_path:
            slide_path = _infer_slide_path_from_tile_manifest(meta_path)
        if feature_path is None or not feature_path.exists():
            continue
        rows.append(
            {
                "slide_id": slide_id,
                "slide_path": slide_path,
                "label_index": label,
                "split": "external",
                "feature_path": str(feature_path),
            }
        )
    if not rows:
        raise ValueError("No usable external rows found (feature lookup failed for all selected slides).")
    return rows, feature_dir


def _pick_best_split(cv_summary_path: Path) -> str:
    import pandas as pd

    df = pd.read_csv(cv_summary_path)
    if df.empty or "split" not in df.columns:
        raise ValueError(f"Unexpected cv_summary format: {cv_summary_path}")
    score_col = "test_roc_auc" if "test_roc_auc" in df.columns else ("val_roc_auc" if "val_roc_auc" in df.columns else None)
    if not score_col:
        return str(df["split"].iloc[0])
    scores = pd.to_numeric(df[score_col], errors="coerce")
    if scores.notna().any():
        idx = scores.fillna(float("-inf")).idxmax()
        return str(df.loc[idx, "split"])
    return str(df["split"].iloc[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs", help="Pipeline output root (default: runs/)")
    parser.add_argument("--run-name", default="", help="Project folder name under --output (default: --project-id)")
    parser.add_argument("--project-id", default="TCGA-LUAD", help="TCGA project id (default: TCGA-LUAD)")
    parser.add_argument("--gene", default="KRAS", help="Gene to label (default: KRAS)")
    parser.add_argument(
        "--per-class",
        type=int,
        default=0,
        help="Slides per class (default: 0 = full cohort). Use 0 to label and use all available cases (very large).",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience (default: 50)")
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--target-mpp", type=float, default=0.5)
    parser.add_argument(
        "--extra-target-mpp",
        default="0.25",
        help=(
            "Comma-separated target-mpp buckets (default: 0.25 for 40x). "
            "Each slide is assigned to the nearest bucket and tiled once."
        ),
    )
    parser.add_argument("--limit-tiles", type=int, default=0, help="Optional tile cap per slide (default: 0 = no cap).")
    parser.add_argument("--encoder", default="h-optimus-0")
    parser.add_argument("--device", default="auto", help="Feature/training device (default: auto)")
    parser.add_argument("--with-overlays", action="store_true", help="Generate overlays (requires OpenSlide + cv2)")
    parser.add_argument("--no-plots", action="store_true", help="Skip ROC/PR plot generation")
    parser.add_argument("--gdc-client", default=None, help="Path to gdc-client (default: auto-install/bin/ or PATH)")
    parser.add_argument("--token", default=None, help="Optional GDC token file for controlled-access data")
    parser.add_argument("--mutation-strategy", default="WXS", help="Experimental strategy filter for mutation calls")
    parser.add_argument("--retry-amount", type=int, default=20, help="gdc-client retry amount (default: 20)")
    parser.add_argument(
        "--force-downloads",
        action="store_true",
        help="Re-download SVS/MAF files even if already present.",
    )
    parser.add_argument(
        "--secrets-env",
        default="configs/secrets.env",
        help="Optional .env file to load tokens from (default: configs/secrets.env)",
    )
    parser.add_argument(
        "--external-manifest",
        default="",
        help="External cohort manifest CSV (required for external inference).",
    )
    parser.add_argument(
        "--external-root",
        default="",
        help="External root containing features/<encoder> or <cohort>/features/<encoder> (optional if manifest has feature_path).",
    )
    parser.add_argument(
        "--external-per-class",
        type=int,
        default=0,
        help="External cases per class (default: 0 = full cohort). Use 0 to run all labeled cases.",
    )
    parser.add_argument("--allow-non-dx", action="store_true", help="Include non-diagnostic slides (disable -00-DX filter)")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing run directory")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run directory (reuse downloaded SVS/MAF when present).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild tiling/features/training/inference while preserving downloads.",
    )
    args = parser.parse_args()

    if not args.run_name:
        args.run_name = str(args.project_id)

    os.chdir(REPO_ROOT)

    secrets_path = Path(str(args.secrets_env)).expanduser()
    if not secrets_path.is_absolute():
        secrets_path = (REPO_ROOT / secrets_path).resolve()
    if secrets_path.exists():
        load_secrets_env(secrets_path, verbose=True)

    if not args.token:
        args.token = os.environ.get("GDC_TOKEN_FILE") or None

    output_root = Path(args.output).expanduser().resolve()
    run_dir = output_root / str(args.run_name)
    if run_dir.exists():
        if args.force:
            shutil.rmtree(run_dir)
        elif args.rebuild:
            # Preserve downloads + derived labels, but wipe stage outputs.
            for stage in ("tiling", "features", "training", "inference", "external_inference", "plots"):
                target = run_dir / stage
                if target.exists():
                    shutil.rmtree(target)
        elif not args.resume:
            raise FileExistsError(
                f"Run directory already exists: {run_dir} (use --resume to continue, --rebuild to rerun stages, or --force to overwrite)"
            )

    gene_upper = str(args.gene).strip().upper()
    checkpoints_root = run_dir / "training" / "checkpoints"
    target_dir = checkpoints_root / gene_upper
    target_manifest_dir = target_dir / "manifests"
    target_manifest_dir.mkdir(parents=True, exist_ok=True)

    gdc_root = run_dir / "gdc_downloads"
    gdc_root.mkdir(parents=True, exist_ok=True)
    full_manifest = gdc_root / f"{args.project_id}_svs_manifest.tsv"
    subset_manifest = target_manifest_dir / f"{args.project_id}_{gene_upper}_svs_manifest_subset.tsv"
    slide_manifest = target_manifest_dir / f"{args.project_id}_{gene_upper}_slides.csv"
    svs_download_dir = gdc_root / "svs"
    maf_download_dir = gdc_root / "maf"

    if (args.resume or args.rebuild) and slide_manifest.exists():
        print(f"[resume] Using existing slide manifest: {slide_manifest}")
        if subset_manifest.exists():
            svs_download_dir.mkdir(parents=True, exist_ok=True)
            subset_rows = _load_gdc_manifest(subset_manifest)
            for row in subset_rows:
                _ensure_valid_svs(
                    row,
                    download_root=svs_download_dir,
                    token_file=args.token,
                    retry_amount=int(args.retry_amount),
                    force_downloads=bool(args.force or args.force_downloads),
                )
    else:
        gdc_client = _ensure_gdc_client(args.gdc_client)
        print(f"[gdc-client] Using: {gdc_client}")

        _run_goldmark(["gdc-manifest", "svs", "--project-id", str(args.project_id), "--out", str(full_manifest)])

        rows = _load_gdc_manifest(full_manifest)
        if not args.allow_non_dx:
            rows = [row for row in rows if _is_dx_slide(row.barcode)]
        if not rows:
            raise ValueError("No SVS rows found after filtering; check project-id or --allow-non-dx.")

        subset = _select_gene_subset(
            rows,
            project_id=str(args.project_id),
            gene=str(args.gene),
            per_class=int(args.per_class),
            maf_download_dir=maf_download_dir,
            token_file=args.token,
            retry_amount=int(args.retry_amount),
            force_downloads=bool(args.force or args.force_downloads),
            experimental_strategy=str(args.mutation_strategy or ""),
        )

        tumor = sum(1 for r in subset if r.label_index == 1)
        normal = sum(1 for r in subset if r.label_index == 0)
        size_gb = sum(r.size for r in subset) / (1024**3)
        print(f"[subset] Selected slides: pos={tumor} neg={normal} total={len(subset)} download_size≈{size_gb:.2f} GB")

        _write_filtered_gdc_manifest(subset, subset_manifest)
        svs_download_dir.mkdir(parents=True, exist_ok=True)
        force_downloads = bool(args.force or args.force_downloads)
        download_rows = (
            subset
            if force_downloads
            else [
                r
                for r in subset
                if not _download_is_complete_with_md5(
                    r.file_id,
                    r.filename,
                    r.size,
                    svs_download_dir,
                    r.md5,
                )
            ]
        )
        if not download_rows:
            print("[gdc] All SVS files already present; skipping download.")
        elif _gdc_client_usable(gdc_client):
            manifest_to_download = target_manifest_dir / f"{args.project_id}_{gene_upper}_svs_manifest_download.tsv"
            _write_filtered_gdc_manifest(download_rows, manifest_to_download)
            _download_with_gdc_client(
                gdc_client=gdc_client,
                manifest_path=manifest_to_download,
                out_dir=svs_download_dir,
                token=args.token,
                retry_amount=int(args.retry_amount),
            )
        else:
            print("[gdc-client] Falling back to GDC API download (resume/retry enabled).")
            _download_with_gdc_api(
                rows=[
                    _GdcFileRow(file_id=r.file_id, filename=r.filename, md5=r.md5, size=r.size, state=r.state)
                    for r in download_rows
                ],
                out_dir=svs_download_dir,
                token_file=args.token,
                retry_amount=int(args.retry_amount),
                force_downloads=force_downloads,
            )

        slide_rows: List[Dict[str, object]] = []
        for row in subset:
            slide_path = _ensure_valid_svs(
                row,
                download_root=svs_download_dir,
                token_file=args.token,
                retry_amount=int(args.retry_amount),
                force_downloads=bool(args.force or args.force_downloads),
            )
            slide_rows.append(
                {
                    "slide_id": row.barcode,
                    "slide_path": str(slide_path),
                    "label_index": int(row.label_index),
                    "patient_id": row.patient_id,
                }
            )
        with slide_manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["slide_id", "slide_path", "label_index", "patient_id"])
            writer.writeheader()
            writer.writerows(slide_rows)
        print(f"[tcga] Slide manifest: {slide_manifest} (rows={len(slide_rows)})")

    if not slide_manifest.exists():
        raise FileNotFoundError(f"Slide manifest missing: {slide_manifest}")

    mpp_targets = _available_mpp_targets(args.target_mpp, args.extra_target_mpp)
    base_mpp = max(mpp_targets)
    slide_df = _ensure_slide_mpp_columns(slide_manifest)
    bucket_mpp: List[float] = []
    bucket_label: List[str] = []
    for row in slide_df.itertuples():
        mpp_val = getattr(row, "mpp_x", None) or getattr(row, "mpp_y", None)
        chosen = _choose_mpp_bucket(mpp_val, mpp_targets)
        bucket_mpp.append(float(chosen))
        bucket_label.append(_mpp_label(chosen))
    slide_df["bucket_mpp"] = bucket_mpp
    slide_df["bucket_label"] = bucket_label

    slide_index, tiling_manifest_rows = _load_slide_index(slide_manifest, run_dir, str(args.project_id))
    _write_tiling_manifest(run_dir, tiling_manifest_rows, resume=bool(args.resume))

    tiling_manifest_dir = run_dir / "tiling"
    tiling_manifest_dir.mkdir(parents=True, exist_ok=True)
    bucket_tiling_manifests: Dict[str, Path] = {}
    for target_mpp in mpp_targets:
        label = _mpp_label(target_mpp)
        subset = slide_df[slide_df["bucket_label"] == label]
        out_path = tiling_manifest_dir / f"tiling_manifest_{label}.csv"
        if not subset.empty:
            subset[["slide_id", "slide_path"]].to_csv(out_path, index=False)
        else:
            with out_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["slide_id", "slide_path"])
                writer.writeheader()
        bucket_tiling_manifests[label] = out_path

    # Versioned split manifest lives alongside training outputs (matches manuscript layout).
    split_manifest_dir = target_dir / "versioned_split_manifest"
    split_manifest_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = split_manifest_dir / f"{gene_upper}_all_splits_latest.csv"

    split_columns = [f"split_{i}_set" for i in range(1, 6)]
    if args.resume and split_manifest.exists() and not args.rebuild:
        print(f"[resume] Using existing versioned split manifest: {split_manifest}")
    else:
        seed_base = 1337
        max_attempts = 25
        for attempt in range(max_attempts):
            seed = seed_base + attempt
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "generate_versioned_split_manifest.py"),
                    "--manifest",
                    str(slide_manifest),
                    "--target",
                    gene_upper,
                    "--label-column",
                    "label_index",
                    "--target-dir",
                    str(target_dir),
                    "--splits",
                    "5",
                    "--test-frac",
                    "0.30",
                    "--seed",
                    str(seed),
                    "--overwrite",
                ],
                check=True,
            )
            import pandas as pd

            df = pd.read_csv(split_manifest)
            _add_val_assignments(
                df,
                split_columns=split_columns,
                label_column="label_index",
                val_per_class=1,
                seed=10_000 + seed,
            )
            ok = True
            for col in split_columns:
                if not _split_has_both_classes(df, split_col=col, split_value="test", label_column="label_index"):
                    ok = False
                    break
                if not _split_has_both_classes(df, split_col=col, split_value="val", label_column="label_index"):
                    ok = False
                    break
            if ok:
                df.to_csv(split_manifest, index=False)
                break
        else:
            raise RuntimeError(
                "Unable to construct splits with both classes in val/test for all 5 splits. "
                "Try increasing --per-class or changing --gene."
            )

    print(f"[tcga] Versioned split manifest: {split_manifest}")

    bucket_split_manifests = _write_bucketed_split_manifests(
        split_manifest,
        slide_df,
        run_dir,
        mpp_targets,
    )

    # Stage 1: tiling (bucketed by native MPP)
    for target_mpp in mpp_targets:
        label = _mpp_label(target_mpp)
        tiles_dir = _mpp_tiles_dir(run_dir, target_mpp)
        if not _manifest_has_rows(bucket_tiling_manifests[label]) and not _tile_manifests_present(tiles_dir):
            print(f"[tiling] No slides assigned to {label}; skipping tiling.")
            continue
        manifest_dir = tiles_dir / "manifests"
        if args.resume and manifest_dir.exists() and any(manifest_dir.glob("*_tiles.csv")):
            print(f"[resume] Tiling already present: {tiles_dir}")
        else:
            tile_size = _scale_tile_size(args.tile_size, base_mpp, target_mpp)
            stride = _scale_tile_size(args.stride, base_mpp, target_mpp)
            tiling_args = [
                "tiling",
                str(bucket_tiling_manifests[label]),
                "--output",
                str(output_root),
                "--run-name",
                str(args.run_name),
                "--tile-size",
                str(int(tile_size)),
                "--stride",
                str(int(stride)),
                "--target-mpp",
                str(float(target_mpp)),
                "--tiles-dir",
                str(tiles_dir),
            ]
            if int(args.limit_tiles) > 0:
                tiling_args.extend(["--limit-tiles", str(int(args.limit_tiles))])
            _run_goldmark(tiling_args)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        if abs(float(target_mpp) - float(base_mpp)) < 1e-6:
            _ensure_alias_dir(run_dir / "tiling" / "tiles", tiles_dir, label="tiling")
        _write_case_tile_manifests(tiles_dir, slide_manifest, resume=bool(args.resume))
        _write_tile_coords(
            tiles_dir,
            _tile_coords_path(run_dir, target_mpp),
            slide_index,
            project_id=str(args.project_id),
            resume=bool(args.resume),
        )

    # Stage 2: features
    feature_scope = "missing" if args.resume and not args.rebuild else "all"
    for target_mpp in mpp_targets:
        label = _mpp_label(target_mpp)
        split_manifest_path = bucket_split_manifests.get(label)
        if not split_manifest_path or not split_manifest_path.exists():
            continue
        if not _manifest_has_rows(split_manifest_path):
            print(f"[features] No slides assigned to {label}; skipping feature extraction.")
            continue
        tile_size = _scale_tile_size(args.tile_size, base_mpp, target_mpp)
        tiles_dir = _mpp_tiles_dir(run_dir, target_mpp)
        if not _tile_manifests_present(tiles_dir):
            print(f"[features] No tile manifests present in {tiles_dir}; skipping {label}.")
            continue
        _run_goldmark(
            [
                "features",
                str(split_manifest_path),
                "--tile-manifests",
                str(tiles_dir),
                "--output",
                str(output_root),
                "--run-name",
                str(args.run_name),
                "--encoder",
                str(args.encoder),
                "--device",
                str(args.device),
                "--batch-size",
                "32",
                "--num-workers",
                "4",
                "--tile-size",
                str(int(tile_size)),
                "--scope",
                feature_scope,
            ]
        )
    feature_dir = run_dir / "features" / str(args.encoder)

    # Stage 3: 5-split cross-validation training
    aggregator_name = "gma"
    aggregator_label = aggregator_name.upper()
    target_checkpoints_root = _ensure_target_checkpoint_root(
        checkpoints_root,
        target_dir,
        encoder_name=str(args.encoder),
        aggregator_name=aggregator_label,
    )
    _run_goldmark(
        [
            "training",
            str(split_manifest),
            "--feature-dir",
            str(feature_dir),
            "--output",
            str(output_root),
            "--run-name",
            str(args.run_name),
            "--checkpoints-root",
            str(target_checkpoints_root),
            "--target",
            "label_index",
            "--aggregator",
            aggregator_name,
            "--epochs",
            str(int(args.epochs)),
            "--patience",
            str(int(args.patience)),
            "--batch-size",
            "2",
            "--device",
            "cuda" if str(args.device).lower() == "auto" else str(args.device),
            "--split-column",
            "split",
            "--train-value",
            "train",
            "--val-value",
            "val",
            "--test-value",
            "test",
            "--cv-columns",
            *split_columns,
        ]
    )
    _ensure_alias_dir(run_dir / "checkpoints", run_dir / "training" / "checkpoints", label="checkpoints")

    import pandas as pd

    cv_summary = target_checkpoints_root / "classification_report" / "cv_summary.csv"
    if not cv_summary.exists():
        raise FileNotFoundError(f"Expected CV summary not found: {cv_summary}")

    best_epochs: Dict[str, int] = {}
    try:
        cv_df = pd.read_csv(cv_summary)
        for _, row in cv_df.iterrows():
            split = str(row.get("split") or "").strip()
            if not split:
                continue
            try:
                best_epoch_val = int(row.get("best_epoch"))
            except (TypeError, ValueError):
                continue
            if best_epoch_val > 0:
                best_epochs[split] = best_epoch_val
    except Exception:
        best_epochs = {}

    # Stage 4: per-split held-out test inference (attention + plots)

    manifest_df = pd.read_csv(split_manifest)
    for split_col in split_columns:
        ckpt = target_checkpoints_root / split_col / "checkpoint" / "checkpoint_best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint for {split_col}: {ckpt}")
        # Write inference artifacts under the split checkpoint directory so each split is self-contained.
        out_dir = target_checkpoints_root / split_col / "inference" / "test"
        runner = InferenceRunner(
            manifest=manifest_df,
            feature_dir=feature_dir,
            checkpoint_path=ckpt,
            output_dir=out_dir,
            target_column="label_index",
            config=InferenceConfig(
                split_column=split_col,
                split_value="test",
                generate_overlays=bool(args.with_overlays),
                export_attention=True,
            ),
        )
        results_path = runner.run()
        if not args.no_plots:
            _maybe_write_roc_pr_plots(
                results_path,
                out_dir / "plots",
                title=f"{args.project_id} · {gene_upper} · {args.encoder} · {split_col} test",
            )

    # Stage 5: external inference per split (best AUC epoch + final epoch).
    external_manifest_raw = str(args.external_manifest or "").strip()
    external_results: List[Path] = []
    if external_manifest_raw:
        external_manifest_path = Path(external_manifest_raw).expanduser().resolve()
        external_root = Path(str(args.external_root)).expanduser().resolve() if args.external_root else None
        external_rows, external_feature_dir = _load_external_manifest_balanced(
            external_manifest_path,
            gene=gene_upper,
            external_root=external_root,
            encoder=str(args.encoder),
            per_class=int(args.external_per_class),
        )
        external_out_manifest = target_manifest_dir / f"external_manifest_{gene_upper}_{args.encoder}.csv"
        pd.DataFrame(external_rows).to_csv(external_out_manifest, index=False)
        print(f"[external] External manifest: {external_out_manifest} (rows={len(external_rows)})")
        for split_col in split_columns:
            split_dir = target_checkpoints_root / split_col
            ckpt_dir = split_dir / "checkpoint"
            best_epoch = best_epochs.get(split_col)
            final_epoch = _find_last_epoch_checkpoint(ckpt_dir)
            epochs_to_run: List[Tuple[str, int]] = []
            if best_epoch:
                epochs_to_run.append(("best", best_epoch))
            if final_epoch and final_epoch != best_epoch:
                epochs_to_run.append(("final", final_epoch))
            if not epochs_to_run:
                print(f"[external] No checkpoints found for {split_col}; skipping", flush=True)
                continue
            for label, epoch in epochs_to_run:
                if label == "best":
                    ckpt_path = ckpt_dir / "checkpoint_best.pt"
                    if not ckpt_path.exists():
                        ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pt"
                else:
                    ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pt"
                if not ckpt_path.exists():
                    ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pth"
                if not ckpt_path.exists():
                    print(f"[external] Missing checkpoint for {split_col} epoch {epoch}; skipping", flush=True)
                    continue
                suffix = f"ckpt_best_{epoch:03d}" if label == "best" else f"ckpt_{epoch:03d}"
                external_dir = split_dir / "external_inference" / "external" / suffix
                external_dir.mkdir(parents=True, exist_ok=True)
                runner = InferenceRunner(
                    manifest=pd.read_csv(external_out_manifest),
                    feature_dir=external_feature_dir,
                    checkpoint_path=ckpt_path,
                    output_dir=external_dir,
                    target_column="label_index",
                    config=InferenceConfig(
                        split_column="",
                        split_value="external",
                        generate_overlays=bool(args.with_overlays),
                        export_attention=True,
                    ),
                )
                result_path = runner.run()
                external_results.append(result_path)
                if not args.no_plots:
                    _maybe_write_roc_pr_plots(
                        result_path,
                        external_dir / "plots",
                        title=f"External cohort · {gene_upper} · {args.encoder} · {split_col} · {suffix}",
                    )
    else:
        print("[external] Skipping external inference (no --external-manifest provided).")

    print("\nTCGA CV → external cohort full run complete.")
    print(f"- Run dir: {run_dir}")
    print(f"- Split manifest: {split_manifest}")
    print(f"- CV summary: {cv_summary}")
    if external_results:
        print(f"- External inference results: {len(external_results)} runs")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
