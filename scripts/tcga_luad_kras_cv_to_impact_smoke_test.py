#!/usr/bin/env python3
"""End-to-end LUAD KRAS smoke test (TCGA → IMPACT) for GOLDMARK.

This script runs what the manuscript pipeline needs to prove it works:

1) Download a small set of real TCGA-LUAD diagnostic slides from GDC (SVS)
2) Derive *real* Positive/Negative labels for KRAS using per-case mutation calls
   (Masked Somatic Mutation MAF from GDC)
3) Create 5 unique 70/30 train/test splits saved in a versioned split manifest
   - For best-epoch selection, the script also carves a tiny per-split validation
     subset out of the training partition (1 positive + 1 negative when possible).
4) Tile → extract features → train for N epochs (default: 10) across the 5 splits
5) Run held-out test inference for each split with attention export + ROC/PR plot
6) Pick the best split (by test ROC AUC) and run external inference on 10 IMPACT
   LUAD cases using the KRAS column (attention export + ROC/PR plot).

Notes
-----
- Default TCGA slide filter is diagnostic/FFPE-style barcodes containing `-00-DX`.
- Requires tokens in `configs/secrets.env` (auto-loaded) for controlled GDC calls.
- For encoder consistency with the IMPACT feature cache, default encoder is
  `h-optimus-0`.

Example
-------
/data1/vanderbc/vanderbc/anaconda3/bin/python scripts/tcga_luad_kras_cv_to_impact_smoke_test.py \\
  --run-name gdc_smoke_test_sandbox_api2_kras_cv \\
  --device cuda
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from goldmark.cli import main as goldmark_main  # noqa: E402
from goldmark.inference import InferenceConfig, InferenceRunner  # noqa: E402
from goldmark.utils.secrets import load_secrets_env  # noqa: E402


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


def _download_with_gdc_api(*, rows: List[_GdcFileRow], out_dir: Path, token_file: Optional[str]) -> None:
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
        if target_path.exists() and target_path.stat().st_size > 0:
            continue

        url = f"https://api.gdc.cancer.gov/data/{row.file_id}"
        print(f"[gdc-api] Downloading {row.file_id} -> {target_path.name} ({row.size / (1024**2):.1f} MB)")
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        md5 = hashlib.md5()
        with requests.get(url, headers=headers, stream=True, timeout=(30, 600)) as resp:
            resp.raise_for_status()
            with tmp_path.open("wb") as handle:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    md5.update(chunk)
        tmp_path.replace(target_path)
        if row.md5:
            digest = md5.hexdigest()
            if digest != row.md5:
                raise ValueError(f"MD5 mismatch for {row.file_id}: expected={row.md5} got={digest}")


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


def _select_gene_subset(
    rows: List[_ManifestRow],
    *,
    project_id: str,
    gene: str,
    per_class: int,
    maf_download_dir: Path,
    token_file: Optional[str],
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
        _download_with_gdc_api(rows=[maf_row], out_dir=maf_download_dir, token_file=token_file)
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


def _discover_impact_feature(feature_dir: Path, dmp_assay_id: str) -> Tuple[Path, Optional[Path]]:
    dmp_assay_id = str(dmp_assay_id).strip()
    if not dmp_assay_id:
        raise ValueError("Missing DMP_ASSAY_ID")
    candidates = [
        feature_dir / f"features_img{dmp_assay_id}.pt",
        feature_dir / f"features_{dmp_assay_id}.pt",
        feature_dir / f"features_img{dmp_assay_id}.pth",
        feature_dir / f"features_{dmp_assay_id}.pth",
    ]
    feature_path = next((p for p in candidates if p.exists()), None)
    if feature_path is None:
        raise FileNotFoundError(
            f"IMPACT feature tensor not found for {dmp_assay_id} under {feature_dir}. Tried: "
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


def _load_impact_manifest_balanced(
    path: Path,
    *,
    gene: str,
    impact_root: Path,
    encoder: str,
    per_class: int,
) -> Tuple[List[Dict[str, object]], Path]:
    import pandas as pd

    df = pd.read_csv(path)
    gene_col = str(gene).strip().upper()
    if gene_col not in df.columns:
        raise ValueError(f"IMPACT manifest missing gene column {gene_col!r}: {path}")
    if "DMP_ASSAY_ID" not in df.columns:
        raise ValueError(f"IMPACT manifest missing DMP_ASSAY_ID column: {path}")

    if "scanned_slides_exist" in df.columns:
        df = df.loc[df["scanned_slides_exist"].map(_coerce_bool)].copy()
    if df.empty:
        raise ValueError("No IMPACT rows selected (scanned_slides_exist filtered everything).")

    cohort = None
    if "OncoTree_Code" in df.columns:
        cohort = str(df["OncoTree_Code"].iloc[0]).strip().upper() or None
    cohort = cohort or "LUAD"

    cohort_dir = impact_root / cohort
    feature_dir = cohort_dir / "features" / str(encoder)
    if not feature_dir.exists():
        raise FileNotFoundError(f"IMPACT feature dir not found: {feature_dir}")

    labeled = df.copy()
    labeled["label_index"] = labeled[gene_col].map(_map_binary_status)
    labeled = labeled.dropna(subset=["label_index"]).copy()
    labeled["label_index"] = labeled["label_index"].astype(int)
    if labeled.empty:
        raise ValueError(f"No usable IMPACT labels found for {gene_col} (need Positive/Negative).")

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
            raise ValueError(f"IMPACT selection needs both classes; found pos={len(pos)} neg={len(neg)} for {gene_col}.")
        picked = pd.concat([pos.head(take_pos), neg.head(take_neg)], ignore_index=True)
        picked = picked.sample(frac=1.0, random_state=1337).reset_index(drop=True)

    rows: List[Dict[str, object]] = []
    for _, row in picked.iterrows():
        dmp = str(row["DMP_ASSAY_ID"]).strip()
        label = int(row["label_index"])
        feature_path, meta_path = _discover_impact_feature(feature_dir, dmp)
        slide_path = _infer_slide_path_from_tile_manifest(meta_path)
        rows.append(
            {
                "slide_id": dmp,
                "slide_path": slide_path,
                "label_index": label,
                "split": "external",
                "feature_path": str(feature_path),
            }
        )
    if not rows:
        raise ValueError("No usable IMPACT rows found (feature lookup failed for all selected slides).")
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
    parser.add_argument("--run-name", default="tcga_luad_kras_cv_smoke_test", help="Run name under --output")
    parser.add_argument("--project-id", default="TCGA-LUAD", help="TCGA project id (default: TCGA-LUAD)")
    parser.add_argument("--gene", default="KRAS", help="Gene to label (default: KRAS)")
    parser.add_argument(
        "--per-class",
        type=int,
        default=5,
        help="Slides per class (default: 5 -> 10 total). Use 0 to label and use all available cases (very large).",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience (default: 50)")
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--target-mpp", type=float, default=0.5)
    parser.add_argument("--limit-tiles", type=int, default=2048)
    parser.add_argument("--encoder", default="h-optimus-0")
    parser.add_argument("--device", default="auto", help="Feature/training device (default: auto)")
    parser.add_argument("--with-overlays", action="store_true", help="Generate overlays (requires OpenSlide + cv2)")
    parser.add_argument("--no-plots", action="store_true", help="Skip ROC/PR plot generation")
    parser.add_argument("--gdc-client", default=None, help="Path to gdc-client (default: auto-install/bin/ or PATH)")
    parser.add_argument("--token", default=None, help="Optional GDC token file for controlled-access data")
    parser.add_argument("--mutation-strategy", default="WXS", help="Experimental strategy filter for mutation calls")
    parser.add_argument("--retry-amount", type=int, default=20, help="gdc-client retry amount (default: 20)")
    parser.add_argument(
        "--secrets-env",
        default="configs/secrets.env",
        help="Optional .env file to load tokens from (default: configs/secrets.env)",
    )
    parser.add_argument(
        "--impact-manifest",
        default="/data1/vanderbc/foundation_model_training_images/IMPACT/manifests/Lung_Adenocarcinoma_annotated_deidentified.csv",
        help="IMPACT LUAD manifest CSV",
    )
    parser.add_argument(
        "--impact-root",
        default="/data1/vanderbc/foundation_model_training_images/IMPACT",
        help="IMPACT root directory containing cohort subdirs",
    )
    parser.add_argument(
        "--impact-per-class",
        type=int,
        default=5,
        help="IMPACT cases per class for external inference (default: 5). Use 0 to run all labeled cases (very large).",
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
        help="Rebuild tiling/features/training/inference while preserving smoke_data downloads.",
    )
    args = parser.parse_args()

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
            # Preserve smoke_data (downloads + derived labels), but wipe stage outputs.
            for stage in ("tiling", "features", "training", "inference", "external_inference", "plots"):
                target = run_dir / stage
                if target.exists():
                    shutil.rmtree(target)
        elif not args.resume:
            raise FileExistsError(
                f"Run directory already exists: {run_dir} (use --resume to continue, --rebuild to rerun stages, or --force to overwrite)"
            )

    data_dir = run_dir / "smoke_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    full_manifest = data_dir / f"{args.project_id}_svs_manifest.tsv"
    subset_manifest = data_dir / f"{args.project_id}_svs_manifest_subset.tsv"
    slide_manifest = data_dir / f"tcga_{str(args.project_id).lower()}_{str(args.gene).lower()}_slides.csv"
    svs_download_dir = data_dir / "gdc_download_svs"
    maf_download_dir = data_dir / "gdc_download_maf"

    if (args.resume or args.rebuild) and slide_manifest.exists():
        print(f"[resume] Using existing slide manifest: {slide_manifest}")
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
            experimental_strategy=str(args.mutation_strategy or ""),
        )

        tumor = sum(1 for r in subset if r.label_index == 1)
        normal = sum(1 for r in subset if r.label_index == 0)
        size_gb = sum(r.size for r in subset) / (1024**3)
        print(f"[subset] Selected slides: pos={tumor} neg={normal} total={len(subset)} download_size≈{size_gb:.2f} GB")

        _write_filtered_gdc_manifest(subset, subset_manifest)
        svs_download_dir.mkdir(parents=True, exist_ok=True)
        if _gdc_client_usable(gdc_client):
            _download_with_gdc_client(
                gdc_client=gdc_client,
                manifest_path=subset_manifest,
                out_dir=svs_download_dir,
                token=args.token,
                retry_amount=int(args.retry_amount),
            )
        else:
            print("[gdc-client] Falling back to GDC API download (no resume).")
            _download_with_gdc_api(
                rows=[
                    _GdcFileRow(file_id=r.file_id, filename=r.filename, md5=r.md5, size=r.size, state=r.state)
                    for r in subset
                ],
                out_dir=svs_download_dir,
                token_file=args.token,
            )

        slide_rows: List[Dict[str, object]] = []
        for row in subset:
            slide_path = _resolve_downloaded_svs(svs_download_dir, row)
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

    # Versioned split manifest lives alongside training outputs (matches manuscript layout).
    gene_upper = str(args.gene).strip().upper()
    target_dir = run_dir / "training" / "checkpoints" / gene_upper
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

    # Stage 1: tiling
    _run_goldmark(
        [
            "tiling",
            str(split_manifest),
            "--output",
            str(output_root),
            "--run-name",
            str(args.run_name),
            "--tile-size",
            str(int(args.tile_size)),
            "--stride",
            str(int(args.stride)),
            "--target-mpp",
            str(float(args.target_mpp)),
            "--limit-tiles",
            str(int(args.limit_tiles)),
        ]
    )
    tile_dir = run_dir / "tiling" / "tiles"

    # Stage 2: features
    _run_goldmark(
        [
            "features",
            str(split_manifest),
            "--tile-manifests",
            str(tile_dir),
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
            str(int(args.tile_size)),
        ]
    )
    feature_dir = run_dir / "features" / str(args.encoder)

    # Stage 3: 5-split cross-validation training
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
            "--target",
            "label_index",
            "--aggregator",
            "gma",
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

    cv_summary = run_dir / "training" / "checkpoints" / "classification_report" / "cv_summary.csv"
    if not cv_summary.exists():
        raise FileNotFoundError(f"Expected CV summary not found: {cv_summary}")

    # Stage 4: per-split held-out test inference (attention + plots)
    import pandas as pd

    manifest_df = pd.read_csv(split_manifest)
    for split_col in split_columns:
        ckpt = run_dir / "training" / "checkpoints" / split_col / "checkpoint" / "checkpoint_best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint for {split_col}: {ckpt}")
        # Write inference artifacts under the split checkpoint directory so each split is self-contained.
        out_dir = run_dir / "training" / "checkpoints" / split_col / "inference" / "test"
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

    # Stage 5: external inference on IMPACT using best split checkpoint.
    best_split = _pick_best_split(cv_summary)
    best_ckpt = run_dir / "training" / "checkpoints" / best_split / "checkpoint" / "checkpoint_best.pt"
    if not best_ckpt.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt}")
    impact_manifest_path = Path(args.impact_manifest).expanduser().resolve()
    impact_root = Path(args.impact_root).expanduser().resolve()
    impact_rows, impact_feature_dir = _load_impact_manifest_balanced(
        impact_manifest_path,
        gene=gene_upper,
        impact_root=impact_root,
        encoder=str(args.encoder),
        per_class=int(args.impact_per_class),
    )
    impact_out_manifest = data_dir / f"impact_external_{gene_upper}_{args.encoder}.csv"
    pd.DataFrame(impact_rows).to_csv(impact_out_manifest, index=False)
    print(f"[impact] External manifest: {impact_out_manifest} (rows={len(impact_rows)})")

    external_dir = run_dir / "training" / "checkpoints" / best_split / "external_inference" / "IMPACT"
    external_dir.mkdir(parents=True, exist_ok=True)
    runner = InferenceRunner(
        manifest=pd.read_csv(impact_out_manifest),
        feature_dir=impact_feature_dir,
        checkpoint_path=best_ckpt,
        output_dir=external_dir,
        target_column="label_index",
        config=InferenceConfig(
            # External inference runs over the entire cohort (no manifest split filtering required).
            # If you want filtering, set split_column to a real column name and split_value accordingly.
            split_column="",
            split_value="external",
            generate_overlays=bool(args.with_overlays),
            export_attention=True,
        ),
    )
    impact_results = runner.run()
    if not args.no_plots:
        _maybe_write_roc_pr_plots(
            impact_results,
            external_dir / "plots",
            title=f"IMPACT LUAD · {gene_upper} · {args.encoder} · best={best_split}",
        )

    # Convenience: ensure each split directory has an `external_inference/IMPACT` entry so that split
    # context is discoverable in one place. Prefer symlinks; fall back to a pointer file when symlinks
    # are not supported.
    for split_col in split_columns:
        split_external_root = run_dir / "training" / "checkpoints" / split_col / "external_inference"
        link_path = split_external_root / "IMPACT"
        if split_col == best_split:
            continue
        try:
            split_external_root.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if link_path.exists() or link_path.is_symlink():
            try:
                if link_path.is_dir() and not link_path.is_symlink():
                    shutil.rmtree(link_path)
                else:
                    link_path.unlink()
            except OSError:
                pass
        try:
            rel_target = os.path.relpath(external_dir, start=link_path.parent)
            os.symlink(rel_target, link_path)
        except OSError:
            try:
                link_path.mkdir(parents=True, exist_ok=True)
                (link_path / "LOCATION.txt").write_text(
                    "\n".join(
                        [
                            "External inference was run once using the best-performing split checkpoint.",
                            f"best_split: {best_split}",
                            f"checkpoint: {best_ckpt}",
                            f"results_dir: {external_dir}",
                            f"results_csv: {impact_results}",
                            "",
                        ]
                    )
                )
            except OSError:
                pass

    print("\nTCGA LUAD KRAS CV → IMPACT smoke test complete.")
    print(f"- Run dir: {run_dir}")
    print(f"- Split manifest: {split_manifest}")
    print(f"- CV summary: {cv_summary}")
    print(f"- Best split: {best_split}")
    print(f"- Best checkpoint: {best_ckpt}")
    print(f"- External IMPACT inference results: {impact_results}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
