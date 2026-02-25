#!/usr/bin/env python3
"""End-to-end smoke test for GOLDMARK using real TCGA SVS slides from GDC.

This script does what the manuscript pipeline actually does on real inputs:
  gdc download -> tiling -> features -> training -> inference

To keep the run tractable, it downloads a *small* stratified subset of slides
(defaults to tumor vs normal inferred from the TCGA barcode) and limits the
number of tiles per slide.

If you pass `--gene`, the smoke test will also download per-case somatic mutation
calls from GDC (Masked Somatic Mutation, MAF) and derive real Positive/Negative
labels for that gene (per-patient), then train/evaluate using those labels.

By default, this script filters to diagnostic FFPE-style slides with barcodes
containing `-00-DX` (override with `--allow-non-dx`).

Requirements:
  - `gdc-client` (auto-installed to `bin/gdc-client` if missing)
  - WSI deps + native OpenSlide:
      python -m pip install -r requirements-wsi.txt
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
    label_index: int  # 1=positive, 0=negative (meaning depends on labeling mode)


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
    # TCGA-AB-1234-01A-... -> sample type is 01
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


def _label_from_sample_type(sample_type: Optional[int]) -> Optional[int]:
    if sample_type is None:
        return None
    # TCGA convention: 01..09 = tumor; 10..19 = normal; others vary.
    return 1 if int(sample_type) < 10 else 0


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
            label = _label_from_sample_type(sample_type)
            if label is None:
                continue
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
                    label_index=int(label),
                )
            )
    return rows


def _select_small_subset(rows: List[_ManifestRow], *, per_class: int) -> List[_ManifestRow]:
    """Select `per_class` unique patients per label using the label_index already in the rows."""
    per_class = max(1, int(per_class))
    by_label: Dict[int, List[_ManifestRow]] = {0: [], 1: []}
    for row in rows:
        by_label[int(row.label_index)].append(row)

    selected: List[_ManifestRow] = []
    for label_value in (0, 1):
        candidates = sorted(by_label[label_value], key=lambda r: (r.size, r.filename))
        if not candidates:
            raise ValueError(f"No slides found for label={label_value}. Try a different TCGA project.")
        seen_patients: set[str] = set()
        for candidate in candidates:
            if candidate.patient_id in seen_patients:
                continue
            selected.append(candidate)
            seen_patients.add(candidate.patient_id)
            if len(seen_patients) >= per_class:
                break
        if len(seen_patients) < per_class:
            raise ValueError(
                f"Only found {len(seen_patients)} unique patients for label={label_value} "
                f"(requested {per_class}). Try lowering --per-class or using another project."
            )
    return sorted(selected, key=lambda r: (r.label_index, r.size, r.filename))


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


def _select_gene_subset(
    rows: List[_ManifestRow],
    *,
    project_id: str,
    gene: str,
    per_class: int,
    download_dir: Path,
    token_file: Optional[str],
    experimental_strategy: str = "WXS",
) -> List[_ManifestRow]:
    """Select `per_class` positive/negative patients by downloading/parsing GDC MAFs."""
    per_class = max(1, int(per_class))
    gene_upper = str(gene).strip().upper()
    if not gene_upper:
        raise ValueError("--gene is required when deriving mutation labels")

    by_patient: Dict[str, List[_ManifestRow]] = {}
    for row in rows:
        by_patient.setdefault(row.patient_id, []).append(row)

    selected: List[_ManifestRow] = []
    counts = {0: 0, 1: 0}
    checked = 0
    for patient_id in sorted(by_patient.keys()):
        if counts[0] >= per_class and counts[1] >= per_class:
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
        _download_with_gdc_api(rows=[maf_row], out_dir=download_dir, token_file=token_file)
        maf_path = download_dir / file_id / filename
        if not maf_path.exists() or maf_path.stat().st_size == 0:
            continue

        checked += 1
        is_positive = _maf_has_gene_variant(maf_path, gene_upper)
        label = 1 if is_positive else 0
        if counts[label] >= per_class:
            continue

        # Prefer tumor slides (sample_type < 10) for mutation labels.
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
            print(f"[labels] scanned cases={checked} selected pos={counts[1]} neg={counts[0]} (target per_class={per_class})")

    if counts[0] < per_class or counts[1] < per_class:
        raise ValueError(
            f"Unable to find enough labeled cases for {gene_upper} in {project_id}. "
            f"Selected pos={counts[1]} neg={counts[0]} (requested {per_class} each). "
            "Try a different --gene or increase search space by using a larger project."
        )
    return sorted(selected, key=lambda r: (r.label_index, r.size, r.filename))


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

    # If on PATH, use it.
    if shutil.which("gdc-client"):
        return Path("gdc-client")

    # Install into bin/ (turnkey).
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
        result = subprocess.run(
            [str(path), "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
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


def _download_with_gdc_api(*, rows: List[_ManifestRow], out_dir: Path, token_file: Optional[str]) -> None:
    """Download files directly from the GDC API (fallback when gdc-client is unavailable).

    Notes:
    - This is sufficient for small smoke tests.
    - For large cohort downloads, prefer gdc-client (resume/retry behavior).
    """

    import requests  # local import (repo dependency)

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


def _resolve_downloaded_svs(download_root: Path, row: _ManifestRow) -> Path:
    direct = download_root / row.file_id / row.filename
    if direct.exists():
        return direct
    # gdc-client sometimes normalizes naming; fall back to searching within the file_id folder.
    file_dir = download_root / row.file_id
    if file_dir.exists():
        matches = sorted(file_dir.glob("*.svs"))
        if len(matches) == 1:
            return matches[0]
        for candidate in matches:
            if candidate.name == row.filename:
                return candidate
    # Last resort: search by filename under root.
    candidates = sorted(download_root.rglob(row.filename))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Downloaded SVS not found for {row.file_id} ({row.filename}) under {download_root}")


def _assign_splits(rows: List[_ManifestRow]) -> Dict[str, str]:
    """Return mapping: barcode -> split."""
    by_label: Dict[int, List[_ManifestRow]] = {0: [], 1: []}
    for row in rows:
        by_label[int(row.label_index)].append(row)
    for label_value in by_label:
        by_label[label_value].sort(key=lambda r: (r.size, r.filename))

    splits: Dict[str, str] = {}
    # Prefer: first per class -> train, second -> val, third -> test.
    for label_value, items in by_label.items():
        if not items:
            continue
        splits[items[0].barcode] = "train"
        if len(items) > 1:
            splits[items[1].barcode] = "val"
        if len(items) > 2:
            splits[items[2].barcode] = "test"
        for extra in items[3:]:
            splits[extra.barcode] = "train"

    # Ensure at least one validation slide when possible (training expects a val split).
    if len(rows) >= 2 and "val" not in set(splits.values()):
        train_rows = [row for row in rows if splits.get(row.barcode) == "train"]
        if len(train_rows) >= 2:
            train_rows.sort(key=lambda r: (r.size, r.filename))
            splits[train_rows[-1].barcode] = "val"

    for row in rows:
        splits.setdefault(row.barcode, "train")
    return splits


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

    try:
        df = pd.read_csv(results_csv)
    except Exception as exc:  # noqa: BLE001
        print(f"[plots] Skipping ROC/PR plots (failed to read {results_csv}): {exc}")
        return
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs", help="Pipeline output root (default: runs/)")
    parser.add_argument("--run-name", default="gdc_smoke_test", help="Run name under --output")
    parser.add_argument("--project-id", default="TCGA-COAD", help="TCGA project id (default: TCGA-COAD)")
    parser.add_argument("--per-class", type=int, default=2, help="Slides per class (requires >=2 for AUC/plots)")
    parser.add_argument(
        "--gene",
        default=None,
        help=(
            "If set, derive real Positive/Negative labels by downloading per-case somatic mutation calls "
            "(Masked Somatic Mutation, MAF) from GDC and checking for variants in this gene."
        ),
    )
    parser.add_argument(
        "--mutation-strategy",
        default="WXS",
        help="Experimental strategy filter for mutation calls (default: WXS)",
    )
    parser.add_argument(
        "--allow-non-dx",
        action="store_true",
        help="Include non-diagnostic slides (disable default '-00-DX' filtering).",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--target-mpp", type=float, default=0.5, help="Tiling target mpp (default: 0.5 ~ 20x)")
    parser.add_argument("--limit-tiles", type=int, default=2048, help="Limit tiles per slide (default: 2048)")
    parser.add_argument("--encoder", default="toy", help="Encoder to use (default: toy)")
    parser.add_argument("--device", default="cpu", help="Training/inference device (default: cpu)")
    parser.add_argument("--with-overlays", action="store_true", help="Generate overlays (requires OpenSlide + cv2)")
    parser.add_argument("--no-plots", action="store_true", help="Skip ROC/PR plot generation")
    parser.add_argument("--gdc-client", default=None, help="Path to gdc-client (default: auto-install/bin/ or PATH)")
    parser.add_argument("--token", default=None, help="Optional GDC token file for controlled-access data")
    parser.add_argument(
        "--secrets-env",
        default="configs/secrets.env",
        help="Optional .env file to load tokens from (default: configs/secrets.env)",
    )
    parser.add_argument("--retry-amount", type=int, default=20, help="gdc-client retry amount (default: 20)")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing run directory")
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
        if not args.force:
            raise FileExistsError(f"Run directory already exists: {run_dir} (use --force to overwrite)")
        shutil.rmtree(run_dir)

    data_dir = run_dir / "smoke_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    full_manifest = data_dir / f"{args.project_id}_svs_manifest.tsv"
    filtered_manifest = data_dir / f"{args.project_id}_svs_manifest_smoke_subset.tsv"
    download_dir = data_dir / "gdc_download"
    if download_dir.exists():
        shutil.rmtree(download_dir)

    gdc_client = _ensure_gdc_client(args.gdc_client)
    print(f"[gdc-client] Using: {gdc_client}")

    _run_goldmark(
        [
            "gdc-manifest",
            "svs",
            "--project-id",
            str(args.project_id),
            "--out",
            str(full_manifest),
        ]
    )

    all_rows = _load_gdc_manifest(full_manifest)
    if not args.allow_non_dx:
        before = len(all_rows)
        all_rows = [row for row in all_rows if _is_dx_slide(row.barcode)]
        print(f"[smoke] DX filter: kept {len(all_rows)}/{before} slides with '-00-DX' in barcode")

    if args.gene:
        subset = _select_gene_subset(
            all_rows,
            project_id=str(args.project_id),
            gene=str(args.gene),
            per_class=int(args.per_class),
            download_dir=download_dir,
            token_file=args.token,
            experimental_strategy=str(args.mutation_strategy or ""),
        )
        positives = sum(1 for r in subset if int(r.label_index) == 1)
        negatives = sum(1 for r in subset if int(r.label_index) == 0)
        size_gb = sum(r.size for r in subset) / (1024**3)
        print(
            f"[smoke] Selected slides: pos={positives} neg={negatives} total={len(subset)} "
            f"download_size≈{size_gb:.2f} GB (gene={str(args.gene).upper()})"
        )
    else:
        subset = _select_small_subset(all_rows, per_class=int(args.per_class))
        tumor = sum(1 for r in subset if r.label_index == 1)
        normal = sum(1 for r in subset if r.label_index == 0)
        size_gb = sum(r.size for r in subset) / (1024**3)
        print(f"[smoke] Selected slides: tumor={tumor} normal={normal} total={len(subset)} download_size≈{size_gb:.2f} GB")

    _write_filtered_gdc_manifest(subset, filtered_manifest)
    download_dir.mkdir(parents=True, exist_ok=True)
    if _gdc_client_usable(gdc_client):
        _download_with_gdc_client(
            gdc_client=gdc_client,
            manifest_path=filtered_manifest,
            out_dir=download_dir,
            token=args.token,
            retry_amount=int(args.retry_amount),
        )
    else:
        print("[gdc-client] Falling back to GDC API download (no resume).")
        _download_with_gdc_api(rows=subset, out_dir=download_dir, token_file=args.token)

    split_map = _assign_splits(subset)
    manifest_rows: List[Dict[str, object]] = []
    for row in subset:
        slide_path = _resolve_downloaded_svs(download_dir, row)
        manifest_rows.append(
            {
                "slide_id": row.barcode,
                "slide_path": str(slide_path),
                "label_index": int(row.label_index),
                "split": split_map.get(row.barcode, "train"),
            }
        )

    slide_manifest = data_dir / "smoke_manifest.csv"
    with slide_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["slide_id", "slide_path", "label_index", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"[smoke] Wrote slide manifest: {slide_manifest}")

    # Run the pipeline stages.
    _run_goldmark(
        [
            "tiling",
            str(slide_manifest),
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
    _run_goldmark(
        [
            "features",
            str(slide_manifest),
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
            "0",
            "--tile-size",
            str(int(args.tile_size)),
        ]
    )

    feature_dir = run_dir / "features" / str(args.encoder)
    test_value = "test" if any(row.get("split") == "test" for row in manifest_rows) else "none"
    split_for_inference = "test" if test_value == "test" else "val"
    _run_goldmark(
        [
            "training",
            str(slide_manifest),
            "--feature-dir",
            str(feature_dir),
            "--output",
            str(output_root),
            "--run-name",
            str(args.run_name),
            "--target",
            "label_index",
            "--epochs",
            str(int(args.epochs)),
            "--batch-size",
            "2",
            "--device",
            str(args.device),
            "--split-column",
            "split",
            "--train-value",
            "train",
            "--val-value",
            "val",
            "--test-value",
            str(test_value),
        ]
    )

    ckpt = run_dir / "training" / "checkpoints" / "checkpoint" / "checkpoint_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Expected best checkpoint not found: {ckpt}")

    _run_goldmark(
        [
            "inference",
            str(slide_manifest),
            "--feature-dir",
            str(feature_dir),
            "--checkpoint",
            str(ckpt),
            "--output",
            str(output_root),
            "--run-name",
            str(args.run_name),
            "--target",
            "label_index",
            "--split-column",
            "split",
            "--split-value",
            split_for_inference,
            "--export-attention",
        ]
        + ([] if args.with_overlays else ["--no-overlays"])
    )

    results_path = run_dir / "inference" / "inference" / "inference_results.csv"
    if not args.no_plots and results_path.exists():
        label_title = f"{str(args.gene).upper()} mutation" if args.gene else "tumor vs normal"
        _maybe_write_roc_pr_plots(
            results_path,
            run_dir / "plots",
            title=f"{args.project_id} · {label_title} · {args.encoder}",
        )
    print("\nGDC smoke test complete.")
    print(f"- Run directory: {run_dir}")
    print(f"- Slide manifest: {slide_manifest}")
    print(f"- Inference results: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
