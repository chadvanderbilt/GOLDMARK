#!/usr/bin/env python3
"""
Demo wrapper: fetch BLCA-style manifests/features from the GOLDMARK public API
and train a GMA model using existing GOLDMARK training code.

This script intentionally reuses existing pipeline entry points:
  - scripts/train_task_v2.py

It only handles:
  1) discovering downloadable artifacts via /api/runs/{slug}/downloads
  2) downloading/extracting a target split manifest
  3) resolving pre-extracted feature tensors
  4) launching train_task_v2.py
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


def _norm_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _http_get_json(url: str, timeout: int = 120) -> Dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "curl/8.5.0",
            "Accept": "application/json,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        payload = resp.read().decode(charset, errors="replace")
    return json.loads(payload)


def _download_to(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "curl/8.5.0",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as src, dest.open("wb") as out:
        shutil.copyfileobj(src, out, length=1024 * 1024)


def _join_url(base_url: str, maybe_relative: str) -> str:
    if not maybe_relative:
        raise ValueError("Empty URL component received from API.")
    if maybe_relative.startswith("http://") or maybe_relative.startswith("https://"):
        return maybe_relative
    return urllib.parse.urljoin(base_url.rstrip("/") + "/", maybe_relative.lstrip("/"))


def _find_category(downloads_payload: Dict, key: str) -> Dict:
    for category in downloads_payload.get("categories", []):
        if category.get("key") == key:
            return category
    raise ValueError(f"Category {key!r} not found in API payload.")


def _pick_manifest_bundle_item(downloads_payload: Dict) -> Dict:
    manifest_category = _find_category(downloads_payload, "manifest")
    for item in manifest_category.get("items", []):
        metadata = item.get("metadata") or {}
        rel = str(item.get("relative_path") or "")
        if metadata.get("tag") == "manifest_bundle" or rel.endswith("_manifests.zip"):
            if item.get("type") == "file":
                return item
    raise ValueError("Manifest bundle item not found in manifest category.")


def _pick_feature_bundle_item(downloads_payload: Dict, encoder: str) -> Dict:
    features_category = _find_category(downloads_payload, "features")
    wanted = _norm_token(encoder)
    best: Optional[Dict] = None
    for item in features_category.get("items", []):
        if item.get("type") != "file":
            continue
        metadata = item.get("metadata") or {}
        meta_encoder = _norm_token(metadata.get("encoder") or "")
        rel = str(item.get("relative_path") or "").lower()
        label = str(item.get("label") or "").lower()
        if meta_encoder == wanted:
            return item
        if wanted and (f"/{encoder.lower()}.zip" in rel or encoder.lower() in label):
            best = item
    if best is not None:
        return best
    raise ValueError(f"Feature bundle for encoder={encoder!r} not found in features category.")


def _extract_gene_manifest_from_bundle(bundle_zip: Path, gene: str, out_dir: Path) -> Path:
    gene_upper = str(gene).upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_zip) as zf:
        names = [
            n
            for n in zf.namelist()
            if n.lower().endswith(".csv")
            and "/._" not in n
            and f"{gene_upper}_all_splits".lower() in n.lower()
        ]
        if not names:
            raise ValueError(
                f"Could not find {gene_upper}_all_splits*.csv in manifest bundle: {bundle_zip}"
            )
        chosen = sorted(names)[0]
        out_path = out_dir / Path(chosen).name
        with zf.open(chosen) as src, out_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return out_path


def _balanced_subset(df: pd.DataFrame, target_column: str, sample_per_class: int, seed: int) -> pd.DataFrame:
    if sample_per_class <= 0:
        return df
    positives = df[df[target_column] == 1]
    negatives = df[df[target_column] == 0]
    if positives.empty or negatives.empty:
        raise ValueError("Manifest subset has only one class; cannot run binary MIL training.")
    n_pos = min(sample_per_class, len(positives))
    n_neg = min(sample_per_class, len(negatives))
    sampled = pd.concat(
        [
            positives.sample(n=n_pos, random_state=seed),
            negatives.sample(n=n_neg, random_state=seed),
        ],
        axis=0,
        ignore_index=True,
    )
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _prepare_training_manifest(
    source_manifest: Path,
    gene: str,
    output_manifest: Path,
    sample_per_class: int,
    seed: int,
) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(source_manifest)
    slide_col = None
    for col in ("slide_id", "DMP_ASSAY_ID", "case_id"):
        if col in df.columns:
            slide_col = col
            break
    if slide_col is None:
        raise ValueError(
            f"Input manifest missing slide identifier column. Columns={list(df.columns)}"
        )

    target_source = None
    for col in (gene, "target", "label_index"):
        if col in df.columns:
            target_source = col
            break
    if target_source is None:
        raise ValueError(
            f"Input manifest missing target columns ({gene}, target, label_index). Columns={list(df.columns)}"
        )

    out = pd.DataFrame()
    out["slide_id"] = df[slide_col].astype(str).str.strip()
    out[gene] = pd.to_numeric(df[target_source], errors="coerce").fillna(0).astype(int)
    out = out[out["slide_id"] != ""].drop_duplicates(subset=["slide_id"]).reset_index(drop=True)
    out = _balanced_subset(out, gene, sample_per_class=sample_per_class, seed=seed)

    if out[gene].nunique() < 2:
        raise ValueError("Prepared manifest has a single class after filtering.")

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_manifest, index=False)
    return out, gene


def _candidate_feature_basenames(slide_id: str) -> Iterable[str]:
    sid = str(slide_id)
    for prefix in ("features_", "features_img"):
        for ext in (".pt", ".json"):
            yield f"{prefix}{sid}{ext}"


def _find_feature_dir_from_zip_path(zip_path: Path) -> Optional[Path]:
    if zip_path.suffix.lower() != ".zip":
        return None
    candidate = zip_path.with_suffix("")
    if candidate.exists() and candidate.is_dir():
        return candidate
    return None


def _extract_features_subset_from_zip(
    feature_zip: Path,
    slide_ids: Iterable[str],
    output_dir: Path,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = set()
    for sid in slide_ids:
        wanted.update(_candidate_feature_basenames(str(sid)))

    extracted_pt = 0
    with zipfile.ZipFile(feature_zip) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            base = Path(member.filename).name
            if base not in wanted:
                continue
            target = output_dir / base
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            if base.endswith(".pt"):
                extracted_pt += 1
    return extracted_pt


def _count_present_features(feature_dir: Path, slide_ids: Iterable[str]) -> int:
    count = 0
    for sid in slide_ids:
        sid = str(sid)
        candidates = [
            feature_dir / f"features_{sid}.pt",
            feature_dir / f"features_img{sid}.pt",
        ]
        if any(path.exists() for path in candidates):
            count += 1
    return count


def _write_training_config(
    config_path: Path,
    *,
    folds: int,
    epochs: int,
    batch_size: int,
    num_workers: int,
    device: str,
    patience: int,
) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Auto-generated by scripts/api_demo_from_preextracted_features.py",
        "export AGGREGATION_METHOD=gma",
        f"export CV_FOLDS={int(folds)}",
        f"export NUM_EPOCHS={int(epochs)}",
        "export VAL_INTERVAL=1",
        f"export BATCH_SIZE_TRAIN={int(batch_size)}",
        "export LEARNING_RATE=1e-4",
        "export WEIGHT_DECAY=1e-4",
        f"export PATIENCE={int(patience)}",
        f"export DEVICE={device}",
        f"export NUM_WORKERS={int(num_workers)}",
        "export TRAIN_VALUE=train",
        "export VAL_VALUE=test",
        "export TEST_VALUE=test",
    ]
    config_path.write_text("\n".join(lines) + "\n")


def _run_training(
    *,
    repo_root: Path,
    python_bin: str,
    config_path: Path,
    manifest_path: Path,
    feature_dir: Path,
    target_column: str,
    output_dir: Path,
    encoder: str,
    epochs: int,
) -> None:
    train_script = repo_root / "scripts" / "train_task_v2.py"
    cmd = [
        python_bin,
        str(train_script),
        "--config",
        str(config_path),
        "--manifest",
        str(manifest_path),
        "--feature-dir",
        str(feature_dir),
        "--target-column",
        target_column,
        "--output-dir",
        str(output_dir),
        "--encoder-name",
        encoder,
        "--aggregator",
        "gma",
        "--epochs",
        str(int(epochs)),
    ]
    subprocess.run(cmd, check=True, cwd=str(repo_root))


def _build_downloads_url(
    *,
    base_url: str,
    slug: str,
    download_source: str,
    results_mode: str,
    epoch_view: str,
) -> str:
    query = urllib.parse.urlencode(
        {
            "download_source": download_source,
            "results_mode": results_mode,
            "epoch_view": epoch_view,
        }
    )
    slug_q = urllib.parse.quote(slug, safe="")
    return f"{base_url.rstrip('/')}/api/runs/{slug_q}/downloads?{query}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo: run GOLDMARK GMA from pre-extracted features discovered via public API."
    )
    parser.add_argument("--base-url", default="https://artificialintelligencepathology.org")
    parser.add_argument("--slug", default="rf207d22e2c0c~TCGA-BLCA_svs")
    parser.add_argument("--gene", default="FGFR3")
    parser.add_argument("--encoder", default="uni")
    parser.add_argument("--results-mode", default="publication")
    parser.add_argument("--epoch-view", default="best")
    parser.add_argument("--download-source", default="postgres")
    parser.add_argument("--work-dir", default="runs/api_demo_blca_fgfr3_uni")
    parser.add_argument("--sample-per-class", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--feature-dir", help="Existing feature directory; skips feature bundle download/extraction.")
    parser.add_argument(
        "--prefer-local-feature-dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If API item.path points to a local .zip and sibling extracted dir exists, use it directly.",
    )
    parser.add_argument(
        "--extract-all-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When downloading a feature zip, extract everything (default: extract only manifest-referenced slides).",
    )
    parser.add_argument(
        "--skip-training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only prepare/download data and write config+manifest, but do not launch training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    gene = str(args.gene).upper()
    encoder = str(args.encoder)
    work_dir = Path(args.work_dir).expanduser().resolve()
    downloads_dir = work_dir / "downloads"
    manifests_dir = work_dir / "manifests"
    features_out_dir = work_dir / "features" / encoder
    training_root = work_dir / "training" / "checkpoints" / gene / encoder / "gma"
    config_path = work_dir / "configs" / "api_demo_training.env"

    print(f"[api] Base URL: {args.base_url}")
    print(f"[api] Run slug:  {args.slug}")
    downloads_url = _build_downloads_url(
        base_url=args.base_url,
        slug=args.slug,
        download_source=args.download_source,
        results_mode=args.results_mode,
        epoch_view=args.epoch_view,
    )
    payload = _http_get_json(downloads_url)

    manifest_bundle_item = _pick_manifest_bundle_item(payload)
    feature_bundle_item = _pick_feature_bundle_item(payload, encoder=encoder)

    manifest_bundle_url = _join_url(args.base_url, manifest_bundle_item.get("download_url") or "")
    manifest_bundle_local = downloads_dir / "manifest_bundle.zip"
    if not manifest_bundle_local.exists():
        print(f"[download] Manifest bundle -> {manifest_bundle_local}")
        _download_to(manifest_bundle_url, manifest_bundle_local)
    else:
        print(f"[cache] Reusing {manifest_bundle_local}")

    gene_manifest_local = _extract_gene_manifest_from_bundle(manifest_bundle_local, gene=gene, out_dir=manifests_dir)
    print(f"[manifest] Gene split manifest: {gene_manifest_local}")

    training_manifest = manifests_dir / f"{gene}_{encoder}_demo_manifest.csv"
    training_df, target_column = _prepare_training_manifest(
        source_manifest=gene_manifest_local,
        gene=gene,
        output_manifest=training_manifest,
        sample_per_class=int(args.sample_per_class),
        seed=int(args.seed),
    )
    slide_ids = training_df["slide_id"].astype(str).tolist()
    print(
        f"[manifest] Prepared training manifest: {training_manifest} "
        f"(rows={len(training_df)}, positives={int((training_df[target_column] == 1).sum())}, "
        f"negatives={int((training_df[target_column] == 0).sum())})"
    )

    feature_dir: Optional[Path] = None
    if args.feature_dir:
        feature_dir = Path(args.feature_dir).expanduser().resolve()
        print(f"[features] Using user-provided feature dir: {feature_dir}")
    elif args.prefer_local_feature_dir:
        feature_item_path = feature_bundle_item.get("path")
        if feature_item_path:
            zip_candidate = Path(str(feature_item_path))
            local_dir = _find_feature_dir_from_zip_path(zip_candidate)
            if local_dir and local_dir.exists():
                feature_dir = local_dir.resolve()
                print(f"[features] Using local extracted feature dir from API path: {feature_dir}")

    if feature_dir is None:
        feature_zip_url = _join_url(args.base_url, feature_bundle_item.get("download_url") or "")
        feature_zip_local = downloads_dir / f"{encoder}.zip"
        if not feature_zip_local.exists():
            print(f"[download] Feature bundle ({encoder}) -> {feature_zip_local}")
            _download_to(feature_zip_url, feature_zip_local)
        else:
            print(f"[cache] Reusing {feature_zip_local}")

        if args.extract_all_features:
            print(f"[extract] Extracting all features from {feature_zip_local} ...")
            with zipfile.ZipFile(feature_zip_local) as zf:
                zf.extractall(features_out_dir)
            direct_pt = list(features_out_dir.glob("features_*.pt"))
            if direct_pt:
                feature_dir = features_out_dir
            else:
                nested = list(features_out_dir.rglob("features_*.pt"))
                if not nested:
                    raise RuntimeError(
                        f"No features_*.pt files found after extraction: {features_out_dir}"
                    )
                feature_dir = nested[0].parent
            print(f"[features] Extracted feature dir: {feature_dir}")
        else:
            print(f"[extract] Extracting manifest-referenced subset for {len(slide_ids)} slides ...")
            extracted = _extract_features_subset_from_zip(
                feature_zip=feature_zip_local,
                slide_ids=slide_ids,
                output_dir=features_out_dir,
            )
            if extracted == 0:
                raise RuntimeError(
                    "No feature tensors were extracted from zip for manifest slide IDs. "
                    "Use --extract-all-features to inspect archive layout."
                )
            feature_dir = features_out_dir
            print(f"[features] Extracted .pt files: {extracted} -> {feature_dir}")

    feature_dir = feature_dir.resolve()
    present = _count_present_features(feature_dir, slide_ids)
    print(f"[features] Present feature tensors for manifest slide IDs: {present}/{len(slide_ids)}")
    if present < 10:
        raise RuntimeError(
            f"Too few matching feature tensors found in {feature_dir} for the prepared manifest."
        )

    _write_training_config(
        config_path=config_path,
        folds=int(args.folds),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=str(args.device),
        patience=int(args.patience),
    )
    print(f"[config] Training config: {config_path}")

    if args.skip_training:
        print("[done] Data prep completed; training skipped (--skip-training).")
        print(f"[next] Run: {args.python_bin} scripts/train_task_v2.py --config {config_path} "
              f"--manifest {training_manifest} --feature-dir {feature_dir} --target-column {target_column} "
              f"--output-dir {training_root} --encoder-name {encoder} --aggregator gma --epochs {int(args.epochs)}")
        return 0

    print("[train] Launching scripts/train_task_v2.py (existing GOLDMARK trainer)...")
    _run_training(
        repo_root=repo_root,
        python_bin=str(args.python_bin),
        config_path=config_path,
        manifest_path=training_manifest,
        feature_dir=feature_dir,
        target_column=target_column,
        output_dir=training_root,
        encoder=encoder,
        epochs=int(args.epochs),
    )

    print("[done] Training completed.")
    print(f"[artifacts] {training_root}")
    print(f"[manifest]  {training_manifest}")
    print(f"[features]  {feature_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
