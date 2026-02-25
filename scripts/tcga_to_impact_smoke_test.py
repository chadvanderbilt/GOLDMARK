#!/usr/bin/env python3
"""Train a TCGA mutation model and run reciprocal external inference on IMPACT.

This is a pragmatic, HPC-friendly smoke test that exercises the *actual* pieces you
care about for "TCGA → IMPACT" evaluation:
  - uses real mutation-derived labels (0/1) for a gene target (e.g., PIK3CA)
  - trains a MIL aggregator model from scratch on precomputed foundation features
  - runs external inference on a small IMPACT subset (default: 1 slide) using the
    same encoder's feature tensors
  - exports attention weights (CSV + NPY) and (optionally) overlays

By default, it expects the Vanderbilt "foundation_model_training_images" layout.
No sudo required.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from goldmark.cli import main as goldmark_main  # noqa: E402
from goldmark.inference import InferenceConfig, InferenceRunner  # noqa: E402
from goldmark.utils.secrets import load_secrets_env  # noqa: E402


def _run_goldmark(argv: List[str]) -> None:
    print("\n==> goldmark " + " ".join(argv))
    goldmark_main(argv)


def _coerce_bool(value: object) -> bool:
    token = str(value).strip().lower()
    return token in {"1", "true", "yes", "y", "on"}


def _map_binary_status(value: object) -> Optional[int]:
    token = str(value).strip().lower()
    if not token:
        return None
    if token in {"positive", "pos", "1", "true", "yes", "y"}:
        return 1
    if token in {"negative", "neg", "0", "false", "no", "n"}:
        return 0
    return None


def _select_tcga_subset(
    rows: List[Dict[str, object]],
    *,
    split_col: str,
    train_per_class: int,
    test_per_class: int,
) -> List[Dict[str, object]]:
    if train_per_class <= 0 and test_per_class <= 0:
        return rows

    def _bucket(value: str) -> str:
        token = str(value or "").strip().lower()
        return "test" if token == "test" else "train"

    train_per_class = max(0, int(train_per_class))
    test_per_class = max(0, int(test_per_class))

    # Deterministic: stable sort first.
    ordered = sorted(rows, key=lambda r: (str(r.get("slide_id") or ""), str(r.get("slide_path") or "")))

    picked: List[Dict[str, object]] = []
    counts: Dict[Tuple[str, int], int] = {}
    limits = {"train": train_per_class, "test": test_per_class}

    for row in ordered:
        label = row.get("label_index")
        split = _bucket(row.get(split_col))
        if label is None:
            continue
        try:
            label_int = int(label)
        except (TypeError, ValueError):
            continue
        limit = limits.get(split, 0)
        if limit <= 0:
            continue
        key = (split, label_int)
        if counts.get(key, 0) >= limit:
            continue
        counts[key] = counts.get(key, 0) + 1
        picked.append(row)

    return picked


def _discover_tcga_paths(tcga_root: Path, gene: str, encoder: str, split_id: int) -> Tuple[Path, Path, str]:
    gene_upper = str(gene).strip().upper()
    if not gene_upper:
        raise ValueError("--gene is required")
    encoder_token = str(encoder).strip()
    if not encoder_token:
        raise ValueError("--encoder is required")

    split_id = int(split_id)
    if split_id < 1 or split_id > 5:
        raise ValueError("--split-id must be 1..5")
    split_col = f"split_{split_id}_set"

    manifest = (
        tcga_root
        / "checkpoints"
        / gene_upper
        / "versioned_split_manifest"
        / f"{gene_upper}_all_splits_latest.csv"
    )
    feature_dir = tcga_root / "features" / encoder_token
    return manifest, feature_dir, split_col


def _load_tcga_manifest(
    path: Path,
    *,
    split_col: str,
    train_per_class: int,
    test_per_class: int,
) -> List[Dict[str, object]]:
    import pandas as pd

    df = pd.read_csv(path)
    required = {"slide_id", "slide_path", "label_index", split_col}
    missing = required.difference(set(df.columns))
    if missing:
        raise ValueError(f"TCGA split manifest missing columns {sorted(missing)}: {path}")

    records = df.to_dict(orient="records")
    subset = _select_tcga_subset(
        records,
        split_col=split_col,
        train_per_class=train_per_class,
        test_per_class=test_per_class,
    )
    if not subset:
        raise ValueError("Selected 0 TCGA slides; increase --tcga-train-per-class/--tcga-test-per-class.")

    # Normalize to the pipeline schema we use elsewhere.
    out: List[Dict[str, object]] = []
    for row in subset:
        label = row.get("label_index")
        try:
            label_int = int(label)  # 0/1
        except (TypeError, ValueError):
            continue
        out.append(
            {
                "slide_id": row.get("slide_id"),
                "slide_path": row.get("slide_path"),
                "label_index": label_int,
                "split": str(row.get(split_col) or "").strip().lower() or "train",
            }
        )

    # Safety: ensure test split has both classes when we're going to compute AUC.
    test_rows = [r for r in out if str(r.get("split")) == "test"]
    if test_rows:
        labels = sorted({int(r["label_index"]) for r in test_rows})
        if len(labels) < 2:
            raise ValueError(
                "TCGA test subset is single-class; increase --tcga-test-per-class to include both labels."
            )
    return out


def _discover_impact_feature(
    feature_dir: Path,
    dmp_assay_id: str,
) -> Tuple[Path, Optional[Path]]:
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


def _load_impact_manifest(
    path: Path,
    *,
    gene: str,
    impact_root: Path,
    encoder: str,
    dmp_assay_id: Optional[str],
    limit: int,
) -> Tuple[List[Dict[str, object]], Path]:
    import pandas as pd

    df = pd.read_csv(path)
    gene_col = str(gene).strip().upper()
    if gene_col not in df.columns:
        raise ValueError(f"IMPACT manifest missing gene column {gene_col!r}: {path}")
    if "DMP_ASSAY_ID" not in df.columns:
        raise ValueError(f"IMPACT manifest missing DMP_ASSAY_ID column: {path}")

    if dmp_assay_id:
        df = df.loc[df["DMP_ASSAY_ID"].astype(str) == str(dmp_assay_id)].copy()
    else:
        if "scanned_slides_exist" in df.columns:
            df = df.loc[df["scanned_slides_exist"].map(_coerce_bool)].copy()

    if df.empty:
        raise ValueError("No IMPACT rows selected (check --impact-dmp-assay-id or scanned_slides_exist).")

    # Infer which IMPACT cohort dir to use.
    cohort = None
    if "OncoTree_Code" in df.columns:
        cohort = str(df["OncoTree_Code"].iloc[0]).strip().upper() or None
    cohort = cohort or "COAD"
    cohort_dir = impact_root / cohort
    feature_dir = cohort_dir / "features" / str(encoder)
    if not feature_dir.exists():
        raise FileNotFoundError(f"IMPACT feature dir not found: {feature_dir}")

    rows: List[Dict[str, object]] = []
    take = min(int(limit), len(df)) if int(limit) > 0 else len(df)
    for _, row in df.head(take).iterrows():
        dmp = str(row["DMP_ASSAY_ID"]).strip()
        label = _map_binary_status(row.get(gene_col))
        if label is None:
            continue
        feature_path, meta_path = _discover_impact_feature(feature_dir, dmp)
        slide_path = _infer_slide_path_from_tile_manifest(meta_path)
        rows.append(
            {
                "slide_id": dmp,
                "slide_path": slide_path,
                "label_index": int(label),
                "split": "external",
                "feature_path": str(feature_path),
            }
        )
    if not rows:
        raise ValueError(
            f"No usable IMPACT rows found for {gene_col} (need Positive/Negative and matching feature tensors)."
        )
    return rows, feature_dir


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs", help="Output root (default: runs/)")
    parser.add_argument("--run-name", default="tcga_to_impact_smoke_test", help="Run name under --output")
    parser.add_argument("--gene", default="PIK3CA", help="Gene target (default: PIK3CA)")
    parser.add_argument(
        "--tcga-root",
        default="/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-COAD_svs",
        help="TCGA cohort root (default: COAD SVS bundle)",
    )
    parser.add_argument("--encoder", default="h-optimus-0", help="Encoder/features folder name (default: h-optimus-0)")
    parser.add_argument("--split-id", type=int, default=1, help="Which split column to use (1..5)")
    parser.add_argument("--tcga-train-per-class", type=int, default=3, help="Train slides per class (0=all)")
    parser.add_argument("--tcga-test-per-class", type=int, default=1, help="Test/val slides per class (0=all)")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs (default: 1)")
    parser.add_argument("--device", default="cpu", help="Training device (default: cpu)")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size (default: 2)")
    parser.add_argument("--aggregator", default="gma", help="MIL aggregator (default: gma)")
    parser.add_argument("--no-overlays", action="store_true", help="Disable overlay generation")
    parser.add_argument("--no-plots", action="store_true", help="Skip ROC/PR plot generation")

    parser.add_argument(
        "--impact-manifest",
        default="/data1/vanderbc/foundation_model_training_images/IMPACT/manifests/Colon_Adenocarcinoma_annotated_deidentified.csv",
        help="IMPACT deidentified manifest csv (Colon Adenocarcinoma)",
    )
    parser.add_argument(
        "--impact-root",
        default="/data1/vanderbc/foundation_model_training_images/IMPACT",
        help="IMPACT dataset root (default: foundation_model_training_images/IMPACT)",
    )
    parser.add_argument("--impact-dmp-assay-id", default=None, help="Optional specific DMP_ASSAY_ID to run")
    parser.add_argument("--impact-limit", type=int, default=1, help="Number of IMPACT slides to infer (default: 1)")

    parser.add_argument(
        "--secrets-env",
        default="configs/secrets.env",
        help="Optional .env file to load tokens from (default: configs/secrets.env)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing run directory")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    os_output = Path(args.output).expanduser().resolve()
    run_dir = os_output / str(args.run_name)
    if run_dir.exists():
        if not args.force:
            raise FileExistsError(f"Run dir exists: {run_dir} (use --force)")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = run_dir / "smoke_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    secrets_path = Path(str(args.secrets_env)).expanduser()
    if not secrets_path.is_absolute():
        secrets_path = (REPO_ROOT / secrets_path).resolve()
    if secrets_path.exists():
        load_secrets_env(secrets_path, verbose=True)
    else:
        load_secrets_env(verbose=False)

    tcga_root = Path(args.tcga_root).expanduser().resolve()
    tcga_manifest_path, tcga_feature_dir, tcga_split_col = _discover_tcga_paths(
        tcga_root,
        str(args.gene),
        str(args.encoder),
        int(args.split_id),
    )
    if not tcga_manifest_path.exists():
        raise FileNotFoundError(f"TCGA split manifest not found: {tcga_manifest_path}")
    if not tcga_feature_dir.exists():
        raise FileNotFoundError(f"TCGA feature dir not found: {tcga_feature_dir}")

    tcga_rows = _load_tcga_manifest(
        tcga_manifest_path,
        split_col=tcga_split_col,
        train_per_class=int(args.tcga_train_per_class),
        test_per_class=int(args.tcga_test_per_class),
    )
    tcga_out_manifest = data_dir / f"tcga_{str(args.gene).upper()}_{args.encoder}_split{int(args.split_id)}_smoke.csv"
    try:
        import pandas as pd

        pd.DataFrame(tcga_rows).to_csv(tcga_out_manifest, index=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to write TCGA smoke manifest: {tcga_out_manifest}: {exc}") from exc
    print(f"[tcga] Smoke manifest: {tcga_out_manifest} (rows={len(tcga_rows)})")

    # Train from scratch (using precomputed features).
    _run_goldmark(
        [
            "training",
            str(tcga_out_manifest),
            "--feature-dir",
            str(tcga_feature_dir),
            "--output",
            str(os_output),
            "--run-name",
            str(args.run_name),
            "--target",
            "label_index",
            "--aggregator",
            str(args.aggregator),
            "--epochs",
            str(int(args.epochs)),
            "--batch-size",
            str(int(args.batch_size)),
            "--device",
            str(args.device),
            "--split-column",
            "split",
            "--train-value",
            "train",
            "--val-value",
            "test",
            "--test-value",
            "none",
        ]
    )

    ckpt = run_dir / "training" / "checkpoints" / "checkpoint" / "checkpoint_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Expected checkpoint not found: {ckpt}")

    # Inference on TCGA test split (writes overlays + attention if deps available).
    _run_goldmark(
        [
            "inference",
            str(tcga_out_manifest),
            "--feature-dir",
            str(tcga_feature_dir),
            "--checkpoint",
            str(ckpt),
            "--output",
            str(os_output),
            "--run-name",
            str(args.run_name),
            "--target",
            "label_index",
            "--split-column",
            "split",
            "--split-value",
            "test",
            "--export-attention",
        ]
        + (["--no-overlays"] if args.no_overlays else [])
    )

    tcga_results = run_dir / "inference" / "inference" / "inference_results.csv"
    if not tcga_results.exists():
        raise FileNotFoundError(f"TCGA inference results not found: {tcga_results}")
    if not args.no_plots:
        _maybe_write_roc_pr_plots(tcga_results, run_dir / "plots", title=f"TCGA {str(args.gene).upper()} ({args.encoder})")

    # External IMPACT inference (default: 1 slide).
    impact_manifest_path = Path(args.impact_manifest).expanduser().resolve()
    impact_root = Path(args.impact_root).expanduser().resolve()
    impact_rows, impact_feature_dir = _load_impact_manifest(
        impact_manifest_path,
        gene=str(args.gene),
        impact_root=impact_root,
        encoder=str(args.encoder),
        dmp_assay_id=args.impact_dmp_assay_id,
        limit=int(args.impact_limit),
    )
    impact_out_manifest = data_dir / f"impact_external_{str(args.gene).upper()}_{args.encoder}.csv"
    try:
        import pandas as pd

        pd.DataFrame(impact_rows).to_csv(impact_out_manifest, index=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to write IMPACT smoke manifest: {impact_out_manifest}: {exc}") from exc
    print(f"[impact] External manifest: {impact_out_manifest} (rows={len(impact_rows)})")

    external_dir = run_dir / "external_inference" / "IMPACT"
    external_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    runner = InferenceRunner(
        manifest=pd.read_csv(impact_out_manifest),
        feature_dir=impact_feature_dir,
        checkpoint_path=ckpt,
        output_dir=external_dir,
        target_column="label_index",
        config=InferenceConfig(
            split_column="split",
            split_value="external",
            generate_overlays=not bool(args.no_overlays),
            export_attention=True,
        ),
    )
    impact_results = runner.run()
    print("\nTCGA→IMPACT smoke test complete.")
    print(f"- Run dir: {run_dir}")
    print(f"- TCGA manifest: {tcga_out_manifest}")
    print(f"- TCGA checkpoint: {ckpt}")
    print(f"- TCGA inference results: {tcga_results}")
    print(f"- IMPACT manifest: {impact_out_manifest}")
    print(f"- IMPACT inference results: {impact_results}")
    print(f"- IMPACT attention dir: {external_dir / 'attention'}")
    if not args.no_overlays:
        print(f"- IMPACT overlays dir: {external_dir / 'overlays'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
