#!/usr/bin/env python3
"""Run GMA inference with TapFM-style artifacts and tile attention exports."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from goldmark.training.aggregators import LegacyGMAGatedAttention, create_aggregator


@dataclass(frozen=True)
class SlideEntry:
    slide_id: str
    feature_id: str
    feature_path: Path
    target: int
    slide_path: Optional[str] = None


class LegacyGMAClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, attn_dim: int, num_classes: int, dropout: bool = True) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(0.25))
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        if dropout:
            layers.append(nn.Dropout(0.25))
        layers.append(LegacyGMAGatedAttention(hidden_dim, attn_dim, dropout=dropout, n_tasks=1))
        self.attention_net = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_logits, features = self.attention_net(x)
        attn_logits = attn_logits.transpose(1, 2)
        weights = F.softmax(attn_logits, dim=2)
        pooled = torch.sum(weights.transpose(1, 2) * features, dim=1)
        logits = self.classifier(pooled)
        return weights.squeeze(1), pooled, logits


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def _infer_slide_id_from_feature_path(feature_path: Path) -> str:
    stem = feature_path.stem
    if stem.startswith("features_"):
        stem = stem[len("features_"):]
    if stem.startswith("img"):
        stem = stem[len("img"):]
    if "_" in stem:
        prefix, remainder = stem.split("_", 1)
        if remainder.startswith(prefix):
            stem = remainder
    return stem


def _detect_feature_prefix(feature_dir: Path) -> str:
    for entry in feature_dir.iterdir():
        if entry.name.startswith("features_img"):
            return "img"
    return ""


def _feature_id_for_slide_id(slide_id: str, prefix: str) -> str:
    if prefix and not slide_id.startswith(prefix):
        return f"{prefix}{slide_id}"
    return slide_id


def _load_feature_tensor(feature_path: Path) -> torch.Tensor:
    try:
        payload = torch.load(feature_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(feature_path, map_location="cpu")
    tensor = None
    if torch.is_tensor(payload):
        tensor = payload
    elif isinstance(payload, dict):
        for key in ("features", "embeddings", "data", "values"):
            value = payload.get(key)
            if torch.is_tensor(value):
                tensor = value
                break
    elif isinstance(payload, (list, tuple)) and payload:
        first = payload[0]
        if torch.is_tensor(first):
            tensor = first
        elif isinstance(first, dict):
            return _load_feature_tensor(first)
    if tensor is None:
        raise ValueError(f"Unsupported feature payload at {feature_path}")
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.dim() != 2:
        raise ValueError(f"Unexpected feature tensor shape {tuple(tensor.shape)} from {feature_path}")
    return tensor


def _load_checkpoint_state(ckpt_path: Path) -> Tuple[Dict[str, torch.Tensor], int]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict") or ckpt.get("model_state") or ckpt.get("model_state_dict")
    if state is None:
        raise ValueError(f"Checkpoint {ckpt_path} missing state_dict")
    epoch = ckpt.get("epoch")
    if epoch is None:
        match = re.search(r"(\\d+)", ckpt_path.stem)
        epoch = int(match.group(1)) if match else -1
    return _strip_module_prefix(state), int(epoch)


def _build_model_from_state(state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    if any(key.startswith("attention_net.0") for key in state_dict):
        input_dim = int(state_dict["attention_net.0.weight"].shape[1])
        hidden_dim = int(state_dict["attention_net.0.weight"].shape[0])
        attn_dim = int(state_dict["attention_net.6.attention_a.0.weight"].shape[0])
        num_classes = int(state_dict["classifier.weight"].shape[0])
        model = LegacyGMAClassifier(input_dim, hidden_dim, attn_dim, num_classes, dropout=True)
    else:
        input_dim = int(state_dict["fc1.weight"].shape[1])
        num_classes = int(state_dict["classifier.weight"].shape[0])
        model = create_aggregator("gma", feature_dim=input_dim, num_classes=num_classes, dropout=True)
    model.expected_input_dim = input_dim
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(f"State dict mismatch (missing={missing}, unexpected={unexpected})")
    return model


def _extract_tcga_tumor(project_root: Path) -> Optional[str]:
    match = re.match(r"TCGA-([A-Z0-9]+)_svs", project_root.name)
    if not match:
        return None
    return match.group(1)


def _resolve_impact_project_root(tcga_project_root: Path) -> Optional[Path]:
    tumor = _extract_tcga_tumor(tcga_project_root)
    if not tumor:
        return None
    base_root = tcga_project_root.parent.parent
    candidate = base_root / "IMPACT" / tumor
    return candidate if candidate.exists() else None


def _write_metrics(metrics: Dict[str, float], output_dir: Path, epoch: int) -> None:
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(metrics)
    payload["epoch"] = int(epoch)
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (metrics_dir / f"metrics_epoch_{epoch:03d}.json").write_text(json.dumps(payload, indent=2))


def _write_classification_report(y_true: Sequence[int], y_pred: Sequence[int], output_dir: Path, epoch: int) -> None:
    report_dir = output_dir / "classification_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(report_dir / f"epoch_{epoch:03d}.csv")


def _append_cumulative_results(
    metrics: Dict[str, float],
    counts: Dict[str, int],
    output_dir: Path,
    epoch: int,
    split_col: str,
    set_type: str,
    checkpoint_path: Path,
) -> None:
    report_dir = output_dir / "classification_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "auc": metrics.get("auc"),
        "ber": metrics.get("ber"),
        "fpr": metrics.get("fpr"),
        "fnr": metrics.get("fnr"),
        "tn": counts.get("tn"),
        "fp": counts.get("fp"),
        "fn": counts.get("fn"),
        "tp": counts.get("tp"),
        "n_ok": counts.get("n_total"),
        "n_total": counts.get("n_total"),
        "runtime_seconds": metrics.get("runtime_seconds"),
        "tile_ckpt": str(checkpoint_path),
        "slide_ckpt": str(checkpoint_path),
        "split_col": split_col,
        "set_type": set_type,
        "target": "target",
    }
    df = pd.DataFrame([payload])
    cumulative_path = report_dir / "cumulative_results.csv"
    header = not cumulative_path.exists()
    df.to_csv(cumulative_path, mode="a", header=header, index=False)


def _write_confusion_matrix_plot(matrix: np.ndarray, output_dir: Path, epoch: int) -> None:
    cm_dir = output_dir / "confusion_matrix"
    cm_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, int(val), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(cm_dir / f"epoch_{epoch:03d}.png")
    plt.close(fig)


def _write_roc_curve_plot(y_true: Sequence[int], y_score: Sequence[float], output_dir: Path, epoch: int) -> None:
    plot_dir = output_dir / "multiclass_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if len(set(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # lightweight JSON for web consumption
    (plot_dir / f"roc_curve_epoch_{epoch:03d}.json").write_text(
        json.dumps({"fpr": fpr.tolist(), "tpr": tpr.tolist(), "epoch": int(epoch)}, indent=2)
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, color="navy", lw=2)
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(plot_dir / f"roc_curve_epoch_{epoch:03d}.png")
    plt.close(fig)


def _write_probabilities(
    slide_ids: Sequence[str],
    targets: Sequence[int],
    scores: Sequence[float],
    output_dir: Path,
    epoch: int,
) -> None:
    probs = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    payload = pd.DataFrame(
        {
            "slide_id": slide_ids,
            "prob_class0": 1.0 - probs,
            "prob_class1": probs,
            "targets": targets,
        }
    )
    filename = output_dir / "probabilities_test_set.csv"
    payload.to_csv(filename, index=False)
    epoch_name = output_dir / f"probabilities_test_epoch_{epoch:03d}.csv"
    payload.to_csv(epoch_name, index=False)
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    payload.to_csv(pred_dir / epoch_name.name, index=False)


def _update_tile_attention(
    tile_manifest_path: Path,
    feature_id: str,
    attn_dir: Path,
    column: str,
    weights: np.ndarray,
    allow_mismatch: bool,
    overwrite: bool,
    attn_index: Optional[List[dict]] = None,
    attn_meta: Optional[Dict[str, object]] = None,
) -> bool:
    if not tile_manifest_path.exists():
        tiling_dir = tile_manifest_path.parent.parent.parent
        tcga_root = tiling_dir.parent
        if feature_id.startswith("imgTCGA-") and "TCGA-" in str(tile_manifest_path):
            sample_id = feature_id[len("img") :]
            coords_20x = tiling_dir / "tile_coords_20x.csv"
            coords_40x = tiling_dir / "tile_coords_40x.csv"
            for coords, size in ((coords_20x, 224), (coords_40x, 448)):
                if not coords.exists():
                    continue
                print(
                    f"[warn] Missing tile manifest for {feature_id}; rebuilding from {coords.name}",
                    flush=True,
                )
                _ensure_tcga_tile_manifests(
                    [sample_id],
                    tiling_dir,
                    size,
                    tcga_root,
                    tile_coords_path=coords,
                )
                if tile_manifest_path.exists():
                    break
        if not tile_manifest_path.exists():
            print(f"[warn] Missing tile manifest for {feature_id}: {tile_manifest_path}; skipping", flush=True)
            return False
    tile_df = pd.read_csv(tile_manifest_path)
    if len(weights) != len(tile_df):
        # Try to self-heal before falling back to trimming.
        if not allow_mismatch and feature_id.startswith("imgTCGA-") and "TCGA-" in str(tile_manifest_path):
            tiling_dir = tile_manifest_path.parent.parent.parent
            tcga_root = tiling_dir.parent
            sample_id = feature_id[len("img") :]
            alt_coords = tiling_dir / "tile_coords_40x.csv"
            if alt_coords.exists():
                print(
                    f"[warn] Tile/feature mismatch for {feature_id}: {len(weights)} vs {len(tile_df)}; "
                    f"rebuilding manifest from 40x coords",
                    flush=True,
                )
                if tile_manifest_path.exists():
                    tile_manifest_path.unlink()
                _ensure_tcga_tile_manifests(
                    [sample_id],
                    tiling_dir,
                    int(tile_df["tile_size"].iloc[0]) if "tile_size" in tile_df.columns else 224,
                    tcga_root,
                    tile_coords_path=alt_coords,
                )
                tile_df = pd.read_csv(tile_manifest_path)
        if len(weights) != len(tile_df):
            min_len = min(len(weights), len(tile_df))
            print(
                f"[warn] Tile/feature mismatch for {feature_id}: {len(weights)} vs {len(tile_df)}; trimming to {min_len}",
                flush=True,
            )
            tile_df = tile_df.iloc[:min_len].copy()
            weights = weights[:min_len]
    attn_dir.mkdir(parents=True, exist_ok=True)
    attn_path = attn_dir / f"{feature_id}_tiles_attn.csv"
    if attn_path.exists():
        try:
            df = pd.read_csv(attn_path)
        except pd.errors.EmptyDataError:
            print(f"[warn] Empty attn CSV for {feature_id}: {attn_path}; rebuilding from tile manifest", flush=True)
            df = tile_df.copy()
        if len(df) != len(tile_df):
            # Never drop existing columns. Try to align weights to the existing CSV order.
            if "tile_id" in df.columns and "tile_id" in tile_df.columns:
                tile_lookup = {}
                for idx, tile_id in enumerate(tile_df["tile_id"].tolist()):
                    tile_lookup[tile_id] = idx
                    try:
                        tile_lookup[int(tile_id)] = idx
                    except Exception:
                        pass
                aligned = []
                missing_ids = 0
                for tile_id in df["tile_id"].tolist():
                    idx = tile_lookup.get(tile_id)
                    if idx is None:
                        try:
                            idx = tile_lookup.get(int(tile_id))
                        except Exception:
                            idx = None
                    if idx is None:
                        aligned.append(float("nan"))
                        missing_ids += 1
                    else:
                        aligned.append(weights[idx])
                if missing_ids:
                    print(
                        f"[warn] Tile attention length mismatch for {feature_id}: "
                        f"{len(df)} vs {len(tile_df)}; {missing_ids} tile_ids missing during alignment",
                        flush=True,
                    )
                weights = np.asarray(aligned, dtype=float)
            else:
                min_len = min(len(df), len(tile_df))
                print(
                    f"[warn] Tile attention length mismatch for {feature_id}: {len(df)} vs {len(tile_df)}; trimming to {min_len}",
                    flush=True,
                )
                df = df.iloc[:min_len].copy()
                tile_df = tile_df.iloc[:min_len].copy()
                weights = weights[:min_len]
        if column in df.columns and not overwrite:
            return False
        df[column] = weights
    else:
        df = tile_df.copy()
        df[column] = weights
    df.to_csv(attn_path, index=False)
    if attn_index is not None:
        entry = {
            "slide_id": tile_df["sample_id"].iloc[0] if "sample_id" in tile_df.columns else feature_id,
            "feature_id": feature_id,
            "attn_file": str(attn_path),
            "attn_column": column,
            "tile_manifest": str(tile_manifest_path),
        }
        if attn_meta:
            entry.update(attn_meta)
        attn_index.append(entry)
    return True


def _ensure_tcga_tile_manifests(
    sample_ids: Sequence[str],
    tiling_dir: Path,
    tile_size: int,
    tcga_root: Path,
    tile_coords_path: Optional[Path] = None,
    chunksize: int = 500_000,
) -> Path:
    manifest_dir = tiling_dir / "tiles" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    filtered = []
    for sample_id in sample_ids:
        if not sample_id:
            continue
        if (manifest_dir / f"img{sample_id}_tiles.csv").exists():
            continue
        filtered.append(sample_id)
    if not filtered:
        return manifest_dir
    pending = {sample_id for sample_id in filtered}
    tile_counts: Dict[str, int] = {sample_id: 0 for sample_id in pending}
    tile_coords = tile_coords_path or (tiling_dir / "tile_coords_20x.csv")
    if not tile_coords.exists():
        print(f"[warn] Missing tile coords: {tile_coords}; skipping manifest generation", flush=True)
        return manifest_dir
    for chunk in pd.read_csv(tile_coords, chunksize=chunksize):
        chunk = chunk[chunk["sample_id"].isin(pending)]
        if chunk.empty:
            continue
        chunk["slide"] = chunk["slide"].astype(str).apply(
            lambda value: str((tcga_root / value.lstrip("./")).resolve()) if value.startswith("./") else value
        )
        for sample_id, group in chunk.groupby("sample_id"):
            out_path = manifest_dir / f"img{sample_id}_tiles.csv"
            start = tile_counts[sample_id]
            group = group.reset_index(drop=True)
            group.insert(0, "tile_id", range(start, start + len(group)))
            tile_counts[sample_id] = start + len(group)
            group["level"] = 0
            group["tile_size"] = tile_size
            group["width"] = tile_size
            group["height"] = tile_size
            header = not out_path.exists()
            group.to_csv(out_path, mode="a", header=header, index=False)
    return manifest_dir


def _select_tcga_tile_coords_path(
    tiling_dir: Path,
    sample_ids: Sequence[str],
    tile_size: int,
    chunksize: int = 500_000,
) -> Path:
    candidates = []
    tile_20x = tiling_dir / "tile_coords_20x.csv"
    tile_40x = tiling_dir / "tile_coords_40x.csv"
    if tile_size >= 448:
        candidates = [tile_40x, tile_20x]
    else:
        candidates = [tile_20x, tile_40x]
    candidates = [path for path in candidates if path.exists()]
    if not candidates:
        raise FileNotFoundError(f"No tile coords found in {tiling_dir}")
    if not sample_ids:
        return candidates[0]
    pending = set(sample_ids)
    for path in candidates:
        for chunk in pd.read_csv(path, usecols=["sample_id"], chunksize=chunksize):
            if pending.intersection(set(chunk["sample_id"])):
                return path
    return candidates[0]


def _count_tile_coords(
    tile_coords_path: Path,
    sample_ids: Sequence[str],
    chunksize: int = 500_000,
) -> Dict[str, int]:
    counts = {sample_id: 0 for sample_id in sample_ids}
    if not tile_coords_path.exists() or not sample_ids:
        return counts
    sample_set = set(sample_ids)
    for chunk in pd.read_csv(tile_coords_path, usecols=["sample_id"], chunksize=chunksize):
        chunk = chunk[chunk["sample_id"].isin(sample_set)]
        if chunk.empty:
            continue
        for sample_id, count in chunk["sample_id"].value_counts().items():
            counts[sample_id] = counts.get(sample_id, 0) + int(count)
    return counts


def _group_tcga_samples_by_tile_coords(
    tiling_dir: Path,
    sample_ids: Sequence[str],
    feature_counts: Dict[str, int],
    tile_size: int,
    chunksize: int = 500_000,
) -> Dict[Path, List[str]]:
    tile_20x = tiling_dir / "tile_coords_20x.csv"
    tile_40x = tiling_dir / "tile_coords_40x.csv"
    candidates = [path for path in (tile_20x, tile_40x) if path.exists()]
    if not candidates:
        return {}
    counts_20x = _count_tile_coords(tile_20x, sample_ids, chunksize=chunksize) if tile_20x.exists() else {}
    counts_40x = _count_tile_coords(tile_40x, sample_ids, chunksize=chunksize) if tile_40x.exists() else {}
    prefer_40x = tile_size >= 448
    default_path = tile_40x if prefer_40x and tile_40x.exists() else tile_20x if tile_20x.exists() else candidates[0]
    grouped: Dict[Path, List[str]] = {}
    for sample_id in sample_ids:
        feature_count = feature_counts.get(sample_id)
        count_20x = counts_20x.get(sample_id)
        count_40x = counts_40x.get(sample_id)
        match_20x = feature_count is not None and count_20x == feature_count and count_20x > 0
        match_40x = feature_count is not None and count_40x == feature_count and count_40x > 0
        chosen = None
        if match_20x and not match_40x:
            chosen = tile_20x
        elif match_40x and not match_20x:
            chosen = tile_40x
        elif match_20x and match_40x:
            chosen = default_path
        else:
            if (count_20x or count_40x) and feature_count is not None:
                print(
                    f"[warn] No tile coord count match for {sample_id}: "
                    f"features={feature_count}, 20x={count_20x}, 40x={count_40x}; "
                    f"using {default_path.name}",
                    flush=True,
                )
            chosen = default_path
        if chosen is None:
            continue
        grouped.setdefault(chosen, []).append(sample_id)
    return grouped


def _ensure_feature_metadata(
    feature_dir: Path,
    feature_id: str,
    encoder: str,
    tile_manifest_path: Path,
    feature_dim: int,
    tile_size: int,
) -> Path:
    meta_path = feature_dir / f"features_{feature_id}.json"
    if meta_path.exists():
        return meta_path
    num_tiles = 0
    if tile_manifest_path.exists():
        with tile_manifest_path.open("r", encoding="utf-8") as handle:
            num_tiles = max(sum(1 for _ in handle) - 1, 0)
    payload = {
        "slide_id": feature_id,
        "encoder": encoder,
        "tile_manifest": str(tile_manifest_path),
        "num_tiles": int(num_tiles),
        "feature_dim": int(feature_dim),
        "tile_size": int(tile_size),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path.write_text(json.dumps(payload, indent=2))
    return meta_path


def _ensure_tcga_metadata_for_entries(
    entries: Sequence[SlideEntry],
    feature_dir: Path,
    tiling_dir: Path,
    tcga_root: Path,
    tile_size: int,
    chunksize: int = 500_000,
) -> None:
    """Make sure every TCGA entry has a feature metadata JSON by:
    1) finding which tile_coords file (20x or 40x) actually contains the sample,
    2) materializing the per-slide tile manifest if missing, and
    3) writing the metadata JSON with num_tiles/feature_dim.
    If no tile coords rows exist for a slide, it logs a warning and leaves it missing."""
    missing = [
        entry
        for entry in entries
        if not (feature_dir / f"features_{entry.feature_id}.json").exists()
    ]
    if not missing:
        return

    sample_to_entry: Dict[str, SlideEntry] = {}
    feature_dims: Dict[str, int] = {}
    feature_counts: Dict[str, int] = {}
    for entry in missing:
        if not entry.feature_path.exists():
            continue
        slide_id = entry.feature_id[len("img") :] if entry.feature_id.startswith("img") else entry.feature_id
        # Some TCGA feature files were prefixed with 'g' during earlier export; try both.
        alt_slide_id = slide_id[1:] if slide_id.startswith("gTCGA-") else slide_id
        sample_to_entry[slide_id] = entry
        feats = _load_feature_tensor(entry.feature_path)
        feature_counts[slide_id] = int(feats.shape[0])
        feature_dims[entry.feature_id] = int(feats.shape[1])
        if alt_slide_id != slide_id:
            sample_to_entry[alt_slide_id] = entry
            feature_counts[alt_slide_id] = int(feats.shape[0])

    if not sample_to_entry:
        return

    # Map sample_id -> (tile_coords_path, count)
    choices: Dict[str, Tuple[Path, int]] = {sid: (None, 0) for sid in sample_to_entry}
    for coords_path in [tiling_dir / "tile_coords_40x.csv", tiling_dir / "tile_coords_20x.csv"]:
        if not coords_path.exists():
            continue
        pending = {sid for sid, (p, _) in choices.items() if p is None}
        if not pending:
            break
        for chunk in pd.read_csv(coords_path, usecols=["sample_id"], chunksize=chunksize):
            chunk = chunk[chunk["sample_id"].isin(pending)]
            if chunk.empty:
                continue
            vc = chunk["sample_id"].value_counts()
            for sid, cnt in vc.items():
                # Prefer the first coords file that has rows; keep the count
                if choices[sid][0] is None:
                    choices[sid] = (coords_path, int(cnt))

    for sid, (coords_path, count) in choices.items():
        entry = sample_to_entry[sid]
        if coords_path is None or count == 0:
            print(f"[warn] No tile coords rows for {sid}; cannot create metadata", flush=True)
            continue
        manifest_dir = _ensure_tcga_tile_manifests(
            [sid], tiling_dir, tile_size, tcga_root, tile_coords_path=coords_path, chunksize=chunksize
        )
        tile_manifest = manifest_dir / f"img{sid}_tiles.csv"
        _ensure_feature_metadata(
            feature_dir,
            entry.feature_id,
            encoder="tcga",  # encoder name not used downstream from JSON
            tile_manifest_path=tile_manifest,
            feature_dim=feature_dims.get(entry.feature_id, 0),
            tile_size=tile_size,
        )


def _scan_tile_mismatches(
    slides: Sequence[SlideEntry],
    feature_dir: Path,
    allow_missing_json: bool,
) -> List[Dict[str, object]]:
    mismatches = []
    for entry in slides:
        if not entry.feature_path.exists():
            mismatches.append(
                {
                    "slide_id": entry.slide_id,
                    "feature_id": entry.feature_id,
                    "issue": "missing_feature",
                    "feature_path": str(entry.feature_path),
                }
            )
            continue
        meta_path = feature_dir / f"features_{entry.feature_id}.json"
        if not meta_path.exists():
            if allow_missing_json:
                mismatches.append(
                    {
                        "slide_id": entry.slide_id,
                        "feature_id": entry.feature_id,
                        "issue": "missing_metadata",
                        "metadata_path": str(meta_path),
                    }
                )
                continue
            raise FileNotFoundError(f"Missing feature metadata {meta_path}")
        metadata = json.loads(meta_path.read_text())
        tile_manifest = Path(metadata["tile_manifest"])
        if not tile_manifest.exists():
            mismatches.append(
                {
                    "slide_id": entry.slide_id,
                    "feature_id": entry.feature_id,
                    "issue": "missing_tile_manifest",
                    "tile_manifest": str(tile_manifest),
                }
            )
            continue
        tile_df = pd.read_csv(tile_manifest)
        features = _load_feature_tensor(entry.feature_path)
        if len(tile_df) != features.shape[0]:
            mismatches.append(
                {
                    "slide_id": entry.slide_id,
                    "feature_id": entry.feature_id,
                    "issue": "tile_feature_mismatch",
                    "feature_len": int(features.shape[0]),
                    "tile_len": int(len(tile_df)),
                    "tile_manifest": str(tile_manifest),
                }
            )
    return mismatches


def _build_cv_entries(
    split_df: pd.DataFrame,
    split_col: str,
    feature_dir: Path,
    prefix: str,
    normalized_manifest: Optional[pd.DataFrame],
    limit: Optional[int] = None,
) -> List[SlideEntry]:
    subset = split_df[split_df[split_col].astype(str).str.lower() == "test"].copy()
    if limit:
        subset = subset.head(limit)
    slide_paths = {}
    if normalized_manifest is not None and "DMP_ASSAY_ID" in normalized_manifest.columns:
        slide_paths = dict(
            zip(
                normalized_manifest["DMP_ASSAY_ID"].astype(str),
                normalized_manifest.get("slide_path", pd.Series(dtype=str)).astype(str),
            )
        )
    entries = []
    for _, row in subset.iterrows():
        slide_id = str(row["DMP_ASSAY_ID"])
        feature_id = _feature_id_for_slide_id(slide_id, prefix)
        feature_path = feature_dir / f"features_{feature_id}.pt"
        entries.append(
            SlideEntry(
                slide_id=slide_id,
                feature_id=feature_id,
                feature_path=feature_path,
                target=int(row["target"]),
                slide_path=slide_paths.get(slide_id),
            )
        )
    return entries


def _extract_slide_id_from_row(row: pd.Series) -> str:
    for key in ("slide_id", "sample_id", "case_id", "slideid", "DMP_ASSAY_ID"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        return str(value)
    slide_path = row.get("slide_path")
    if slide_path:
        return _infer_slide_id_from_feature_path(Path(slide_path))
    return ""


def _build_tcga_entries(
    tcga_manifest: pd.DataFrame,
    feature_dir: Optional[Path] = None,
    prefix: str = "",
    limit: Optional[int] = None,
) -> List[SlideEntry]:
    df = tcga_manifest.copy()
    if limit:
        df = df.head(limit)
    entries = []
    for _, row in df.iterrows():
        slide_id = _extract_slide_id_from_row(row)
        if not slide_id:
            continue
        if feature_dir is not None:
            feature_id = None
            # If slide_path is provided, prefer deriving the id directly from the filename to avoid losing sample info.
            slide_path_val = row.get("slide_path")
            if isinstance(slide_path_val, str) and slide_path_val:
                feature_path = Path(slide_path_val)
                feature_id = feature_path.stem
                if feature_id.startswith("features_"):
                    feature_id = feature_id[len("features_"):]
                # Re-base under the supplied feature_dir in case slide_path was absolute.
                feature_path = feature_dir / f"features_{feature_id}.pt"
            else:
                feature_id = _feature_id_for_slide_id(slide_id, prefix)
                feature_path = feature_dir / f"features_{feature_id}.pt"
        else:
            feature_path = Path(row["slide_path"])
            feature_id = feature_path.stem
            if feature_id.startswith("features_"):
                feature_id = feature_id[len("features_"):]
        target = row.get("label_index")
        if target is None or (isinstance(target, float) and np.isnan(target)):
            task = str(row.get("task", "")).lower()
            target = 1 if task == "positive" else 0
        entries.append(
            SlideEntry(
                slide_id=str(slide_id),
                feature_id=str(feature_id),
                feature_path=feature_path,
                target=int(target),
            )
        )
    return entries


def _build_impact_entries(
    impact_manifest: pd.DataFrame,
    target_col: str,
    feature_dir: Path,
    prefix: str,
    limit: Optional[int] = None,
) -> List[SlideEntry]:
    df = impact_manifest.copy()
    if target_col not in df.columns:
        raise ValueError(f"Impact manifest missing target column: {target_col}")
    df = df[df[target_col].notna()].copy()
    # Normalize target values so string labels like "Positive"/"Negative" are supported.
    def _coerce_target(value: object) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, (int, float)) and not (isinstance(value, float) and np.isnan(value)):
            try:
                return int(value)
            except Exception:
                pass
        text = str(value).strip().lower()
        if text in {"1", "pos", "positive", "true", "yes"}:
            return 1
        if text in {"0", "neg", "negative", "false", "no"}:
            return 0
        return None

    df[target_col] = df[target_col].apply(_coerce_target)
    df = df[df[target_col].notna()].copy()
    if limit:
        df = df.head(limit)
    entries = []
    for _, row in df.iterrows():
        slide_id = str(row.get("DMP_ASSAY_ID") or row.get("slide_id") or row.get("slideid") or "")
        if not slide_id:
            continue
        feature_id = _feature_id_for_slide_id(slide_id, prefix)
        feature_path = feature_dir / f"features_{feature_id}.pt"
        target_value = _coerce_target(row[target_col])
        if target_value is None:
            continue
        entries.append(
            SlideEntry(
                slide_id=slide_id,
                feature_id=feature_id,
                feature_path=feature_path,
                target=int(target_value),
                slide_path=row.get("slide_path"),
            )
        )
    return entries


def _load_external_config(cfg_path: Path) -> Tuple[pd.DataFrame, Optional[Path], Optional[Path], int]:
    if not cfg_path.exists():
        raise FileNotFoundError(f"External config not found: {cfg_path}")
    cfg = json.loads(cfg_path.read_text())
    manifest_path = cfg.get("tcga_manifest") or cfg.get("impact_manifest") or cfg.get("manifest")
    if not manifest_path:
        raise ValueError(f"External config missing manifest path: {cfg_path}")
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"External manifest not found: {manifest_path}")
    tiling_dir = Path(cfg.get("tiling_dir")) if cfg.get("tiling_dir") else None
    root_dir = Path(cfg.get("tcga_root") or cfg.get("impact_root") or cfg.get("root")) if (cfg.get("tcga_root") or cfg.get("impact_root") or cfg.get("root")) else None
    tile_size = int(cfg.get("tilesize") or cfg.get("tile_size") or 224)
    manifest = pd.read_csv(manifest_path)
    if "slide_path" not in manifest.columns:
        raise ValueError(f"External manifest missing slide_path column: {manifest_path}")
    if tiling_dir and not tiling_dir.exists():
        print(f"[warn] External tiling_dir not found: {tiling_dir}", flush=True)
        tiling_dir = None
    if root_dir and not root_dir.exists():
        print(f"[warn] External root not found: {root_dir}", flush=True)
        root_dir = None
    return (
        manifest,
        tiling_dir,
        root_dir,
        tile_size,
    )


def _resolve_project_context(gma_root: Path) -> Dict[str, Path]:
    gma_root = gma_root.resolve()
    encoder_dir = gma_root.parent
    target_dir = encoder_dir.parent
    checkpoints_dir = target_dir.parent
    project_dir = checkpoints_dir.parent
    cohort_dir = project_dir.parent
    return {
        "gma_root": gma_root,
        "encoder_dir": encoder_dir,
        "target_dir": target_dir,
        "checkpoints_dir": checkpoints_dir,
        "project_dir": project_dir,
        "cohort_dir": cohort_dir,
    }


def _gather_checkpoint_paths(
    checkpoint_dir: Path,
    epochs: Sequence[int],
) -> List[Tuple[int, Path]]:
    paths = []
    for epoch in epochs:
        path = _find_checkpoint_path(checkpoint_dir, epoch)
        if path is not None:
            paths.append((epoch, path))
    return paths


def _find_checkpoint_path(checkpoint_dir: Path, epoch: int) -> Optional[Path]:
    for suffix in (".pth", ".pt"):
        for name in (
            f"checkpoint_test_{epoch}{suffix}",
            f"checkpoint_test_{epoch:03d}{suffix}",
            f"checkpoint_epoch_{epoch}{suffix}",
            f"checkpoint_epoch_{epoch:03d}{suffix}",
        ):
            path = checkpoint_dir / name
            if path.exists():
                return path
    return None


def _should_skip_inference(metrics_path: Path, predictions_path: Path, expected_slides: int) -> bool:
    """
    Only skip if prior outputs exist and cover the expected number of slides.
    This avoids silently skipping partially-completed runs from earlier smoke tests.
    """
    if not metrics_path.exists():
        return False
    if not predictions_path.exists():
        return False
    try:
        import pandas as pd

        df = pd.read_csv(predictions_path)
        if len(df.index) >= expected_slides:
            return True
    except Exception:
        return False
    return False


def _attn_index_paths(attn_dir: Optional[Path], set_type: str, split_col: str, epoch: int) -> List[Path]:
    if attn_dir is None:
        return []
    split_tag = split_col or ""
    split_name = f"attn_index_{set_type}_{split_tag}_epoch_{epoch:03d}" if split_tag else ""
    legacy_name = f"attn_index_{set_type}_epoch_{epoch:03d}"
    names = []
    if split_name:
        names.append(split_name)
    if legacy_name not in names:
        names.append(legacy_name)
    paths = []
    for name in names:
        paths.append(attn_dir / f"{name}.parquet")
        paths.append(attn_dir / f"{name}.csv")
    return paths


def _attn_index_exists(attn_dir: Optional[Path], set_type: str, split_col: str, epoch: int) -> bool:
    for path in _attn_index_paths(attn_dir, set_type, split_col, epoch):
        if path.exists():
            return True
    return False


def _pick_best_epoch_from_results(report_dir: Path) -> Optional[int]:
    cumulative_path = report_dir / "cumulative_results.csv"
    if not cumulative_path.exists():
        df = pd.DataFrame()
    else:
        df = pd.read_csv(cumulative_path)
        if "auc" in df.columns and "epoch" in df.columns:
            df = df.dropna(subset=["auc"])
    # Fallback: derive best from standalone metrics files when cumulative is missing/empty.
    if df.empty:
        metrics_dir = report_dir.parent / "metrics"
        if metrics_dir.exists():
            rows = []
            for path in metrics_dir.glob("metrics_epoch_*.json"):
                try:
                    payload = json.loads(path.read_text())
                    epoch = int(payload.get("epoch") or re.search(r"(\\d+)", path.stem).group(1))
                    auc = payload.get("auc")
                    if auc is not None and not (isinstance(auc, float) and np.isnan(auc)):
                        rows.append({"epoch": epoch, "auc": float(auc)})
                except Exception:
                    continue
            if rows:
                df = pd.DataFrame(rows)
    if df.empty:
        return None
    best_row = df.loc[df["auc"].astype(float).idxmax()]
    best_epoch = int(best_row["epoch"])
    try:
        payload = best_row.to_dict()
        payload["epoch"] = int(payload.get("epoch", best_epoch))
        if "auc" in payload and payload["auc"] is not None:
            payload["auc"] = float(payload["auc"])
        (report_dir / "best_epoch.json").write_text(json.dumps(payload, indent=2))
    except Exception:
        pass
    return best_epoch


def _pick_best_epoch_from_training(split_dir: Path) -> Optional[int]:
    metrics_path = split_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        payload = json.loads(metrics_path.read_text())
    except Exception:
        return None
    if isinstance(payload, dict):
        best_epoch = payload.get("best_epoch")
        try:
            return int(best_epoch)
        except Exception:
            return None
    return None


def _compute_metrics(y_true: Sequence[int], y_score: Sequence[float]) -> Tuple[Dict[str, float], Dict[str, int], np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= 0.5).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel().tolist()
    counts = {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "n_total": int(len(y_true))}
    fpr_val = fp / (fp + tn) if (fp + tn) else 0.0
    fnr_val = fn / (fn + tp) if (fn + tp) else 0.0
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fpr_val),
        "fnr": float(fnr_val),
        "ber": float((fpr_val + fnr_val) / 2.0),
    }
    if len(set(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["auc"] = float("nan")
    return metrics, counts, matrix


def _run_inference(
    slides: Sequence[SlideEntry],
    model: nn.Module,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    split_col: str,
    checkpoint_path: Path,
    attn_dir: Optional[Path],
    attn_column: Optional[str],
    feature_dir: Path,
    allow_mismatch: bool,
    overwrite_attn: bool,
    prediction_schema: str,
    tumor_label: str,
    target: str,
    encoder: str,
    cohort_label: str,
    set_type: str,
) -> None:
    model.eval()
    y_true: List[int] = []
    y_score: List[float] = []
    y_pred: List[int] = []
    slide_ids: List[str] = []
    slide_paths: List[Optional[str]] = []
    n_tiles: List[int] = []
    attn_index: List[dict] = []
    attn_written = 0
    attn_skipped = 0
    attn_missing_meta = 0
    attn_missing_manifest = 0
    start = time.time()
    total = len(slides)
    for idx, entry in enumerate(slides, start=1):
        if not entry.feature_path.exists():
            continue
        features = _load_feature_tensor(entry.feature_path)
        expected_dim = getattr(model, "expected_input_dim", None)
        if expected_dim and features.shape[1] != expected_dim:
            print(
                f"[warn] Feature dim mismatch for {entry.feature_id}: got {features.shape[1]}, expected {expected_dim}; skipping slide",
                flush=True,
            )
            if idx == 1 or idx % 25 == 0 or idx == total:
                print(f"[progress] {idx}/{total} slides processed", flush=True)
            continue
        n_tiles.append(int(features.shape[0]))
        with torch.no_grad():
            logits_weights = model(features.unsqueeze(0).to(device))
            weights = logits_weights[0][0].detach().cpu().numpy()
            logits = logits_weights[2]
            probs = torch.softmax(logits, dim=1)
            score = float(probs[:, 1].item()) if probs.shape[1] > 1 else float(probs[:, 0].item())
        pred = 1 if score >= 0.5 else 0
        slide_ids.append(entry.slide_id)
        y_true.append(int(entry.target))
        y_score.append(score)
        y_pred.append(pred)
        slide_paths.append(entry.slide_path)
        if attn_dir and attn_column:
            meta_path = feature_dir / f"features_{entry.feature_id}.json"
            if not meta_path.exists():
                print(
                    f"[warn] Missing feature metadata for {entry.feature_id}: {meta_path}; skipping attention export",
                    flush=True,
                )
                attn_missing_meta += 1
                if idx == 1 or idx % 25 == 0 or idx == total:
                    print(f"[progress] {idx}/{total} slides processed", flush=True)
                continue
            metadata = json.loads(meta_path.read_text())
            tile_manifest = Path(metadata["tile_manifest"])
            if not tile_manifest.exists():
                print(
                    f"[warn] Missing tile manifest for {entry.feature_id}: {tile_manifest}; skipping attention export",
                    flush=True,
                )
                attn_missing_manifest += 1
                if idx == 1 or idx % 25 == 0 or idx == total:
                    print(f"[progress] {idx}/{total} slides processed", flush=True)
                continue
            try:
                with tile_manifest.open("r", encoding="utf-8", errors="ignore") as handle:
                    header_line = handle.readline()
                if not header_line.strip():
                    print(
                        f"[warn] Empty tile manifest for {entry.feature_id}: {tile_manifest}; skipping attention export",
                        flush=True,
                    )
                    attn_missing_manifest += 1
                    if idx == 1 or idx % 25 == 0 or idx == total:
                        print(f"[progress] {idx}/{total} slides processed", flush=True)
                    continue
            except Exception:
                print(
                    f"[warn] Unreadable tile manifest for {entry.feature_id}: {tile_manifest}; skipping attention export",
                    flush=True,
                )
                attn_missing_manifest += 1
                if idx == 1 or idx % 25 == 0 or idx == total:
                    print(f"[progress] {idx}/{total} slides processed", flush=True)
                continue
            wrote = _update_tile_attention(
                tile_manifest,
                entry.feature_id,
                attn_dir,
                attn_column,
                weights,
                allow_mismatch,
                overwrite_attn,
                attn_index=attn_index,
                attn_meta={
                    "tumor": tumor_label,
                    "target": target,
                    "encoder": encoder,
                    "split": split_col,
                    "cohort": cohort_label,
                    "set_type": set_type,
                    "epoch": int(epoch),
                },
            )
            if wrote:
                attn_written += 1
            else:
                attn_skipped += 1
        if idx == 1 or idx % 25 == 0 or idx == total:
            print(f"[progress] {idx}/{total} slides processed", flush=True)

    runtime = time.time() - start
    if attn_dir and attn_column:
        print(
            f"[attn] {set_type} {split_col} epoch {epoch:03d}: wrote={attn_written} skipped_existing={attn_skipped} "
            f"missing_meta={attn_missing_meta} missing_manifest={attn_missing_manifest}",
            flush=True,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_probabilities(slide_ids, y_true, y_score, output_dir, epoch)
    if not y_true:
        print(f"[warn] No valid slides processed for {output_dir} epoch {epoch}; skipping metrics", flush=True)
        return

    if prediction_schema == "impact":
        pred_df = pd.DataFrame(
            {
                "slide_id": slide_ids,
                "slide_path": slide_paths,
                "n_tiles_total": n_tiles,
                "n_tiles_used": n_tiles,
                "y_true": y_true,
                "y_score": y_score,
                "error": "",
                "epoch": epoch,
                "slide_runtime_sec": runtime / max(len(slide_ids), 1),
                "tile_bs": "",
                "cache_mode": "",
                "target_prob": y_score,
            }
        )
    else:
        pred_df = pd.DataFrame(
            {
                "slide_id": slide_ids,
                "y_true": y_true,
                "y_score": y_score,
                "error": "",
                "epoch": epoch,
            }
        )
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_dir / f"predictions_epoch_{epoch:03d}.csv", index=False)

    metrics, counts, matrix = _compute_metrics(y_true, y_score)
    metrics["runtime_seconds"] = float(runtime)
    _write_metrics(metrics, output_dir, epoch)
    _write_classification_report(y_true, y_pred, output_dir, epoch)
    _append_cumulative_results(metrics, counts, output_dir, epoch, split_col, "test", checkpoint_path)
    _write_confusion_matrix_plot(matrix, output_dir, epoch)
    _write_roc_curve_plot(y_true, y_score, output_dir, epoch)
    if attn_index and attn_dir:
        attn_dir.mkdir(parents=True, exist_ok=True)
        split_tag = split_col or ""
        idx_name = (
            f"attn_index_{set_type}_{split_tag}_epoch_{epoch:03d}"
            if split_tag
            else f"attn_index_{set_type}_epoch_{epoch:03d}"
        )
        idx_path = attn_dir / f"{idx_name}.parquet"
        try:
            pd.DataFrame(attn_index).to_parquet(idx_path, index=False)
        except Exception as e:
            print(f"[warn] Failed to write parquet index {idx_path}: {e}; writing CSV fallback", flush=True)
            pd.DataFrame(attn_index).to_csv(idx_path.with_suffix(".csv"), index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GMA inference and tile attention exports.")
    parser.add_argument("--gma-root", required=True, help="Path to <TARGET>_gma directory.")
    parser.add_argument("--external-config", help="Path to external inference config (tcga_run_config.json or impact_run_config.json).")
    parser.add_argument("--tcga-config", help="Deprecated: use --external-config.")
    parser.add_argument("--splits", help="Comma-separated split names (e.g. split_1_set).")
    parser.add_argument(
        "--cv-checkpoints", default="2,5,10,20,50,80,120", help="CV checkpoint epochs."
    )
    parser.add_argument(
        "--tcga-checkpoints",
        default="best,2,5,10,20,50,80,120",
        help="TCGA checkpoint epochs (best plus specific epochs).",
    )
    parser.add_argument("--device", default="cuda", help="Device for inference.")
    parser.add_argument("--max-slides", type=int, help="Limit slides per split for quick smoke tests.")
    parser.add_argument("--skip-attn", action="store_true", help="Skip tile attention exports.")
    parser.add_argument("--skip-cv", action="store_true", help="Skip CV inference (run external only).")
    parser.add_argument("--skip-external", action="store_true", help="Skip external cohort inference.")
    parser.add_argument("--allow-mismatch", action="store_true", help="Allow tile/feature length mismatch.")
    parser.add_argument("--overwrite-attn", action="store_true", help="Overwrite existing attention columns.")
    parser.add_argument("--scan-mismatches", action="store_true", help="Scan tile/feature mismatches and exit.")
    parser.add_argument("--keep-cumulative", action="store_true", help="Keep existing cumulative_results.csv (default resets).")
    args = parser.parse_args()

    context = _resolve_project_context(Path(args.gma_root))
    gma_root = context["gma_root"]
    target = context["target_dir"].name
    encoder = context["encoder_dir"].name
    cohort = context["cohort_dir"].name.lower()
    project_root = context["project_dir"]
    # Derive tumor label for naming: e.g., BLCA or TCGA-BLCA
    tumor_label = project_root.name.replace("_svs", "")

    feature_dir = project_root / "features" / encoder
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")
    prefix = _detect_feature_prefix(feature_dir)

    split_manifest = context["encoder_dir"] / "manifests" / f"{target}_all_splits.csv"
    if not split_manifest.exists():
        raise FileNotFoundError(f"Split manifest not found: {split_manifest}")
    split_df = pd.read_csv(split_manifest)
    split_columns = [col for col in split_df.columns if col.endswith("_set")]
    if args.splits:
        requested = {item.strip() for item in args.splits.split(",") if item.strip()}
        split_columns = [col for col in split_columns if col in requested]
    if not split_columns:
        raise ValueError("No split columns found")

    normalized_manifest = None
    normalized_path = project_root / "dashboard_manifests" / "normalized_manifest.csv"
    if normalized_path.exists():
        normalized_manifest = pd.read_csv(normalized_path)

    external_manifest = None
    external_tiling_dir = None
    external_root = None
    external_feature_dir = None
    external_prefix = None
    external_tile_size = 224
    external_mode = None
    external_cfg = args.external_config or args.tcga_config
    if external_cfg:
        external_cfg_path = Path(external_cfg)
        external_manifest, external_tiling_dir, external_root, external_tile_size = _load_external_config(
            external_cfg_path
        )
        # Infer which cohort the external config represents. When users pass
        # --external-config explicitly, we should not assume TCGA unconditionally:
        # - IMPACT-trained models typically validate on TCGA
        # - TCGA-trained models typically validate on IMPACT
        mode_hint = ""
        try:
            cfg_payload = json.loads(external_cfg_path.read_text())
            mode_hint = str(cfg_payload.get("mode") or "").strip().lower()
            if not mode_hint:
                if cfg_payload.get("impact_manifest") or cfg_payload.get("impact_root"):
                    mode_hint = "impact"
                elif cfg_payload.get("tcga_manifest") or cfg_payload.get("tcga_root"):
                    mode_hint = "tcga"
        except Exception:
            mode_hint = ""
        if mode_hint in {"impact", "tcga"}:
            external_mode = mode_hint
        else:
            external_mode = "tcga" if cohort == "impact" else "impact"
        if external_root is not None:
            candidate = external_root / "features" / encoder
            if candidate.exists():
                external_feature_dir = candidate
        if external_feature_dir is None and not external_manifest.empty:
            external_feature_dir = Path(external_manifest.iloc[0]["slide_path"]).parent
        if external_feature_dir is not None and external_feature_dir.exists():
            external_prefix = _detect_feature_prefix(external_feature_dir)
    elif cohort == "tcga":
        impact_root = _resolve_impact_project_root(project_root)
        impact_manifest_path = impact_root / "dashboard_manifests" / "normalized_manifest.csv" if impact_root else None
        if impact_manifest_path and impact_manifest_path.exists():
            external_manifest = pd.read_csv(impact_manifest_path)
            external_feature_dir = impact_root / "features" / encoder if impact_root else None
            if external_feature_dir and external_feature_dir.exists():
                external_prefix = _detect_feature_prefix(external_feature_dir)
            external_tiling_dir = impact_root / "tiling" if impact_root else None
            external_root = impact_root
            external_tile_size = 224
            external_mode = "impact"
    if args.skip_external:
        external_manifest = None
        external_mode = None

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    external_scanned = False

    for split_col in split_columns:
        split_dir = gma_root / split_col
        checkpoint_dir = split_dir / "checkpoint"
        cumulative_path = split_dir / "inference" / "test" / "classification_report" / "cumulative_results.csv"
        cumulative_reset = False
        cv_tokens = [item.strip().lower() for item in args.cv_checkpoints.split(",") if item.strip()]
        cv_epochs: List[int] = []
        if "best" in cv_tokens:
            best_epoch = _pick_best_epoch_from_training(split_dir)
            if best_epoch is not None:
                cv_epochs.append(int(best_epoch))
        for token in cv_tokens:
            if token.isdigit():
                cv_epochs.append(int(token))
        cv_epochs = sorted({epoch for epoch in cv_epochs if epoch is not None})
        cv_paths = _gather_checkpoint_paths(checkpoint_dir, cv_epochs)
        if not cv_paths:
            fallback_dir = split_dir
            cv_paths = _gather_checkpoint_paths(fallback_dir, cv_epochs)
            if cv_paths:
                checkpoint_dir = fallback_dir
            else:
                print(f"No CV checkpoints found in {checkpoint_dir} or {fallback_dir}")
        cv_entries = _build_cv_entries(split_df, split_col, feature_dir, prefix, normalized_manifest, args.max_slides)

        if args.scan_mismatches:
            mismatches = _scan_tile_mismatches(cv_entries, feature_dir, allow_missing_json=True)
            if mismatches:
                report_path = gma_root / f"tile_mismatch_{split_col}.csv"
                pd.DataFrame(mismatches).to_csv(report_path, index=False)
                print(f"Mismatch report written to {report_path}")
            if external_manifest is not None and not external_scanned:
                if external_feature_dir is None or external_tiling_dir is None or external_root is None:
                    print("[warn] External config incomplete; skipping external mismatch scan", flush=True)
                    external_scanned = True
                    continue
                if external_mode == "impact":
                    external_entries = _build_impact_entries(
                        external_manifest,
                        target,
                        external_feature_dir,
                        external_prefix or "",
                        args.max_slides,
                    )
                else:
                    external_entries = _build_tcga_entries(
                        external_manifest,
                        external_feature_dir,
                        external_prefix or "",
                        args.max_slides,
                    )
                _ensure_tcga_metadata_for_entries(
                    external_entries,
                    external_feature_dir,
                    external_tiling_dir,
                    external_root,
                    external_tile_size,
                )
                external_mismatches = _scan_tile_mismatches(
                    external_entries, external_feature_dir, allow_missing_json=True
                )
                if external_mismatches:
                    report_path = gma_root / "tile_mismatch_external.csv"
                    pd.DataFrame(external_mismatches).to_csv(report_path, index=False)
                    print(f"Mismatch report written to {report_path}")
                external_scanned = True
            continue

        if not args.skip_cv:
            if cv_paths:
                print(
                    f"[cv] {gma_root} {split_col} epochs={','.join(str(e) for e, _ in cv_paths)}",
                    flush=True,
                )
            for epoch, ckpt_path in cv_paths:
                print(
                    f"[cv] {split_col} epoch {epoch:03d} checkpoint={ckpt_path}",
                    flush=True,
                )
                metrics_path = split_dir / "inference" / "test" / "metrics" / f"metrics_epoch_{epoch:03d}.json"
                predictions_path = split_dir / "inference" / "test" / "predictions" / f"predictions_epoch_{epoch:03d}.csv"
                if not args.keep_cumulative and not cumulative_reset:
                    if cumulative_path.exists():
                        cumulative_path.unlink()
                    cumulative_reset = True
                state_dict, _ = _load_checkpoint_state(ckpt_path)
                model = _build_model_from_state(state_dict).to(device)
                attn_column = (
                    f"attn_{tumor_label}_{target}_{encoder}_{split_col.replace('_set','')}_cv_{cohort}_ckpt{epoch}"
                )
                attn_dir = None if args.skip_attn else gma_root / "tile_attn" / "inference"
                if _should_skip_inference(metrics_path, predictions_path, len(cv_entries)):
                    if args.skip_attn or _attn_index_exists(attn_dir, "cv", split_col, epoch):
                        print(f"[skip] CV metrics exists for {split_col} epoch {epoch:03d}", flush=True)
                        continue
                _run_inference(
                    cv_entries,
                    model,
                    device,
                    split_dir / "inference" / "test",
                    epoch,
                    split_col,
                    ckpt_path,
                    attn_dir,
                    attn_column,
                    feature_dir,
                    args.allow_mismatch,
                    args.overwrite_attn,
                    prediction_schema="impact",
                    tumor_label=tumor_label,
                    target=target,
                    encoder=encoder,
                    cohort_label=cohort,
                    set_type="cv",
                )

        if external_manifest is None or external_mode is None:
            continue

        if external_mode == "impact":
            if external_feature_dir is None:
                raise ValueError("External feature directory not resolved for IMPACT inference.")
            external_entries = _build_impact_entries(
                external_manifest,
                target,
                external_feature_dir,
                external_prefix or "",
                args.max_slides,
            )
        else:
            external_entries = _build_tcga_entries(
                external_manifest,
                external_feature_dir,
                external_prefix or "",
                args.max_slides,
            )
        external_skip_attn = args.skip_attn
        if not external_skip_attn:
            if external_feature_dir is None:
                print("[warn] External feature dir missing; skipping attention export", flush=True)
                external_skip_attn = True
            elif external_tiling_dir is None or external_root is None:
                print("[warn] External tiling/root missing; skipping attention export", flush=True)
                external_skip_attn = True
        if not external_skip_attn:
            missing_meta = [
                entry
                for entry in external_entries
                if not (external_feature_dir / f"features_{entry.feature_id}.json").exists()
            ]
            if missing_meta:
                _ensure_tcga_metadata_for_entries(
                    external_entries,
                    external_feature_dir,
                    external_tiling_dir,
                    external_root,
                    external_tile_size,
                )

        external_epochs: List[Tuple[int, Path]] = []
        best_epoch = _pick_best_epoch_from_results(split_dir / "inference" / "test" / "classification_report")
        for token in [item.strip().lower() for item in args.tcga_checkpoints.split(",") if item.strip()]:
            if token == "best":
                if best_epoch is None:
                    print(f"No best epoch available for {split_col}")
                else:
                    best_path = _find_checkpoint_path(checkpoint_dir, best_epoch)
                    if best_path is not None:
                        external_epochs.append((best_epoch, best_path))
            elif token.isdigit():
                epoch_val = int(token)
                path = _find_checkpoint_path(checkpoint_dir, epoch_val)
                if path is not None:
                    external_epochs.append((epoch_val, path))

        external_label = "tcga_inference" if cohort == "impact" else "impact_inference"
        external_schema = "tcga" if cohort == "impact" else "impact"
        if external_epochs:
            print(
                f"[external] {gma_root} {split_col} epochs={','.join(str(e) for e, _ in external_epochs)}",
                flush=True,
            )
        for epoch, ckpt_path in external_epochs:
            print(
                f"[external] {split_col} epoch {epoch:03d} checkpoint={ckpt_path}",
                flush=True,
            )
            state_dict, _ = _load_checkpoint_state(ckpt_path)
            model = _build_model_from_state(state_dict).to(device)
            if best_epoch is not None and epoch == best_epoch:
                external_dir = split_dir / external_label / f"ckpt_best_{epoch:03d}"
            else:
                external_dir = split_dir / external_label / f"ckpt_{epoch:03d}"
            infer_tag = "TCGA" if external_label == "tcga_inference" else "IMPACT"
            attn_column = (
                f"attn_{tumor_label}_{target}_{encoder}_{split_col.replace('_set','')}_inf_{infer_tag}_ckpt{epoch}"
            )
            attn_dir = None if external_skip_attn else gma_root / "tile_attn" / external_label
            metrics_path = external_dir / "metrics" / f"metrics_epoch_{epoch:03d}.json"
            predictions_path = external_dir / "predictions" / f"predictions_epoch_{epoch:03d}.csv"
            if _should_skip_inference(metrics_path, predictions_path, len(external_entries)):
                if external_skip_attn or _attn_index_exists(attn_dir, external_label, split_col, epoch):
                    print(f"[skip] External metrics exists for {split_col} epoch {epoch:03d}", flush=True)
                    continue
            _run_inference(
                external_entries,
                model,
                device,
                external_dir,
                epoch,
                split_col,
                ckpt_path,
                attn_dir,
                attn_column,
                external_feature_dir or feature_dir,
                args.allow_mismatch,
                args.overwrite_attn,
                prediction_schema=external_schema,
                tumor_label=tumor_label,
                target=target,
                encoder=encoder,
                cohort_label=external_label,
                set_type=external_label,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
