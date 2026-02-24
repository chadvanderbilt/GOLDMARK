from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    feature_dir: Optional[Path]
    target_column: str
    slide_id_column: str = "slide_id"
    split_column: Optional[str] = None
    subset_value: Optional[str] = None


class SlideLevelDataset(Dataset):
    """Dataset for loading per-slide feature tensors."""

    def __init__(self, manifest: pd.DataFrame, config: DatasetConfig) -> None:
        self.manifest = manifest.copy()
        self.config = config
        self.feature_dir = Path(config.feature_dir) if config.feature_dir else None
        if self.feature_dir:
            self.feature_dir.mkdir(parents=True, exist_ok=True)
        self.naming_pattern = self._detect_pattern()
        self.records = self._filter_manifest()
        self._format_notices: set[str] = set()
        self._degenerate_notices: set[str] = set()
        self._degenerate_paths: dict[str, str] = {}
        self._degenerate_threshold = 1e-6

    def _filter_manifest(self) -> pd.DataFrame:
        df = self.manifest
        if self.config.split_column and self.config.subset_value is not None:
            df = df[df[self.config.split_column] == self.config.subset_value].copy()
        feature_paths = None
        if "feature_path" in df.columns:
            feature_paths = df["feature_path"].astype(str)
        elif "slide_path" in df.columns and df["slide_path"].astype(str).str.endswith(".pt").all():
            feature_paths = df["slide_path"].astype(str)

        if feature_paths is not None:
            if self.feature_dir:
                feature_paths = feature_paths.apply(
                    lambda p: str((self.feature_dir / p).resolve()) if not Path(p).is_absolute() else p
                )
            df["feature_path"] = feature_paths
        else:
            if not self.feature_dir:
                raise ValueError(
                    "feature_dir must be provided when manifest does not contain feature_path or slide_path columns"
                )
            df["feature_path"] = df[self.config.slide_id_column].astype(str).apply(self._feature_path)
        def _is_failed_feature_path(path: str) -> bool:
            name = Path(path).name
            return ".FAILED" in name or name.endswith(".failed") or name.endswith(".FAILED")

        existing = df["feature_path"].map(lambda p: Path(p).exists() and not _is_failed_feature_path(str(p)))
        missing = df[~existing]
        if not missing.empty:
            print(f"Warning: dropping {len(missing)} slides without features")
        return df[existing].reset_index(drop=True)

    def _detect_pattern(self) -> tuple[str, str]:
        if not self.feature_dir or not self.feature_dir.exists():
            return ("", ".pt")
        files = [p.name for p in self.feature_dir.iterdir() if p.name.startswith("features_")]
        for name in files:
            stem = name[9:-3]
            if stem.startswith("imgP-"):
                return ("img", ".pt")
            if stem.startswith("P-"):
                return ("", ".pt")
            if stem.startswith("img"):
                return ("img", ".pt")
        return ("", ".pt")

    def _feature_path(self, feature_id: str) -> str:
        if not self.feature_dir:
            raise ValueError("feature_dir is required to construct feature paths when they are not embedded in manifest")
        prefix, suffix = self.naming_pattern
        # primary guess
        candidates = []
        if feature_id.startswith("P-") and prefix:
            candidates.append(self.feature_dir / f"features_{prefix}{feature_id}{suffix}")
        candidates.append(self.feature_dir / f"features_{feature_id}{suffix}")
        # some feature dirs store an extra 'img' prefix even when slide_id lacks it
        if not feature_id.startswith("img"):
            candidates.append(self.feature_dir / f"features_img{feature_id}{suffix}")
        for cand in candidates:
            if cand.exists():
                return str(cand)
        # fallback to first candidate
        return str(candidates[0])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        feature_path = row["feature_path"]
        content = self._safe_load(feature_path)
        slide_id = str(row[self.config.slide_id_column])
        features = self._extract_features(content, feature_path, slide_id)
        target = torch.tensor(row[self.config.target_column], dtype=torch.long)
        return {
            "features": features,
            "target": target,
            "slide_id": row[self.config.slide_id_column],
        }

    @staticmethod
    def _safe_load(feature_path: str) -> torch.Tensor:
        try:
            return torch.load(feature_path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(feature_path, map_location="cpu")

    def _extract_features(self, payload, feature_path: str, slide_id: Optional[str] = None) -> torch.Tensor:
        tensor = None
        source = None
        if torch.is_tensor(payload):
            tensor = payload
            source = "tensor"
        elif isinstance(payload, dict):
            for key in ("features", "embeddings", "data", "values"):
                value = payload.get(key)
                if torch.is_tensor(value):
                    tensor = value
                    source = f"dict[{key}]"
                    break
        elif isinstance(payload, (list, tuple)):
            first = payload[0] if payload else None
            if torch.is_tensor(first):
                tensor = first
                source = "sequence"
            elif isinstance(first, dict):
                return self._extract_features(first, feature_path)
        if tensor is None:
            raise ValueError(f"Unsupported feature payload in {feature_path}: type={type(payload)!r}")
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.dim() != 2:
            raise ValueError(f"Unexpected feature tensor shape {tuple(tensor.shape)} from {feature_path}")
        if source and source not in self._format_notices:
            print(
                f"[SlideLevelDataset] Loaded {source} features from {feature_path} with shape {tuple(tensor.shape)}"
            )
            self._format_notices.add(source)
        self._check_degenerate_features(tensor, feature_path, slide_id)
        return tensor

    def _check_degenerate_features(self, tensor: torch.Tensor, feature_path: str, slide_id: Optional[str]) -> None:
        if tensor.shape[0] < 2:
            return
        sample = tensor
        if sample.dtype == torch.float16:
            sample = sample.to(dtype=torch.float32)
        with torch.no_grad():
            variance = torch.var(sample, dim=0, unbiased=False).mean().item()
        if not math.isfinite(variance):
            variance = float("inf")
        if variance <= self._degenerate_threshold:
            key = str(feature_path)
            if key not in self._degenerate_notices:
                label = slide_id or Path(feature_path).stem
                print(
                    f"[SlideLevelDataset] Degenerate embeddings detected for {label}: {feature_path} "
                    f"(mean variance {variance:.3e}). All tiles appear identical."
                )
                self._degenerate_notices.add(key)
            self._degenerate_paths[key] = slide_id or Path(feature_path).stem

    @property
    def degenerate_entries(self) -> List[Dict[str, str]]:
        return [
            {"feature_path": path, "slide_id": slide_id}
            for path, slide_id in self._degenerate_paths.items()
        ]


def collate_fn(batch: List[dict]) -> dict:
    features = [item["features"] for item in batch]
    targets = torch.stack([item["target"] for item in batch])
    slide_ids = [item["slide_id"] for item in batch]
    max_tiles = max(feat.shape[0] for feat in features)
    feat_dim = features[0].shape[1]
    padded = torch.zeros(len(batch), max_tiles, feat_dim)
    mask = torch.zeros(len(batch), max_tiles, dtype=torch.bool)
    for i, feat in enumerate(features):
        length = feat.shape[0]
        padded[i, :length] = feat
        mask[i, :length] = 1
    return {
        "features": padded,
        "mask": mask,
        "targets": targets,
        "slide_ids": slide_ids,
    }
