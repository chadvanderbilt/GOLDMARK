from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import openslide
import pandas as pd
import torch

from goldmark.training.aggregators import create_aggregator
from goldmark.training.datasets import DatasetConfig, SlideLevelDataset
from goldmark.utils.logging import get_logger


@dataclass
class InferenceConfig:
    split_column: str = "split"
    split_value: str = "test"
    threshold: float = 0.5
    generate_overlays: bool = True
    overlay_alpha: float = 0.6


class InferenceRunner:
    def __init__(
        self,
        manifest: pd.DataFrame,
        feature_dir: Optional[Path],
        checkpoint_path: Path,
        output_dir: Path,
        target_column: Optional[str] = None,
        config: Optional[InferenceConfig] = None,
        log_level: str = "INFO",
    ) -> None:
        self.manifest = manifest
        self.feature_dir = Path(feature_dir).resolve() if feature_dir else None
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or InferenceConfig()
        self.logger = get_logger(__name__, level=log_level)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.target_column = target_column or self.checkpoint.get("target_column")
        if not self.target_column:
            raise ValueError("target_column must be provided for inference")

    def run(self) -> Path:
        dataset = SlideLevelDataset(
            self.manifest,
            DatasetConfig(
                feature_dir=self.feature_dir,
                target_column=self.target_column,
                split_column=self.config.split_column,
                subset_value=self.config.split_value,
            ),
        )
        if len(dataset) == 0:
            raise ValueError("No slides available for inference with the provided split configuration")

        feature_dim = dataset[0]["features"].shape[1]
        checkpoint_cfg = self.checkpoint.get("config", {})
        aggregator_name = checkpoint_cfg.get("aggregator", "gma")
        dropout = checkpoint_cfg.get("dropout", True)
        num_classes = self.checkpoint.get("num_classes", 2)
        model = create_aggregator(aggregator_name, feature_dim=feature_dim, num_classes=num_classes, dropout=dropout)
        model.load_state_dict(self.checkpoint["model_state"])
        model.to(self.device)
        model.eval()

        results = []
        overlay_dir = self.output_dir / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)

        for sample in dataset:
            slide_id = sample["slide_id"]
            features = sample["features"].unsqueeze(0).to(self.device)
            target = int(sample["target"].item())
            with torch.no_grad():
                attention, _, logits = model(features)
                probs = torch.softmax(logits, dim=1)
                probability = float(probs[:, 1].item()) if probs.shape[1] > 1 else float(probs[:, 0].item())
                prediction = int(probability >= self.config.threshold)

            results.append(
                {
                    "slide_id": slide_id,
                    "probability": probability,
                    "prediction": prediction,
                    "target": target,
                }
            )

            if self.config.generate_overlays:
                try:
                    self._generate_overlay(slide_id, attention.squeeze(0), probability)
                except Exception as exc:  # pragma: no cover - visualization best effort
                    self.logger.warning("Overlay generation failed for %s: %s", slide_id, exc)

        results_df = pd.DataFrame(results)
        out_path = self.output_dir / "inference_results.csv"
        results_df.to_csv(out_path, index=False)
        (self.output_dir / "inference_config.json").write_text(json.dumps(asdict(self.config), indent=2))
        self.logger.info("Saved inference results to %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    def _generate_overlay(self, slide_id: str, attention: torch.Tensor, probability: float) -> None:
        if self.feature_dir is None:
            self.logger.debug("Feature directory not provided; skipping overlay for %s", slide_id)
            return
        feature_meta_path = self.feature_dir / f"features_{slide_id}.json"
        if not feature_meta_path.exists():
            self.logger.debug("No metadata for slide %s; skipping overlay", slide_id)
            return
        metadata = json.loads(feature_meta_path.read_text())
        tile_manifest_path = Path(metadata["tile_manifest"])
        tile_size = metadata.get("tile_size", 224)
        tile_df = pd.read_csv(tile_manifest_path)
        slide_path_series = self.manifest.loc[
            self.manifest["slide_id"].astype(str) == str(slide_id), "slide_path"
        ]
        if slide_path_series.empty:
            self.logger.debug("Slide path not found for %s; skipping overlay", slide_id)
            return
        slide_path = Path(slide_path_series.iloc[0])

        weights = attention.squeeze().detach().cpu().numpy()
        weights = weights[: len(tile_df)]  # match manifest length if padded
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)

        with openslide.OpenSlide(str(slide_path)) as slide:
            level = int(tile_df["level"].mode().iloc[0])
            downsample = slide.level_downsamples[level]
            dims = slide.level_dimensions[level]
            heatmap = np.zeros((dims[1], dims[0]), dtype=np.float32)
            counts = np.zeros_like(heatmap)

            for weight, (_, row) in zip(weights, tile_df.iterrows()):
                x = int(row["x"] / downsample)
                y = int(row["y"] / downsample)
                size = max(1, int(tile_size / downsample))
                heatmap[y : y + size, x : x + size] += weight
                counts[y : y + size, x : x + size] += 1

            counts[counts == 0] = 1
            heatmap /= counts
            heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=4)
            normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            overlay = (normalized * 255).astype(np.uint8)
            overlay = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)

            base = slide.read_region((0, 0), level, dims).convert("RGB")
            base_np = np.array(base)
            blended = (
                self.config.overlay_alpha * overlay + (1 - self.config.overlay_alpha) * base_np
            ).astype(np.uint8)

            overlay_path = self.output_dir / "overlays" / f"{slide_id}_prob_{probability:.3f}.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
            self.logger.debug("Saved overlay for %s to %s", slide_id, overlay_path)
