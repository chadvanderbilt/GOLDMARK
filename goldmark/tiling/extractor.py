from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
import openslide
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, binary_erosion, square

from goldmark.utils.logging import get_logger


@dataclass
class TilingConfig:
    tile_size: int = 224
    stride: int = 224
    target_mpp: float = 0.5
    minimum_tissue_percentage: float = 0.5
    erosion: int = 2
    dilation: int = 4
    save_tiles: bool = False
    tile_format: str = "png"
    limit_tiles: Optional[int] = None
    random_seed: Optional[int] = None


@dataclass
class TileRecord:
    slide_id: str
    tile_id: str
    x: int
    y: int
    level: int
    width: int
    height: int
    tissue_fraction: float
    path: Optional[str]


@dataclass
class TileSet:
    slide_id: str
    records: List[TileRecord]
    manifest_path: Path


class SlideTiler:
    """Extract fixed-size tiles from whole-slide images."""

    def __init__(self, config: TilingConfig, output_dir: Path, log_level: str = "INFO") -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__, level=log_level)
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def tile_slide(self, slide_path: Path, slide_id: Optional[str] = None) -> TileSet:
        slide_path = Path(slide_path)
        slide_id = slide_id or slide_path.stem
        self.logger.info("Tiling slide %s", slide_id)

        with openslide.OpenSlide(str(slide_path)) as slide:
            level, mult = _find_level(slide, self.config.target_mpp, self.config.tile_size)
            mask = self._create_tissue_mask(slide, level, mult)
            records = self._extract_tiles(slide, slide_id, level, mask, mult)

        manifest_path = self._save_manifest(slide_id, records)
        return TileSet(slide_id=slide_id, records=records, manifest_path=manifest_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_tissue_mask(self, slide: openslide.OpenSlide, level: int, mult: float) -> np.ndarray:
        self.logger.debug("Generating tissue mask at level %s (multiplier %.3f)", level, mult)
        downsample = slide.level_downsamples[level]
        scale = self.config.tile_size / (self.config.tile_size * mult)
        target_size = (
            int(slide.dimensions[0] / downsample * scale),
            int(slide.dimensions[1] / downsample * scale),
        )

        thumbnail = slide.get_thumbnail(target_size)
        rgb = np.array(thumbnail.convert("RGB"))

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        marker_mask = _detect_marker(hsv, int(max(1, mult)))

        blurred = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
        if marker_mask is not None:
            masked = np.ma.masked_array(blurred, marker_mask > 0)
            thresh = threshold_otsu(masked.compressed()) if masked.count() else threshold_otsu(blurred)
        else:
            thresh = threshold_otsu(blurred)
        tissue_mask = (blurred < thresh).astype(np.uint8)

        if marker_mask is not None:
            tissue_mask[marker_mask > 0] = 0

        if self.config.erosion:
            tissue_mask = binary_erosion(tissue_mask, square(self.config.erosion))
        if self.config.dilation:
            tissue_mask = binary_dilation(tissue_mask, square(self.config.dilation))

        return tissue_mask.astype(np.uint8)

    def _extract_tiles(
        self,
        slide: openslide.OpenSlide,
        slide_id: str,
        level: int,
        mask: np.ndarray,
        mult: float,
    ) -> List[TileRecord]:
        tile_size = self.config.tile_size
        stride = self.config.stride
        downsample = slide.level_downsamples[level]
        tile_records: List[TileRecord] = []
        saved_tiles = 0

        mask_height, mask_width = mask.shape
        coords: Iterable[tuple[int, int]] = (
            (x, y)
            for y in range(0, mask_height - int(stride / downsample), int(stride / downsample))
            for x in range(0, mask_width - int(stride / downsample), int(stride / downsample))
        )

        for i, (mask_x, mask_y) in enumerate(coords):
            region = mask[
                mask_y : mask_y + int(tile_size / downsample),
                mask_x : mask_x + int(tile_size / downsample),
            ]
            tissue_fraction = float(region.sum()) / region.size
            if tissue_fraction < self.config.minimum_tissue_percentage:
                continue

            loc_x = int(mask_x * downsample)
            loc_y = int(mask_y * downsample)

            tile_id = f"{slide_id}_{i:05d}"
            tile_path: Optional[str] = None
            if self.config.save_tiles:
                tile_img = slide.read_region((loc_x, loc_y), level=level, size=(tile_size, tile_size)).convert("RGB")
                tile_dir = self.output_dir / slide_id
                tile_dir.mkdir(parents=True, exist_ok=True)
                tile_path = str((tile_dir / f"{tile_id}.{self.config.tile_format}").resolve())
                tile_img.save(tile_path)
                saved_tiles += 1

            tile_records.append(
                TileRecord(
                    slide_id=slide_id,
                    tile_id=tile_id,
                    x=loc_x,
                    y=loc_y,
                    level=level,
                    width=tile_size,
                    height=tile_size,
                    tissue_fraction=tissue_fraction,
                    path=tile_path,
                )
            )

            if self.config.limit_tiles and saved_tiles >= self.config.limit_tiles:
                break

        self.logger.info("Selected %d tiles for slide %s", len(tile_records), slide_id)
        return tile_records

    def _save_manifest(self, slide_id: str, records: List[TileRecord]) -> Path:
        manifest_dir = self.output_dir / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{slide_id}_tiles.csv"

        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "slide_id": r.slide_id,
                    "tile_id": r.tile_id,
                    "x": r.x,
                    "y": r.y,
                    "level": r.level,
                    "width": r.width,
                    "height": r.height,
                    "tissue_fraction": r.tissue_fraction,
                    "tile_path": r.path,
                }
                for r in records
            ]
        )
        df.to_csv(manifest_path, index=False)
        return manifest_path


def _find_level(slide: openslide.OpenSlide, target_mpp: float, patch_size: int) -> tuple[int, float]:
    mpp_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.5))
    downsample = target_mpp / mpp_x
    for level in reversed(range(slide.level_count)):
        level_downsample = slide.level_downsamples[level]
        mult = downsample / level_downsample
        pixel_difference = abs(mult * patch_size - patch_size) / patch_size
        if pixel_difference < 0.1 or downsample > level_downsample:
            mult = max(mult, 1.0)
            return level, mult
    return slide.level_count - 1, 1.0


def _detect_marker(hsv_image: np.ndarray, kernel_size: int) -> Optional[np.ndarray]:
    kernels = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    black_marker = cv2.inRange(hsv_image, np.array([0, 0, 0]), np.array([180, 255, 125]))
    blue_marker = cv2.inRange(hsv_image, np.array([90, 30, 30]), np.array([130, 255, 255]))
    green_marker = cv2.inRange(hsv_image, np.array([40, 30, 30]), np.array([90, 255, 255]))
    marker = cv2.bitwise_or(cv2.bitwise_or(black_marker, blue_marker), green_marker)
    marker = cv2.erode(marker, kernels)
    marker = cv2.dilate(marker, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size * 3, kernel_size * 3)))
    return marker if np.count_nonzero(marker) else None
