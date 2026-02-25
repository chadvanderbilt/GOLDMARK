from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image

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
    """Extract fixed-size tiles from slides.

    - If OpenSlide can open the input, use WSI-aware tissue masking.
    - Otherwise fall back to Pillow and tile the raster image on a grid.
    """

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

        slide_handle = _try_open_openslide(slide_path)
        if slide_handle is None:
            image = Image.open(slide_path).convert("RGB")
            records = self._extract_tiles_from_image(image, slide_id)
        else:
            with slide_handle as slide:
                level, mult = _find_level(slide, self.config.target_mpp, self.config.tile_size)
                mask, mask_scale_x, mask_scale_y = self._create_tissue_mask(slide, level, mult)
                records = self._extract_tiles(
                    slide,
                    slide_id,
                    level,
                    mask,
                    mask_scale_x=mask_scale_x,
                    mask_scale_y=mask_scale_y,
                )

        manifest_path = self._save_manifest(slide_id, records)
        return TileSet(slide_id=slide_id, records=records, manifest_path=manifest_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_tiles_from_image(self, image: Image.Image, slide_id: str) -> List[TileRecord]:
        tile_size = int(self.config.tile_size)
        stride = int(self.config.stride)
        width, height = image.size

        tile_records: List[TileRecord] = []
        count = 0
        for y in range(0, max(1, height - tile_size + 1), stride):
            for x in range(0, max(1, width - tile_size + 1), stride):
                tile_id = f"{slide_id}_{count:05d}"
                tile_path: Optional[str] = None
                if self.config.save_tiles:
                    tile = image.crop((x, y, x + tile_size, y + tile_size))
                    if tile.size != (tile_size, tile_size):
                        padded = Image.new("RGB", (tile_size, tile_size), color=(255, 255, 255))
                        padded.paste(tile, (0, 0))
                        tile = padded
                    tile_dir = self.output_dir / slide_id
                    tile_dir.mkdir(parents=True, exist_ok=True)
                    tile_path = str((tile_dir / f"{tile_id}.{self.config.tile_format}").resolve())
                    tile.save(tile_path)

                tile_records.append(
                    TileRecord(
                        slide_id=slide_id,
                        tile_id=tile_id,
                        x=int(x),
                        y=int(y),
                        level=0,
                        width=tile_size,
                        height=tile_size,
                        tissue_fraction=1.0,
                        path=tile_path,
                    )
                )
                count += 1
                if self.config.limit_tiles and len(tile_records) >= int(self.config.limit_tiles):
                    self.logger.info("Reached limit_tiles=%s for %s", self.config.limit_tiles, slide_id)
                    return tile_records

        self.logger.info("Selected %d tiles for slide %s", len(tile_records), slide_id)
        return tile_records

    def _create_tissue_mask(self, slide: openslide.OpenSlide, level: int, mult: float) -> tuple[np.ndarray, float, float]:
        import cv2  # type: ignore
        from skimage.filters import threshold_otsu  # type: ignore
        from skimage.morphology import binary_dilation, binary_erosion, square  # type: ignore

        # Always build the tissue mask on a small thumbnail to avoid huge memory/time costs.
        # Coordinate mapping back to level-0 pixels is computed from the thumbnail size.
        self.logger.debug("Generating tissue mask for %s at max_dim=2048 (level=%s mult=%.3f)", slide, level, mult)
        thumbnail = slide.get_thumbnail((2048, 2048))
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
        # Remove near-white background (low saturation + high value).
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        background = (sat < 15) & (val > 220)
        tissue_mask[background] = 0

        if self.config.erosion:
            tissue_mask = binary_erosion(tissue_mask, square(self.config.erosion))
        if self.config.dilation:
            tissue_mask = binary_dilation(tissue_mask, square(self.config.dilation))

        tissue_mask = tissue_mask.astype(np.uint8)
        # Scale factors: level-0 pixels per mask pixel.
        height, width = tissue_mask.shape
        scale_x = float(slide.dimensions[0]) / float(width) if width else 1.0
        scale_y = float(slide.dimensions[1]) / float(height) if height else 1.0
        return tissue_mask, scale_x, scale_y

    def _extract_tiles(
        self,
        slide: openslide.OpenSlide,
        slide_id: str,
        level: int,
        mask: np.ndarray,
        *,
        mask_scale_x: float,
        mask_scale_y: float,
    ) -> List[TileRecord]:
        tile_size = self.config.tile_size
        stride = self.config.stride
        tile_records: List[TileRecord] = []
        height0 = int(slide.dimensions[1])
        width0 = int(slide.dimensions[0])
        downsample = float(slide.level_downsamples[level])
        tile_size0 = int(round(float(tile_size) * downsample))
        stride0 = int(round(float(stride) * downsample))
        mask_height, mask_width = mask.shape
        mask_tile_w = max(1, int(round(float(tile_size0) / float(mask_scale_x or 1.0))))
        mask_tile_h = max(1, int(round(float(tile_size0) / float(mask_scale_y or 1.0))))

        for y0 in range(0, max(1, height0 - tile_size0 + 1), max(1, stride0)):
            for x0 in range(0, max(1, width0 - tile_size0 + 1), max(1, stride0)):
                mask_x = int(float(x0) / float(mask_scale_x or 1.0))
                mask_y = int(float(y0) / float(mask_scale_y or 1.0))
                if mask_x < 0 or mask_y < 0:
                    continue
                if mask_x + mask_tile_w > mask_width or mask_y + mask_tile_h > mask_height:
                    continue
                region = mask[mask_y : mask_y + mask_tile_h, mask_x : mask_x + mask_tile_w]
                if region.size == 0:
                    continue
                tissue_fraction = float(region.sum()) / float(region.size)
                if tissue_fraction < self.config.minimum_tissue_percentage:
                    continue

                loc_x = int(x0)
                loc_y = int(y0)

                tile_id = f"{slide_id}_{len(tile_records):05d}"
                tile_path: Optional[str] = None
                if self.config.save_tiles:
                    tile_img = slide.read_region((loc_x, loc_y), level=level, size=(tile_size, tile_size)).convert("RGB")
                    tile_dir = self.output_dir / slide_id
                    tile_dir.mkdir(parents=True, exist_ok=True)
                    tile_path = str((tile_dir / f"{tile_id}.{self.config.tile_format}").resolve())
                    tile_img.save(tile_path)

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

                if self.config.limit_tiles and len(tile_records) >= int(self.config.limit_tiles):
                    self.logger.info("Reached limit_tiles=%s for %s", self.config.limit_tiles, slide_id)
                    self.logger.info("Selected %d tiles for slide %s", len(tile_records), slide_id)
                    return tile_records

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
    import openslide  # type: ignore

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
    import cv2  # type: ignore

    kernels = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    black_marker = cv2.inRange(hsv_image, np.array([0, 0, 0]), np.array([180, 255, 125]))
    blue_marker = cv2.inRange(hsv_image, np.array([90, 30, 30]), np.array([130, 255, 255]))
    green_marker = cv2.inRange(hsv_image, np.array([40, 30, 30]), np.array([90, 255, 255]))
    marker = cv2.bitwise_or(cv2.bitwise_or(black_marker, blue_marker), green_marker)
    marker = cv2.erode(marker, kernels)
    marker = cv2.dilate(marker, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size * 3, kernel_size * 3)))
    return marker if np.count_nonzero(marker) else None


def _try_open_openslide(slide_path: Path):
    """Return an OpenSlide handle if available, else None."""

    try:
        import openslide  # type: ignore
    except Exception:
        return None

    try:
        return openslide.OpenSlide(str(slide_path))
    except Exception:
        return None
