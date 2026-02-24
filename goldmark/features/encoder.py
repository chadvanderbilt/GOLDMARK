from __future__ import annotations

import json
import math
import time
import importlib
import importlib.util
import hashlib
import sys
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any, TYPE_CHECKING

import numpy as np
import openslide
import torch
import torchvision.transforms as T
from torch import nn

from goldmark.utils.logging import get_logger
from goldmark.features.canonical_sources import load_canonical_encoder

if TYPE_CHECKING:  # pragma: no cover
    from goldmark.features.progress import FeatureProgressTracker

REPO_ROOT = Path(__file__).resolve().parents[2]
ICML_ENCODER_PREFIXES = (
    "foundationmodel_",
    "foundation_model_",
    "foundationmodels_",
    "foundation-vit",
    "tcga_",
    "tcga-",
)


def _within_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPO_ROOT)
        return True
    except ValueError:
        return False


def _is_icml_encoder(name: Optional[str]) -> bool:
    slug = (name or "").strip().lower()
    if not slug:
        return False
    return any(slug.startswith(prefix) for prefix in ICML_ENCODER_PREFIXES)


@dataclass
class EncoderConfig:
    name: str = "prov-gigapath"
    batch_size: int = 256
    precision: str = "fp16"
    num_workers: int = 4
    custom_encoder: Optional[str] = None
    custom_encoder_script: Optional[str] = None
    custom_encoder_module: Optional[str] = None
    custom_encoder_kwargs: Dict[str, Any] = field(default_factory=dict)
    device: str = "auto"
    gpu_min_free_gb: float = 2.0
    tile_size: int = 224
    max_gpu_memory_gb: float = 70.0
    feature_variant: str = "cls_post"


@dataclass
class FeatureSet:
    slide_id: str
    feature_path: Path
    meta_path: Path


def _sha256sum(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _failed_feature_path(feature_path: Path, reason: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(reason or "failed")).strip("._-")
    safe = safe[:96] if len(safe) > 96 else safe
    return feature_path.with_name(f"{feature_path.stem}.FAILED_{safe}{feature_path.suffix}")




def _resolve_custom_script(script: str) -> Path:
    candidates = []
    raw = Path(script)

    if raw.is_absolute():
        candidates.append(raw.resolve())
    else:
        candidates.append(REPO_ROOT / script)
        candidates.append(REPO_ROOT / "custom_encoders" / script)
        candidates.append(Path(__file__).resolve().parent / script)

    for candidate in candidates:
        if candidate.is_dir():
            for filename in ("encoder.py", "custom_encoder.py", "__init__.py"):
                candidate_file = candidate / filename
                if candidate_file.exists():
                    if not _within_repo(candidate_file):
                        print(
                            f"[feature-extractor] Warning: custom encoder script '{candidate_file}' is outside MIL_CODE_BETA"
                        )
                    return candidate_file.resolve()
        elif candidate.exists():
            if not _within_repo(candidate):
                print(
                    f"[feature-extractor] Warning: custom encoder script '{candidate}' is outside MIL_CODE_BETA"
                )
            return candidate.resolve()

    raise FileNotFoundError(
        f"Unable to locate custom encoder script '{script}'. Checked within MIL_CODE_BETA and provided paths."
    )


def _import_custom_module(script: Optional[str], module_name: Optional[str]):
    if script:
        script_path = _resolve_custom_script(script)
        module_name_hint = f"goldmark.custom_encoder_{abs(hash(str(script_path))) & 0xffffffff:x}"
        spec = importlib.util.spec_from_file_location(module_name_hint, script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load custom encoder from {script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    if module_name:
        if str(REPO_ROOT) not in sys.path:
            sys.path.append(str(REPO_ROOT))
        module = importlib.import_module(module_name)
        module_path = getattr(module, "__file__", None)
        if module_path and not _within_repo(Path(module_path)):
            print(
                f"[feature-extractor] Warning: custom encoder module '{module_name}' is outside MIL_CODE_BETA"
            )
        return module
    raise ValueError("Either script or module_name must be provided for custom encoders")

def _free_vram_gb(idx: Optional[int] = None) -> float:
    try:
        free, _ = torch.cuda.mem_get_info(idx)
        return free / (1024 ** 3)
    except Exception:
        return 0.0


def _pick_device(min_free_gb: float, strict: bool = False) -> torch.device:
    if not torch.cuda.is_available():
        print('[feature-extractor] CUDA not available → CPU')
        return torch.device('cpu')

    best_idx: Optional[int] = None
    best_free = -1.0
    per_gpu: Dict[int, float] = {}
    for i in range(torch.cuda.device_count()):
        free = _free_vram_gb(i)
        per_gpu[i] = free
        if free > best_free:
            best_idx, best_free = i, free

    print(f"[feature-extractor] GPU free memory (GiB): {{ { {i: round(f, 2) for i, f in per_gpu.items()} } }}")

    if best_idx is None:
        return torch.device('cpu')

    if best_free >= float(min_free_gb):
        torch.cuda.set_device(best_idx)
        print(f'[feature-extractor] Selecting cuda:{best_idx} ({best_free:.2f} GiB free)')
        return torch.device(f'cuda:{best_idx}')

    if strict:
        print('[feature-extractor] All devices below threshold (strict) → CPU')
        return torch.device('cpu')

    print(f'[feature-extractor] Using cuda:{best_idx} despite low free memory ({best_free:.2f} GiB)')
    torch.cuda.set_device(best_idx)
    return torch.device(f'cuda:{best_idx}')


class FeatureExtractor:
    """Run foundation-model encoders over slide tiles."""

    def __init__(self, config: EncoderConfig, output_dir: Path, log_level: str = "INFO") -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__, level=log_level)
        self._degenerate_threshold = 1e-6
        if _is_icml_encoder(config.name):
            precision = str(getattr(config, "precision", "") or "").lower()
            if precision in {"fp16", "float16"}:
                self.logger.warning(
                    "ICML encoder detected (%s); forcing fp32 to avoid fp16 collapse.",
                    config.name,
                )
                self.config.precision = "fp32"
        if config.device == "auto":
            self.device = _pick_device(config.gpu_min_free_gb, strict=False)
        else:
            if config.device.startswith("cuda") and not torch.cuda.is_available():
                self.logger.warning("Requested CUDA device '%s' but CUDA is unavailable; falling back to CPU", config.device)
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(config.device)
        if self.device.type == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            except Exception:
                pass
        self.model, self.transform, self.feature_dim = self._load_model()
        self._checkpoint_path = self._resolve_checkpoint_path()
        self.logger.info("Loaded encoder %s (feature_dim=%d)", config.name, self.feature_dim)

    # ------------------------------------------------------------------
    def extract(
        self,
        slide_path: Path,
        tile_manifest: Path,
        slide_id: Optional[str] = None,
        progress: Optional["FeatureProgressTracker"] = None,
    ) -> FeatureSet:
        slide_path = Path(slide_path)
        tile_manifest = Path(tile_manifest)
        slide_id = slide_id or slide_path.stem

        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        feature_path = out_dir / f"features_{slide_id}.pt"
        meta_path = out_dir / f"features_{slide_id}.json"

        import pandas as pd

        df = pd.read_csv(tile_manifest)
        if "tile_id" not in df.columns:
            self.logger.warning("tile_id column missing in %s; synthesizing sequential IDs", tile_manifest)
            df = df.reset_index(drop=True)
            df.insert(0, "tile_id", df.index.astype(int))
        if "level" not in df.columns:
            self.logger.warning("level column missing in %s; defaulting to level 0", tile_manifest)
            df["level"] = 0
        tile_indices = df["tile_id"].tolist()
        total_tiles = len(df)

        batch_size = self._determine_batch_size()
        self.logger.info("Processing %d tiles with batch size %d", len(df), batch_size)

        features = []
        start_time = time.time()
        tracker_start = progress.begin_slide(slide_id, total_tiles) if progress else start_time
        embedding_stats: Dict[str, Optional[float]] = {}
        status = "ok"
        failure_reason: Optional[str] = None
        num_features: Optional[int] = None
        try:
            coords = df[["x", "y", "level"]].astype(int).to_numpy()

            from torch.utils.data import Dataset, DataLoader  # type: ignore

            class _TileDataset(Dataset):
                def __init__(self, slide_path: Path, coords: np.ndarray, tile_size: int, transform) -> None:
                    self.slide_path = str(slide_path)
                    self.coords = coords
                    self.tile_size = int(tile_size)
                    self.transform = transform
                    self._slide = None

                def __len__(self) -> int:
                    return int(self.coords.shape[0])

                def __getitem__(self, idx: int):
                    if self._slide is None:
                        self._slide = openslide.OpenSlide(self.slide_path)
                    x, y, level = self.coords[idx].tolist()
                    region = self._slide.read_region((int(x), int(y)), level=int(level), size=(self.tile_size, self.tile_size))
                    img = region.convert("RGB")
                    return self.transform(img)

            dataset = _TileDataset(slide_path, coords, self.config.tile_size, self.transform)
            num_workers = max(0, int(getattr(self.config, "num_workers", 0) or 0))
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(self.device.type == "cuda"),
                drop_last=False,
            )

            processed = 0
            for batch_tensor in loader:
                batch_tensor = batch_tensor.to(self.device, non_blocking=True)
                with torch.no_grad():
                    encoded = self._encode_batch(batch_tensor)
                features.append(encoded.cpu())
                processed += int(batch_tensor.shape[0])
                percent = (processed / total_tiles * 100.0) if total_tiles else 100.0
                self.logger.info("[progress] encoded %d/%d tiles (%.1f%%)", processed, total_tiles, percent)

            feature_tensor = torch.cat(features, dim=0)
            embedding_stats = self._summarize_embeddings(feature_tensor, slide_id)
            num_features = int(feature_tensor.shape[0])
            if num_features != total_tiles:
                status = "failed"
                failure_reason = f"tile_count_mismatch_features_{num_features}_tiles_{total_tiles}"
                self.logger.error(
                    "Tile-feature count mismatch for %s: features=%d tiles=%d",
                    slide_id,
                    num_features,
                    total_tiles,
                )
            if bool(embedding_stats.get("degenerate")):
                status = "failed"
                failure_reason = (failure_reason + ";degenerate") if failure_reason else "degenerate_embeddings"

            torch.save({"features": feature_tensor, "tile_ids": tile_indices}, feature_path)
        except Exception as exc:
            duration = time.time() - tracker_start if progress else 0.0
            if progress:
                progress.fail_slide(slide_id, total_tiles, duration, str(exc))
            raise

        final_feature_path = feature_path
        if status != "ok":
            failed_path = _failed_feature_path(feature_path, failure_reason or "failed")
            try:
                feature_path.replace(failed_path)
                final_feature_path = failed_path
            except OSError:
                final_feature_path = feature_path

        metadata: Dict[str, object] = {
            "slide_id": slide_id,
            "encoder": self.config.name,
            "tile_manifest": str(tile_manifest),
            "num_tiles": int(len(df)),
            "num_features": int(num_features if num_features is not None else -1),
            "feature_dim": int(self.feature_dim),
            "tile_size": self.config.tile_size,
            "precision": str(self.config.precision),
            "feature_variant": str(getattr(self.config, "feature_variant", "cls_post")),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": status,
            "failure_reason": failure_reason or "",
            "feature_path": str(final_feature_path),
        }
        if self._checkpoint_path:
            metadata["checkpoint_path"] = self._checkpoint_path
        metadata["embedding_variance"] = embedding_stats.get("mean_variance")
        metadata["embedding_degenerate"] = embedding_stats.get("degenerate")
        if embedding_stats:
            metadata["embedding_stats"] = embedding_stats
        try:
            metadata["feature_sha256"] = _sha256sum(final_feature_path)
            metadata["feature_bytes"] = int(final_feature_path.stat().st_size)
        except OSError:
            metadata["feature_sha256"] = ""
        meta_path.write_text(json.dumps(metadata, indent=2))

        if progress:
            duration = time.time() - tracker_start
            if status == "ok":
                progress.complete_slide(slide_id, total_tiles, duration, final_feature_path)
            else:
                progress.fail_slide(slide_id, total_tiles, duration, failure_reason or "failed")

        self.logger.info("Saved features to %s", final_feature_path)
        if status != "ok":
            raise ValueError(f"Feature extraction failed for {slide_id}: {failure_reason}")
        return FeatureSet(slide_id=slide_id, feature_path=final_feature_path, meta_path=meta_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> tuple[nn.Module, T.Compose, int]:
        if any([self.config.custom_encoder, self.config.custom_encoder_script, self.config.custom_encoder_module]):
            return self._load_custom_encoder()
        return _load_known_encoder(self.config.name, self.device)

    def _load_custom_encoder(self) -> tuple[nn.Module, T.Compose, int]:
        script = self.config.custom_encoder_script or self.config.custom_encoder
        module_name = self.config.custom_encoder_module
        module = _import_custom_module(script, module_name)

        if not hasattr(module, "load_model") or not hasattr(module, "get_transform"):
            raise AttributeError("Custom encoder module must define load_model and get_transform")

        kwargs = self.config.custom_encoder_kwargs or {}
        if isinstance(kwargs, dict) and ("model" in kwargs or "transform" in kwargs):
            model_kwargs = dict(kwargs.get("model", {}) or {})
            transform_kwargs = dict(kwargs.get("transform", {}) or {})
            direct_checkpoint = kwargs.get("checkpoint_path")
            if direct_checkpoint:
                model_kwargs.setdefault("checkpoint_path", direct_checkpoint)
                transform_kwargs.setdefault("checkpoint_path", direct_checkpoint)
        else:
            model_kwargs = kwargs if isinstance(kwargs, dict) else {}
            transform_kwargs = model_kwargs

        try:
            model = module.load_model(self.config.name, **model_kwargs)
        except TypeError:
            model = module.load_model(self.config.name)

        try:
            transform_result = module.get_transform(self.config.name, **transform_kwargs)
        except TypeError:
            transform_result = module.get_transform(self.config.name)

        if isinstance(transform_result, tuple):
            if len(transform_result) != 2:
                raise ValueError("Custom encoder get_transform must return (transform, feature_dim) tuple")
            transform, feature_dim = transform_result
        else:
            transform = transform_result
            feature_dim = getattr(model, "num_features", None) or getattr(model, "feature_dim", None)
            if feature_dim is None:
                raise ValueError("Custom encoder must provide feature_dim via get_transform or model attribute")

        model.eval().to(self.device)
        return model, transform, int(feature_dim)

    # ------------------------------------------------------------------
    def _encode_batch(self, batch: torch.Tensor) -> torch.Tensor:
        variant = str(getattr(self.config, "feature_variant", "cls_post") or "cls_post").lower()
        if variant not in ("cls_post", "cls_pre"):
            raise ValueError(f"Unsupported feature_variant '{variant}'. Expected 'cls_post' or 'cls_pre'.")

        if variant == "cls_pre":
            if not self._supports_icml_variant():
                raise ValueError(
                    "feature_variant=cls_pre requires an ICML-style VisionTransformer "
                    "with prepare_tokens/blocks/norm modules."
                )
            if self.config.precision == "fp16" and self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return self._encode_icml_variant(batch, variant)
            return self._encode_icml_variant(batch, variant)

        if self.config.precision == "fp16" and self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = self.model(batch)
        else:
            output = self.model(batch)

        if isinstance(output, (tuple, list)):
            output = output[0]
        if hasattr(output, "logits"):
            output = output.logits
        return output

    def _supports_icml_variant(self) -> bool:
        return all(
            hasattr(self.model, attr)
            for attr in ("prepare_tokens", "patch_drop", "norm_pre", "blocks", "norm")
        )

    def _encode_icml_variant(self, batch: torch.Tensor, variant: str) -> torch.Tensor:
        x = self.model.prepare_tokens(batch)
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)
        x = self.model.blocks(x)
        if variant == "cls_pre":
            tokens = x
        else:
            tokens = self.model.norm(x)
        if tokens.dim() != 3:
            tokens = tokens.reshape(tokens.shape[0], -1)
        return tokens[:, 0]

    def _resolve_checkpoint_path(self) -> Optional[str]:
        for attr in ("_checkpoint_path", "checkpoint_path"):
            value = getattr(self.model, attr, None)
            if value:
                return str(value)
        kwargs = self.config.custom_encoder_kwargs or {}
        if isinstance(kwargs, dict):
            direct = kwargs.get("checkpoint_path")
            if direct:
                return str(direct)
            model_block = kwargs.get("model")
            if isinstance(model_block, dict):
                nested = model_block.get("checkpoint_path")
                if nested:
                    return str(nested)
        return None

    def _summarize_embeddings(self, tensor: torch.Tensor, slide_id: str) -> Dict[str, Optional[float]]:
        stats: Dict[str, Optional[float]] = {
            "mean_variance": None,
            "degenerate": None,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "mean_abs": None,
            "max_abs": None,
            "tile_variance_mean": None,
            "tile_variance_min": None,
            "tile_variance_max": None,
            "l2_norm_mean": None,
            "l2_norm_std": None,
            "l2_norm_min": None,
            "l2_norm_max": None,
            "cosine_first_last": None,
            "finite_fraction": None,
            "nan_count": None,
            "nan_fraction": None,
            "inf_count": None,
            "inf_fraction": None,
            "zero_vector_count": None,
            "zero_vector_fraction": None,
            "sample_cosine_mean": None,
            "sample_cosine_std": None,
            "sample_cosine_max": None,
            "near_duplicate_fraction": None,
            "near_duplicate_threshold": None,
            "sample_cosine_pairs": None,
            "unique_rows_sample": None,
            "unique_rows_sample_size": None,
        }
        if tensor.dim() != 2 or tensor.shape[0] == 0:
            return stats
        sample = tensor
        if sample.dtype == torch.float16:
            sample = sample.to(dtype=torch.float32)
        with torch.no_grad():
            stats["mean"] = float(sample.mean().item())
            stats["std"] = float(sample.std(unbiased=False).item())
            stats["min"] = float(sample.min().item())
            stats["max"] = float(sample.max().item())
            abs_sample = sample.abs()
            stats["mean_abs"] = float(abs_sample.mean().item())
            stats["max_abs"] = float(abs_sample.max().item())
            finite_mask = torch.isfinite(sample)
            stats["finite_fraction"] = float(finite_mask.float().mean().item())
            stats["nan_count"] = float(torch.isnan(sample).sum().item())
            stats["nan_fraction"] = float(torch.isnan(sample).float().mean().item())
            stats["inf_count"] = float(torch.isinf(sample).sum().item())
            stats["inf_fraction"] = float(torch.isinf(sample).float().mean().item())
            tile_variance = torch.var(sample, dim=1, unbiased=False)
            stats["tile_variance_mean"] = float(tile_variance.mean().item())
            stats["tile_variance_min"] = float(tile_variance.min().item())
            stats["tile_variance_max"] = float(tile_variance.max().item())
            norms = torch.linalg.norm(sample, dim=1)
            stats["l2_norm_mean"] = float(norms.mean().item())
            stats["l2_norm_std"] = float(norms.std(unbiased=False).item())
            stats["l2_norm_min"] = float(norms.min().item())
            stats["l2_norm_max"] = float(norms.max().item())
            zero_mask = norms == 0
            stats["zero_vector_count"] = float(zero_mask.sum().item())
            stats["zero_vector_fraction"] = float(zero_mask.float().mean().item())
            if sample.shape[0] >= 2:
                denom = (norms[0] * norms[-1]).item()
                if denom > 0:
                    cosine = float(torch.dot(sample[0], sample[-1]).item() / denom)
                    stats["cosine_first_last"] = cosine
            cosine_sample_size = min(64, sample.shape[0])
            if cosine_sample_size >= 2:
                sub = sample[:cosine_sample_size]
                sub_norms = norms[:cosine_sample_size]
                valid = sub_norms > 0
                if valid.sum() >= 2:
                    sub = sub[valid]
                    sub_norms = sub_norms[valid]
                    sub_normed = sub / sub_norms.unsqueeze(1)
                    cos = sub_normed @ sub_normed.T
                    triu = torch.triu(torch.ones_like(cos, dtype=torch.bool), diagonal=1)
                    vals = cos[triu]
                    if vals.numel() > 0:
                        stats["sample_cosine_mean"] = float(vals.mean().item())
                        stats["sample_cosine_std"] = float(vals.std(unbiased=False).item())
                        stats["sample_cosine_max"] = float(vals.max().item())
                        threshold = 0.999
                        stats["near_duplicate_threshold"] = float(threshold)
                        stats["near_duplicate_fraction"] = float((vals > threshold).float().mean().item())
                        stats["sample_cosine_pairs"] = float(vals.numel())
            variance = torch.var(sample, dim=0, unbiased=False).mean().item()
            stats["mean_variance"] = float(variance)
            if math.isfinite(variance):
                degenerate = variance <= self._degenerate_threshold
                stats["degenerate"] = bool(degenerate)
                if degenerate:
                    self.logger.warning(
                        "Degenerate embeddings detected for %s (mean variance %.3e).",
                        slide_id,
                        variance,
                    )
            sample_size = min(128, sample.shape[0])
            if sample_size >= 2:
                unique = torch.unique(sample[:sample_size], dim=0).shape[0]
                stats["unique_rows_sample"] = float(unique)
                stats["unique_rows_sample_size"] = float(sample_size)
        self.logger.info(
            "[qc] %s var=%s std=%s norm_mean=%s mean_abs=%s zero_vec=%s/%s cos_mean=%s near_dup=%s",
            slide_id,
            f"{stats['mean_variance']:.3e}" if stats.get("mean_variance") is not None else "n/a",
            f"{stats['std']:.3e}" if stats.get("std") is not None else "n/a",
            f"{stats['l2_norm_mean']:.3e}" if stats.get("l2_norm_mean") is not None else "n/a",
            f"{stats['mean_abs']:.3e}" if stats.get("mean_abs") is not None else "n/a",
            int(stats["zero_vector_count"]) if stats.get("zero_vector_count") is not None else "n/a",
            int(sample.shape[0]) if sample is not None else "n/a",
            f"{stats['sample_cosine_mean']:.3f}" if stats.get("sample_cosine_mean") is not None else "n/a",
            f"{stats['near_duplicate_fraction']:.3f}" if stats.get("near_duplicate_fraction") is not None else "n/a",
        )
        return stats

    def _determine_batch_size(self) -> int:
        target = self.config.batch_size
        if self.device.type != "cuda":
            return target
        try:
            total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1e9
        except Exception:  # pragma: no cover - device queries may fail on CPU machines
            return target

        dummy = torch.randn(4, 3, self.config.tile_size, self.config.tile_size, device=self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = self._encode_batch(dummy)
        used = torch.cuda.max_memory_allocated() / 1e9
        if used == 0:
            return min(target, 512)

        available = min(total_memory * 0.8, self.config.max_gpu_memory_gb)
        provisional = int((available / used) * 4 * 0.6)
        return max(1, min(target, provisional))


# ------------------------------------------------------------------
# Built-in encoder registry
# ------------------------------------------------------------------

def _load_known_encoder(name: str, device: torch.device) -> tuple[nn.Module, T.Compose, int]:
    try:
        return load_canonical_encoder(name, device)
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unsupported encoder '{name}'") from exc
