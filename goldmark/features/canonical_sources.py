from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import torch
import numpy as np
from torch import nn
import torchvision.transforms as T
from torch.serialization import add_safe_globals

# Optional heavy imports are resolved lazily inside loader call sites to avoid
# paying the import cost when the canonical registry is unused.


@dataclass(frozen=True)
class CanonicalEncoderSpec:
    """Configuration for a canonical encoder bundle."""

    slug: str
    feature_dim: int
    tile_size: int
    loader: Callable[[torch.device], nn.Module]
    transform_factory: Callable[[nn.Module], T.Compose]


_DEFAULT_WEIGHTS_DIR = Path(os.environ.get("MIL_WEIGHTS_DIR", "weights")).expanduser()
_UNI_CHECKPOINT = _DEFAULT_WEIGHTS_DIR / "uni_checkpoint.pth"
_GIGAPATH_FT_CHECKPOINT = _DEFAULT_WEIGHTS_DIR / "gigapath_ft_checkpoint.pth"
_VIT_LARGE_CKPT = _DEFAULT_WEIGHTS_DIR / "vit_large"
_VIT_BASE_20X_CKPT = _DEFAULT_WEIGHTS_DIR / "vit_base_20x"
_OPENMIDNIGHT_REPO = "SophontAI/OpenMidnight"
_OPENMIDNIGHT_CHECKPOINT = "teacher_checkpoint_load.pt"


def _resolve_path(env_var: str, default: Path) -> Path:
    override = os.getenv(env_var)
    if override:
        candidate = Path(override).expanduser()
        if candidate.exists():
            return candidate
    return default.expanduser()


def _load_vit_large(device: torch.device) -> nn.Module:
    from transformers import ViTConfig, ViTForImageClassification

    checkpoint = _resolve_path("MIL_VIT_LARGE_CHECKPOINT", _VIT_LARGE_CKPT)
    if not checkpoint.exists():
        raise FileNotFoundError(f"ViT-Large checkpoint not found at {checkpoint}")
    config = ViTConfig.from_pretrained(str(checkpoint))
    config.output_hidden_states = True
    core = ViTForImageClassification.from_pretrained(str(checkpoint), config=config)
    return _ViTHiddenStateWrapper(core).to(device).eval()


def _load_vit_base(device: torch.device) -> nn.Module:
    from transformers import ViTConfig, ViTForImageClassification

    checkpoint = _resolve_path("MIL_VIT_BASE_CHECKPOINT", _VIT_BASE_20X_CKPT)
    if not checkpoint.exists():
        raise FileNotFoundError(f"ViT-Base checkpoint not found at {checkpoint}")
    config = ViTConfig.from_pretrained(str(checkpoint))
    config.output_hidden_states = True
    core = ViTForImageClassification.from_pretrained(str(checkpoint), config=config)
    return _ViTHiddenStateWrapper(core).to(device).eval()


def _load_uni(device: torch.device) -> nn.Module:
    import timm

    checkpoint = _resolve_path("MIL_UNI_CHECKPOINT", _UNI_CHECKPOINT)
    if not checkpoint.exists():
        raise FileNotFoundError(f"UNI checkpoint missing at {checkpoint}")
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _load_prov_gigapath(device: torch.device) -> nn.Module:
    import timm

    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    return model.to(device).eval()


def _fix_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    patched: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        clean_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        patched[clean_key] = value
    return patched


def _load_gigapath_ft(device: torch.device) -> nn.Module:
    import timm

    checkpoint = _resolve_path("MIL_GIGAPATH_FT_CHECKPOINT", _GIGAPATH_FT_CHECKPOINT)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Gigapath fine-tuned checkpoint missing at {checkpoint}")
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    try:
        add_safe_globals([np.core.multiarray.scalar])
    except Exception:
        pass
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("tile_model") if isinstance(payload, dict) else payload
    if isinstance(state, dict):
        state = _fix_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[canonical] gigapath_ft missing keys: {missing[:5]} ...")
    if unexpected:
        print(f"[canonical] gigapath_ft unexpected keys: {unexpected[:5]} ...")
    return model.to(device).eval()


def _load_virchow(device: torch.device) -> nn.Module:
    import timm
    from timm.layers import SwiGLUPacked

    core = timm.create_model(
        "hf-hub:paige-ai/Virchow",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    return _VirchowWrapper(core, register_tokens=0).to(device).eval()


def _load_virchow2(device: torch.device) -> nn.Module:
    import timm
    from timm.layers import SwiGLUPacked

    core = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    return _VirchowWrapper(core, register_tokens=4).to(device).eval()


def _load_h_optimus(device: torch.device) -> nn.Module:
    import timm

    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=False,
    )
    return model.to(device).eval()


def _load_openmidnight(device: torch.device) -> nn.Module:
    from huggingface_hub import hf_hub_download

    override = os.getenv("MIL_OPENMIDNIGHT_CHECKPOINT")
    if override:
        checkpoint_path = Path(override).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"OpenMidnight checkpoint not found at {checkpoint_path}")
    else:
        try:
            checkpoint_path = Path(
                hf_hub_download(repo_id=_OPENMIDNIGHT_REPO, filename=_OPENMIDNIGHT_CHECKPOINT)
            )
        except Exception as exc:
            raise RuntimeError(
                "OpenMidnight weights are gated on Hugging Face. "
                "Request access and login via huggingface-cli, or set MIL_OPENMIDNIGHT_CHECKPOINT."
            ) from exc

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14_reg", weights=None)
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if state is None and isinstance(payload, dict) and "model" in payload:
        state = payload["model"]
    if state is None:
        state = payload
    if isinstance(state, dict) and "pos_embed" in state:
        model.pos_embed = nn.Parameter(state["pos_embed"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[canonical] openmidnight missing keys: {missing[:5]} ...")
    if unexpected:
        print(f"[canonical] openmidnight unexpected keys: {unexpected[:5]} ...")
    return model.to(device).eval()


class _ToyAvgPoolEncoder(nn.Module):
    """A tiny, dependency-free encoder for smoke tests.

    This intentionally does *not* download weights and is deterministic given the input.
    """

    def __init__(self, grid: int = 8) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(inputs)
        return pooled.flatten(1)


def _load_toy(device: torch.device) -> nn.Module:
    model = _ToyAvgPoolEncoder(grid=8)
    return model.to(device).eval()


def _toy_transform(_model: nn.Module) -> T.Compose:
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
        ]
    )


class _ViTHiddenStateWrapper(nn.Module):
    """Project HuggingFace ViT outputs down to the CLS token features."""

    def __init__(self, backbone: "nn.Module") -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            outputs = self.backbone(pixel_values=pixel_values)
            hidden = outputs.hidden_states[-1][:, 0, :]
        return hidden


class _VirchowWrapper(nn.Module):
    """Match the Virchow feature construction used in MIL_CODE."""

    def __init__(self, backbone: "nn.Module", register_tokens: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_tokens = max(0, int(register_tokens))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            tokens = self.backbone(inputs)
            class_token = tokens[:, 0]
            start = self.register_tokens + 1
            patch_tokens = tokens[:, start:]
            pooled = patch_tokens.mean(dim=1)
        return torch.cat([class_token, pooled], dim=-1)


def _basic_transform(normalize_mean: Tuple[float, float, float], normalize_std: Tuple[float, float, float]) -> Callable[[nn.Module], T.Compose]:
    def factory(_model: nn.Module) -> T.Compose:
        return T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=normalize_mean, std=normalize_std),
            ]
        )

    return factory


def _gigapath_transform(_model: nn.Module) -> T.Compose:
    return T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _virchow_transform(model: nn.Module) -> T.Compose:
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    cfg = resolve_data_config(model.backbone.pretrained_cfg, model=model.backbone)
    return create_transform(**cfg)


def _hoptimus_transform(_model: nn.Module) -> T.Compose:
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=(0.707223, 0.578729, 0.703617),
                std=(0.211883, 0.230117, 0.177517),
            ),
        ]
    )


CANONICAL_SPECS: Dict[str, CanonicalEncoderSpec] = {
    "toy": CanonicalEncoderSpec(
        slug="toy",
        feature_dim=3 * 8 * 8,
        tile_size=224,
        loader=_load_toy,
        transform_factory=_toy_transform,
    ),
    "vit-large": CanonicalEncoderSpec(
        slug="vit-large",
        feature_dim=1024,
        tile_size=224,
        loader=_load_vit_large,
        transform_factory=_basic_transform((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ),
    "vit-base": CanonicalEncoderSpec(
        slug="vit-base",
        feature_dim=768,
        tile_size=224,
        loader=_load_vit_base,
        transform_factory=_basic_transform((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ),
    "uni": CanonicalEncoderSpec(
        slug="uni",
        feature_dim=1024,
        tile_size=224,
        loader=_load_uni,
        transform_factory=_basic_transform((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ),
    "prov-gigapath": CanonicalEncoderSpec(
        slug="prov-gigapath",
        feature_dim=1536,
        tile_size=224,
        loader=_load_prov_gigapath,
        transform_factory=lambda model: _gigapath_transform(model),
    ),
    "gigapath_ft": CanonicalEncoderSpec(
        slug="gigapath_ft",
        feature_dim=1536,
        tile_size=224,
        loader=_load_gigapath_ft,
        transform_factory=lambda model: _gigapath_transform(model),
    ),
    "virchow": CanonicalEncoderSpec(
        slug="virchow",
        feature_dim=2560,
        tile_size=224,
        loader=_load_virchow,
        transform_factory=_virchow_transform,
    ),
    "virchow2": CanonicalEncoderSpec(
        slug="virchow2",
        feature_dim=2560,
        tile_size=224,
        loader=_load_virchow2,
        transform_factory=_virchow_transform,
    ),
    "h-optimus-0": CanonicalEncoderSpec(
        slug="h-optimus-0",
        feature_dim=1536,
        tile_size=224,
        loader=_load_h_optimus,
        transform_factory=_hoptimus_transform,
    ),
    "openmidnight": CanonicalEncoderSpec(
        slug="openmidnight",
        feature_dim=1536,
        tile_size=224,
        loader=_load_openmidnight,
        transform_factory=_basic_transform((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ),
}


def load_canonical_encoder(name: str, device: torch.device) -> tuple[nn.Module, T.Compose, int]:
    """Return (model, transform, dim) for the requested canonical encoder."""

    slug = name.lower()
    if slug not in CANONICAL_SPECS:
        raise KeyError(f"Unknown canonical encoder '{name}'")
    spec = CANONICAL_SPECS[slug]
    model = spec.loader(device)
    transform = spec.transform_factory(model)
    return model, transform, spec.feature_dim


def list_canonical_names() -> Tuple[str, ...]:
    return tuple(CANONICAL_SPECS.keys())
