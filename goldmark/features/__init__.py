"""Feature extraction utilities."""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["EncoderConfig", "FeatureExtractor", "FeatureSet"]

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from .encoder import EncoderConfig, FeatureExtractor, FeatureSet


def __getattr__(name: str):
    if name in __all__:
        try:
            module = import_module(".encoder", __name__)
        except OSError as exc:
            raise OSError(
                "OpenSlide native library not found. Install libopenslide "
                "(e.g., `conda install -c conda-forge openslide`) or ensure "
                "it is on LD_LIBRARY_PATH before using feature extraction."
            ) from exc
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
