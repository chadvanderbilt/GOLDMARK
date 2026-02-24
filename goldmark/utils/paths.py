from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PipelinePaths:
    """Resolve canonical directories used across the pipeline."""

    root: Path
    run_name: str
    stage: Optional[str] = None

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        self.stage = self.stage or "pipeline"

    @property
    def run_dir(self) -> Path:
        return self.root / self.run_name

    @property
    def stage_dir(self) -> Path:
        return self.run_dir / self.stage

    @property
    def tiles_dir(self) -> Path:
        return self.stage_dir / "tiles"

    @property
    def features_dir(self) -> Path:
        if (self.stage or "").lower() == "features":
            return self.stage_dir
        return self.stage_dir / "features"

    @property
    def checkpoints_dir(self) -> Path:
        return self.stage_dir / "checkpoints"

    @property
    def inference_dir(self) -> Path:
        return self.stage_dir / "inference"

    def ensure(self) -> None:
        for directory in [
            self.run_dir,
            self.stage_dir,
            self.tiles_dir,
            self.features_dir,
            self.checkpoints_dir,
            self.inference_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
