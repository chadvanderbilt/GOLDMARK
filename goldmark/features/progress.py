from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _timestamp() -> str:
    return datetime.utcnow().isoformat()


@dataclass
class FeatureProgressTracker:
    """Track per-slide progress for feature extraction jobs."""

    output_dir: Path
    encoder_name: str
    display_name: Optional[str] = None
    total_slides: int = 0
    progress_dir: Path = field(init=False)
    events_path: Path = field(init=False)
    summary_path: Path = field(init=False)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_slides: int = 0
    failed_slides: int = 0
    skipped_slides: int = 0
    total_seconds: float = 0.0
    total_tiles: int = 0
    last_slide_event: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.encoder_name = str(self.encoder_name or "").strip() or "encoder"
        self.display_name = self.display_name or self.encoder_name
        self.total_slides = max(int(self.total_slides or 0), 0)

        self.progress_dir = self.output_dir / ".progress"
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.progress_dir / "events.jsonl"
        self.summary_path = self.progress_dir / "summary.json"
        self._write_summary(initial=True)

    # ------------------------------------------------------------------
    def begin_slide(self, slide_id: str, tile_count: int) -> float:
        payload = {
            "event": "start",
            "slide_id": slide_id,
            "tile_count": int(tile_count or 0),
        }
        self._append_event(payload)
        return time.time()

    def complete_slide(
        self,
        slide_id: str,
        tile_count: int,
        duration_seconds: float,
        feature_path: Path,
    ) -> None:
        self.completed_slides += 1
        self.total_seconds += max(float(duration_seconds or 0.0), 0.0)
        self.total_tiles += int(tile_count or 0)
        event = {
            "event": "complete",
            "slide_id": slide_id,
            "tile_count": int(tile_count or 0),
            "duration_seconds": float(duration_seconds or 0.0),
            "feature_path": str(feature_path),
        }
        self.last_slide_event = dict(event)
        self._append_event(event)
        self._write_summary()

    def skip_slide(
        self,
        slide_id: str,
        tile_count: int,
        duration_seconds: float,
        feature_path: Optional[Path] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Record a slide that was skipped because features already exist."""
        self.skipped_slides += 1
        self.total_seconds += max(float(duration_seconds or 0.0), 0.0)
        self.total_tiles += int(tile_count or 0)
        event = {
            "event": "skip",
            "slide_id": slide_id,
            "tile_count": int(tile_count or 0),
            "duration_seconds": float(duration_seconds or 0.0),
        }
        if feature_path:
            event["feature_path"] = str(feature_path)
        if reason:
            event["reason"] = reason
        self.last_slide_event = dict(event)
        self._append_event(event)
        self._write_summary()

    def fail_slide(
        self,
        slide_id: str,
        tile_count: int,
        duration_seconds: float,
        error_message: str,
    ) -> None:
        self.failed_slides += 1
        event = {
            "event": "error",
            "slide_id": slide_id,
            "tile_count": int(tile_count or 0),
            "duration_seconds": float(duration_seconds or 0.0),
            "error": error_message,
        }
        self.last_slide_event = dict(event)
        self._append_event(event)
        self._write_summary()

    # ------------------------------------------------------------------
    def _append_event(self, payload: Dict[str, Any]) -> None:
        record = dict(payload)
        record["timestamp"] = _timestamp()
        try:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _write_summary(self, *, initial: bool = False) -> None:
        pending = max(
            self.total_slides - self.completed_slides - self.failed_slides - self.skipped_slides,
            0,
        )
        avg_seconds = (self.total_seconds / self.completed_slides) if self.completed_slides else None
        tiles_per_second = (self.total_tiles / self.total_seconds) if self.total_seconds else None
        eta_seconds = (avg_seconds * pending) if avg_seconds and pending else None

        payload = {
            "encoder": self.encoder_name,
            "display_name": self.display_name,
            "total_slides": self.total_slides,
            "completed_slides": self.completed_slides,
            "failed_slides": self.failed_slides,
            "skipped_slides": self.skipped_slides,
            "pending_slides": pending,
            "avg_seconds_per_slide": avg_seconds,
            "tiles_per_second": tiles_per_second,
            "eta_seconds": eta_seconds,
            "started_at": self.started_at.isoformat(),
            "updated_at": _timestamp(),
            "events_path": str(self.events_path),
            "last_slide": self.last_slide_event,
        }
        if initial and self.last_slide_event is None:
            payload["last_slide"] = None
        tmp_path = self.summary_path.with_suffix(".tmp")
        try:
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(self.summary_path)
        except OSError:
            pass


def load_progress_summary(encoder_dir: Path) -> Optional[Dict[str, Any]]:
    summary_path = Path(encoder_dir) / ".progress" / "summary.json"
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
