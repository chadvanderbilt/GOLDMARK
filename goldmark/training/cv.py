from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from goldmark.training.trainer import MILTrainer, TrainerConfig
from goldmark.utils.logging import get_logger
from datetime import datetime


LOGGER = get_logger(__name__)


def sanitize_name(name: str) -> str:
    return name.replace("/", "_")


def run_cross_validation(
    manifest: pd.DataFrame,
    feature_dir: Optional[Path],
    base_output_dir: Path,
    target_column: str,
    base_config: TrainerConfig,
    split_columns: Iterable[str],
    log_level: str = "INFO",
) -> pd.DataFrame:
    summary_rows: List[Dict[str, float]] = []
    base_output_dir = Path(base_output_dir)
    split_columns = list(split_columns)
    total_splits = len(split_columns) or 1
    plot_entries: List[Dict[str, object]] = []

    for idx, split_column in enumerate(split_columns, start=1):
        split_name = sanitize_name(split_column)
        split_dir = base_output_dir / split_name
        split_config = replace(base_config, split_column=split_column)
        LOGGER.info("=== Training split %d/%d (%s) ===", idx, total_splits, split_column)
        trainer = MILTrainer(
            manifest=manifest,
            feature_dir=feature_dir,
            output_dir=split_dir,
            target_column=target_column,
            config=split_config,
            log_level=log_level,
        )
        result = trainer.run()
        for entry in result.pop("plot_entries", []) or []:
            normalized = dict(entry)
            normalized.setdefault("split", split_column)
            normalized.setdefault("target", target_column)
            normalized.setdefault("encoder", base_config.encoder_name)
            normalized.setdefault("aggregator", base_config.aggregator)
            plot_entries.append(normalized)
        row: Dict[str, float] = {
            "split": split_column,
            "best_epoch": result.get("best_epoch", 0),
        }
        for key, value in (result.get("best_val_metrics") or {}).items():
            row[f"val_{key}"] = value
        for key, value in (result.get("test_metrics") or {}).items():
            row[f"test_{key}"] = value
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    report_dir = base_output_dir / "classification_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(report_dir / "cv_summary.csv", index=False)
    _write_plot_cache(base_output_dir, plot_entries)
    return summary_df


def _write_plot_cache(base_output_dir: Path, entries: List[Dict[str, object]]) -> None:
    cache_path = Path(base_output_dir) / "plot_cache.json"
    if not entries:
        if cache_path.exists():
            try:
                cache_path.unlink()
            except OSError:
                pass
        return
    payload = {
        "version": 1,
        "generated_at": datetime.utcnow().isoformat(),
        "entries": entries,
    }
    try:
        cache_path.write_text(json.dumps(payload, indent=2))
    except OSError:
        LOGGER.warning("Failed to write plot cache to %s", cache_path)
