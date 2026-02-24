from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


_ENV_PREFIX = "MILCFG__"


def load_config(path: Path, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load a YAML configuration file and merge overrides.

    Environment variables prefixed with ``MILCFG__`` override nested keys using
    double underscores as separators, e.g. ``MILCFG__TRAINING__EPOCHS=25``.
    """
    path = Path(path)
    data: Dict[str, Any] = {}
    if path.exists():
        with path.open("r") as handle:
            raw = yaml.safe_load(handle) or {}
            data.update(raw)

    env_updates: Dict[str, Any] = {}
    for key, value in os.environ.items():
        if key.startswith(_ENV_PREFIX):
            nested = key[len(_ENV_PREFIX) :].lower().split("__")
            cursor = env_updates
            for part in nested[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[nested[-1]] = _coerce(value)

    def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in updates.items():
            if isinstance(value, dict):
                target[key] = _deep_update(target.get(key, {}), value)
            else:
                target[key] = value
        return target

    if env_updates:
        _deep_update(data, env_updates)
    if overrides:
        _deep_update(data, overrides)
    return data


def _coerce(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
