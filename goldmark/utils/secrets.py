from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

_DEFAULT_RELATIVE_PATH = Path("configs/secrets.env")

_LOADED_PATH: Optional[Path] = None


def load_secrets_env(
    path: Optional[Path] = None,
    *,
    override: bool = False,
    verbose: bool = False,
) -> Optional[Path]:
    """Load KEY=VALUE secrets into ``os.environ`` from a local .env-style file.

    This is a convenience helper for HPC / academic setups where users prefer to
    store tokens in a single non-committed file.

    - Defaults to ``configs/secrets.env`` under the repo root.
    - Existing environment variables take precedence unless ``override=True``.
    - Returns the loaded path, or ``None`` if nothing was loaded.
    """

    global _LOADED_PATH  # noqa: PLW0603

    if _LOADED_PATH is not None and path is None and not override:
        return _LOADED_PATH

    candidate_paths: list[Path] = []
    env_override = os.environ.get("GOLDMARK_SECRETS_ENV")
    if path is not None:
        candidate_paths.extend(_candidate_paths(Path(path)))
    elif env_override:
        candidate_paths.extend(_candidate_paths(Path(env_override)))
    else:
        repo_root = _infer_repo_root()
        if repo_root:
            candidate_paths.append(repo_root / _DEFAULT_RELATIVE_PATH)
        candidate_paths.append(Path.cwd() / _DEFAULT_RELATIVE_PATH)

    seen: set[Path] = set()
    for candidate in candidate_paths:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.exists():
            continue
        try:
            text = resolved.read_text()
        except OSError:
            continue
        updates = _parse_env(text)
        if not updates:
            continue
        for key, value in updates.items():
            if not override and key in os.environ:
                continue
            os.environ[key] = value
        _LOADED_PATH = resolved
        if verbose:
            print(f"[secrets] Loaded: {resolved}")
        return resolved
    return None


def _candidate_paths(path: Path) -> list[Path]:
    if path.is_absolute():
        return [path.expanduser()]
    repo_root = _infer_repo_root()
    candidates = [Path.cwd() / path]
    if repo_root:
        candidates.append(repo_root / path)
    return candidates


def _infer_repo_root() -> Optional[Path]:
    # goldmark/utils/secrets.py -> goldmark/utils -> goldmark -> repo root
    try:
        return Path(__file__).resolve().parents[2]
    except Exception:
        return None


def _parse_env(text: str) -> Dict[str, str]:
    updates: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        updates[key] = value
    return updates

