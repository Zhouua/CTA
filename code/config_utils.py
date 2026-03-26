from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_project_config(config_path: str | None = None) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).expanduser().resolve() if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config, path.parent


def get_section(config: dict[str, Any], key: str, default: Any | None = None) -> dict[str, Any]:
    value = config.get(key, default if default is not None else {})
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{key}' must be a mapping.")
    return value


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def resolve_paths(base_dir: Path, mapping: dict[str, Any], keys: list[str]) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for key in keys:
        if key not in mapping:
            raise KeyError(f"Missing config path key: {key}")
        resolved[key] = resolve_path(base_dir, mapping[key])
    return resolved


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
