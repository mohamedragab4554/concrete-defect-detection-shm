"""Tiny YAML config loader with path resolution relative to the config file."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config. Keys ending in ``_dir``/``_yaml``/``_path`` are resolved
    relative to the repository root (the parent of ``configs/``) unless absolute."""
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}
    root = path.parent.parent
    for key, value in list(cfg.items()):
        if isinstance(value, str) and key.endswith(("_dir", "_yaml", "_path")):
            p = Path(value).expanduser()
            cfg[key] = str(p if p.is_absolute() else (root / p))
    return cfg
