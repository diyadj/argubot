"""Config loading. Every script reads configs/config.yaml through here."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

DEFAULT_CONFIG_PATH = "configs/config.yaml"


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def flatten(cfg: Dict[str, Any], prefix: str = "") -> List[Tuple[str, Any]]:
    """Turn the nested config into (dotted.key, value) pairs, handy for the report table."""
    rows: List[Tuple[str, Any]] = []
    for key, value in cfg.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            rows.extend(flatten(value, prefix=f"{name}."))
        else:
            rows.append((name, value))
    return rows


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
