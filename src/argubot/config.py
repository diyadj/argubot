"""Config loading. Every script reads configs/config.yaml through here.

A config can start from another one with `extends:` and list only what it changes. That is how
configs/config_4b.yaml reuses every setting of the main config and swaps the model, the precision
and the output folders, so the 0.8B and 4B runs can never drift apart by accident.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

DEFAULT_CONFIG_PATH = "configs/config.yaml"


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    return _load(Path(path), seen=())


def _load(path: Path, seen: Tuple[Path, ...]) -> Dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(str(p) for p in (*seen, resolved))
        raise ValueError(f"These config files extend each other in a loop: {chain}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent is None:
        return cfg
    parent_path = _find_parent(str(parent), path)
    base = _load(parent_path, seen + (resolved,))
    return merge_configs(base, cfg, source=str(path), base_source=str(parent_path))


def _find_parent(name: str, child: Path) -> Path:
    """Look in the current folder first (scripts run from the repo root), then next to the child.

    Checking the current folder first is what lets scripts/violet_eval.sbatch copy a config into a
    results folder and still load it.
    """
    for candidate in (Path(name), child.parent / name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{child} says `extends: {name}`, but that file is not in the current folder or next to {child}. "
        "Run the command from the repo root."
    )


def merge_configs(
    base: Dict[str, Any],
    override: Dict[str, Any],
    *,
    source: str = "the override",
    base_source: str = "the base config",
    _prefix: str = "",
) -> Dict[str, Any]:
    """Return base with override laid on top. Nested sections merge key by key.

    Every key in the override has to exist in the base. A misspelt key (say `trainig:`) would
    otherwise be ignored without a word and the run would quietly use the base value.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        name = f"{_prefix}{key}"
        if key not in out:
            raise KeyError(
                f"{source} sets `{name}`, which is not a setting in {base_source}. Check the spelling, "
                f"or add it to {base_source} first."
            )
        if isinstance(value, dict) and isinstance(out[key], dict):
            out[key] = merge_configs(
                out[key], value, source=source, base_source=base_source, _prefix=f"{name}."
            )
        else:
            out[key] = copy.deepcopy(value)
    return out


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
