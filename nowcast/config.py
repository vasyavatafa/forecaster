"""Loads default per-method hyperparameters from config/default_params.yaml
(single default values, not the search grids used by grid_search --
those live as `param_grid` on each method class in methods.py /
nn_method.py)."""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default_params.yaml"


def load_default_params(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}
