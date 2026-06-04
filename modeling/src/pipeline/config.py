"""Loads config.yaml and exposes a typed config object."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return config as a plain dict. Paths inside are resolved relative to config.yaml."""
    cfg_path = Path(path) if path else _DEFAULT_CONFIG
    with open(cfg_path, "r") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    base = cfg_path.parent
    for key in ("raw_path", "train_path", "test_path"):
        cfg["data"][key] = str((base / cfg["data"][key]).resolve())

    cfg["model"]["save_path"] = str((base / cfg["model"]["save_path"]).resolve())

    # Only resolve file-based tracking URIs; leave sqlite:// and http:// untouched.
    tracking_uri: str = cfg["mlflow"]["tracking_uri"]
    if not ("://" in tracking_uri):
        cfg["mlflow"]["tracking_uri"] = str((base / tracking_uri).resolve())

    return cfg
