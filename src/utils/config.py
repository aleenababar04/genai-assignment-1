"""YAML config loading/saving and device selection."""

from pathlib import Path

import torch
import yaml

# This file is <repo>/src/utils/config.py, so the repo root is two folders up
# from the folder that contains it.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path) -> dict:
    """Read a YAML file and return its contents as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # An empty YAML file loads as None; return an empty dict instead.
    return cfg if cfg is not None else {}


def save_config(cfg: dict, path) -> None:
    """Write a dict to a YAML file, creating parent folders if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def get_device() -> torch.device:
    """Return the GPU if one is available, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
