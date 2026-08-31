"""YAML configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cropai.utils.paths import repo_root


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _cfg(name: str) -> Path:
    return repo_root() / "configs" / name


def load_crop_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or _cfg("crop_config.yaml"))


def load_dataset_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or _cfg("dataset.yaml"))


def load_training_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or _cfg("training.yaml"))


def load_inference_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or _cfg("inference.yaml"))


def load_risk_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or _cfg("risk.yaml"))


def load_geo_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or _cfg("geo.yaml"))


def load_runtime_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or _cfg("runtime.yaml"))
