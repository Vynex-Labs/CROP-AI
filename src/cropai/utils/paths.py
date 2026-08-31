"""Repository path resolution. No machine-specific absolute paths."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("CROP_AI_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "MASTER_PROMPT.md").exists() and (parent / "configs").is_dir():
            return parent
    return Path.cwd().resolve()


def data_dir() -> Path:
    env = os.environ.get("CROP_AI_DATA_DIR")
    if env:
        p = Path(env).expanduser()
        return p if p.is_absolute() else repo_root() / p
    return repo_root() / "data"


def runs_dir() -> Path:
    env = os.environ.get("CROP_AI_RUNS_DIR", "runs")
    p = Path(env).expanduser()
    return p if p.is_absolute() else repo_root() / p


def models_dir() -> Path:
    env = os.environ.get("CROP_AI_MODELS_DIR", "models")
    p = Path(env).expanduser()
    return p if p.is_absolute() else repo_root() / p


def logs_dir() -> Path:
    env = os.environ.get("CROP_AI_LOGS_DIR", "logs")
    p = Path(env).expanduser()
    return p if p.is_absolute() else repo_root() / p


def ensure_runtime_dirs() -> dict[str, Path]:
    mapping = {
        "data": data_dir(),
        "raw": data_dir() / "raw",
        "processed": data_dir() / "processed",
        "splits": data_dir() / "splits",
        "synthetic": data_dir() / "synthetic",
        "manifests": data_dir() / "manifests",
        "runs": runs_dir(),
        "models": models_dir(),
        "logs": logs_dir(),
    }
    for key in ("detector", "classifier", "segmentation", "risk", "fusion", "deployment"):
        mapping[f"runs_{key}"] = runs_dir() / key
    for path in mapping.values():
        path.mkdir(parents=True, exist_ok=True)
    return mapping
