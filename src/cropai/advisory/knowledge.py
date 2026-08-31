"""Load structured IPM YAML. Never invent chemical rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cropai.config.loader import load_crop_config, load_yaml
from cropai.utils.paths import repo_root


def knowledge_dir() -> Path:
    cfg = load_crop_config()
    rel = str((cfg.get("advisory") or {}).get("knowledge_dir") or "knowledge/ipm")
    return repo_root() / rel


def load_engine_rules() -> dict[str, Any]:
    path = knowledge_dir() / "_engine_rules.yaml"
    return load_yaml(path)


def load_crop_ipm(crop_id: str) -> dict[str, Any] | None:
    path = knowledge_dir() / f"{crop_id}.yaml"
    if not path.exists():
        return None
    return load_yaml(path)


def find_entry(crop_id: str, disease_id: str = "", pest_id: str = "") -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    pack = load_crop_ipm(crop_id)
    if not pack:
        return None, None
    entries = dict(pack.get("entries") or {})
    for _key, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        if disease_id and entry.get("disease_id") == disease_id:
            return pack, entry
        if pest_id and entry.get("pest_id") == pest_id:
            return pack, entry
    return pack, None
