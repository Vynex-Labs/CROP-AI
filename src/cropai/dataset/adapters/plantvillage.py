"""PlantVillage ImageFolder adapter (lab images)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cropai.dataset.adapters.base import SourceAdapter, discover_imagefolder, load_class_map
from cropai.dataset.schema import ImageRecord
from cropai.utils.hashing import sha256_file
from cropai.utils.paths import data_dir, repo_root


class PlantVillageAdapter(SourceAdapter):
    name = "plantvillage"

    def ingest(self, root: Path, **kwargs: Any) -> list[ImageRecord]:
        if not root.exists():
            return []
        mapping = load_class_map("configs/mappings/plantvillage.yaml")
        keep_unmapped = bool(kwargs.get("keep_unmapped", True))
        version = str(kwargs.get("dataset_version", "v0.1.0-phase1"))
        records: list[ImageRecord] = []
        color_root = root / "color" if (root / "color").is_dir() else root
        for folder_name, path in discover_imagefolder(color_root):
            class_id = mapping.get(folder_name)
            if class_id is None:
                if not keep_unmapped:
                    continue
                class_id = _slug(folder_name)
            crop_id = class_id[: -len("_healthy")] if class_id.endswith("_healthy") else class_id.split("_")[0]
            records.append(
                ImageRecord(
                    sample_id=f"pv_{path.stem}_{len(records):06d}",
                    relpath=_rel(path),
                    crop_id=crop_id,
                    class_id=class_id,
                    task="classification",
                    source="plantvillage",
                    license="CC0-1.0",
                    dataset_version=version,
                    is_synthetic=False,
                    is_field=False,
                    source_group=f"plantvillage:{folder_name}",
                    sha256=sha256_file(path) if path.exists() else "",
                    notes="lab_image",
                )
            )
        return records


def _slug(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
        .replace("__", "_")
    )


def _rel(path: Path) -> str:
    for base in (data_dir(), repo_root()):
        try:
            return str(path.resolve().relative_to(base.resolve()))
        except ValueError:
            continue
    return str(path)
