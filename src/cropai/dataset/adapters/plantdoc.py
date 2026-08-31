"""PlantDoc adapter (in-the-wild scraped images)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cropai.dataset.adapters.base import SourceAdapter, discover_imagefolder, load_class_map
from cropai.dataset.adapters.plantvillage import _rel
from cropai.dataset.schema import ImageRecord
from cropai.utils.hashing import sha256_file


class PlantDocAdapter(SourceAdapter):
    name = "plantdoc"

    def ingest(self, root: Path, **kwargs: Any) -> list[ImageRecord]:
        if not root.exists():
            return []
        mapping = load_class_map("configs/mappings/plantdoc.yaml")
        version = str(kwargs.get("dataset_version", "v0.1.0-phase1"))
        records: list[ImageRecord] = []
        for folder_name, path in discover_imagefolder(root):
            class_id = mapping.get(folder_name, folder_name.replace(" ", "_").lower())
            crop_id = class_id.split("_")[0]
            records.append(
                ImageRecord(
                    sample_id=f"pd_{path.stem}_{len(records):06d}",
                    relpath=_rel(path),
                    crop_id=crop_id,
                    class_id=class_id,
                    task="classification",
                    source="plantdoc",
                    license="research_use_verify_before_redistribution",
                    dataset_version=version,
                    is_synthetic=False,
                    is_field=True,
                    source_group=f"plantdoc:{folder_name}",
                    sha256=sha256_file(path),
                    notes="public_field_scraped_not_maharashtra",
                )
            )
        return records
