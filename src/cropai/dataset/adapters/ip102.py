"""IP102 pest dataset adapter (classification folders 0-101 or named)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cropai.dataset.adapters.base import SourceAdapter, discover_imagefolder
from cropai.dataset.adapters.plantvillage import _rel
from cropai.dataset.schema import ImageRecord
from cropai.utils.hashing import sha256_file


class IP102Adapter(SourceAdapter):
    name = "ip102"

    def ingest(self, root: Path, **kwargs: Any) -> list[ImageRecord]:
        if not root.exists():
            return []
        version = str(kwargs.get("dataset_version", "v0.1.0-phase1"))
        records: list[ImageRecord] = []
        for folder_name, path in discover_imagefolder(root):
            class_id = f"ip102_{folder_name}"
            records.append(
                ImageRecord(
                    sample_id=f"ip102_{path.stem}_{len(records):06d}",
                    relpath=_rel(path),
                    crop_id="unknown",
                    class_id=class_id,
                    task="pest_classification",
                    source="ip102",
                    license="research_use_verify_before_redistribution",
                    dataset_version=version,
                    is_synthetic=False,
                    is_field=True,
                    source_group=f"ip102:{folder_name}",
                    sha256=sha256_file(path),
                    notes="pest_pretraining_not_trap_counts",
                )
            )
        return records
