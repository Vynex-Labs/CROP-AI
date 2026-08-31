"""Generic CSV adapter for authorised field / CROPSAP dumps.

Expected columns (subset allowed):
  sample_id, relpath, crop_id, class_id, farmer_id, location_id, video_id,
  capture_session_id, lat, lon, captured_at, growth_stage, variety,
  severity_bin, affected_area_pct, license, is_field
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from cropai.dataset.adapters.base import SourceAdapter
from cropai.dataset.schema import ImageRecord


class FieldCSVAdapter(SourceAdapter):
    name = "field_csv"

    def ingest(self, root: Path, **kwargs: Any) -> list[ImageRecord]:
        csv_path = Path(kwargs["csv_path"]) if "csv_path" in kwargs else root / "images.csv"
        if not csv_path.exists():
            return []
        source = str(kwargs.get("source", "field_maharashtra"))
        license_name = str(kwargs.get("license", "project_pending_consent"))
        version = str(kwargs.get("dataset_version", "v0.1.0-phase1"))
        records: list[ImageRecord] = []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row = dict(row)
                row.setdefault("source", source)
                row.setdefault("license", license_name)
                row.setdefault("dataset_version", version)
                row.setdefault("is_field", "true")
                row.setdefault("is_synthetic", "false")
                rec = ImageRecord.from_dict(row)
                rec.is_synthetic = False
                rec.is_field = True
                records.append(rec)
        return records
