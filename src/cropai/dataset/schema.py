"""Canonical record schemas. Keep these stable — adapters convert into them."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any, Iterable


IMAGE_FIELDS = [
    "sample_id",
    "relpath",
    "crop_id",
    "class_id",
    "task",
    "split",
    "source",
    "license",
    "dataset_version",
    "is_synthetic",
    "is_field",
    "farmer_id",
    "location_id",
    "video_id",
    "capture_session_id",
    "source_group",
    "width",
    "height",
    "sha256",
    "captured_at",
    "lat",
    "lon",
    "h3_index",
    "growth_stage",
    "variety",
    "severity_bin",
    "affected_area_pct",
    "symptom_region",
    "annotation_path",
    "notes",
]


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _opt_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _opt_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


@dataclass
class ImageRecord:
    sample_id: str
    relpath: str
    crop_id: str
    class_id: str
    task: str = "classification"
    split: str = "unassigned"
    source: str = "unknown"
    license: str = "unknown"
    dataset_version: str = "v0.1.0-phase1"
    is_synthetic: bool = False
    is_field: bool = False
    farmer_id: str = ""
    location_id: str = ""
    video_id: str = ""
    capture_session_id: str = ""
    source_group: str = ""
    width: int | None = None
    height: int | None = None
    sha256: str = ""
    captured_at: str = ""
    lat: float | None = None
    lon: float | None = None
    h3_index: str = ""
    growth_stage: str = "unknown"
    variety: str = ""
    severity_bin: str = ""
    affected_area_pct: float | None = None
    symptom_region: str = "leaf"
    annotation_path: str = ""
    notes: str = ""

    def group_key(self) -> str:
        for attr in ("farmer_id", "location_id", "video_id", "capture_session_id", "source_group"):
            value = str(getattr(self, attr) or "").strip()
            if value:
                return f"{attr}:{value}"
        return f"sample:{self.sample_id}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["is_synthetic"] = bool(self.is_synthetic)
        data["is_field"] = bool(self.is_field)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageRecord":
        known = {f.name for f in fields(cls)}
        payload = {k: v for k, v in data.items() if k in known}
        payload["is_synthetic"] = _bool(payload.get("is_synthetic"))
        payload["is_field"] = _bool(payload.get("is_field"))
        payload["lat"] = _opt_float(payload.get("lat"))
        payload["lon"] = _opt_float(payload.get("lon"))
        payload["width"] = _opt_int(payload.get("width"))
        payload["height"] = _opt_int(payload.get("height"))
        payload["affected_area_pct"] = _opt_float(payload.get("affected_area_pct"))
        return cls(**payload)


@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float
    class_id: str
    confidence: float | None = None
    abs_xywh: bool = True  # True = pixel xywh; False = YOLO cxcywh normalized


@dataclass
class PolygonMask:
    class_id: str
    points: list[tuple[float, float]]  # pixel coordinates
    abs_xy: bool = True


@dataclass
class DetectionAnnotation:
    sample_id: str
    width: int
    height: int
    boxes: list[BBox] = field(default_factory=list)
    polygons: list[PolygonMask] = field(default_factory=list)
    is_synthetic: bool = False


@dataclass
class ObservationRecord:
    observation_id: str
    timestamp: str
    crop_id: str
    farm_id: str = ""
    field_id: str = ""
    farmer_id: str = ""
    lat: float | None = None
    lon: float | None = None
    h3_index: str = ""
    disease_id: str = ""
    pest_id: str = ""
    severity_bin: str = ""
    affected_area_pct: float | None = None
    confidence: float | None = None
    source: str = "human"
    is_synthetic: bool = False
    expert_validated: bool = False
    follow_up_of: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrapRecord:
    trap_id: str
    timestamp: str
    trap_type: str
    pest_id: str
    count: int
    crop_id: str = ""
    lat: float | None = None
    lon: float | None = None
    h3_index: str = ""
    is_synthetic: bool = False
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WeatherRecord:
    station_id: str
    timestamp: str
    lat: float | None = None
    lon: float | None = None
    temperature_c: float | None = None
    humidity_pct: float | None = None
    rainfall_mm: float | None = None
    wind_ms: float | None = None
    soil_moisture: float | None = None
    is_synthetic: bool = False
    source: str = "unknown"
    missing_fields: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_csv(path, records: Iterable[ImageRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [rec.to_dict() for rec in records]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=IMAGE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in IMAGE_FIELDS})
    return len(rows)


def read_csv(path) -> list[ImageRecord]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [ImageRecord.from_dict(dict(row)) for row in reader]


def write_jsonl(path, records: Iterable[Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            payload = rec.to_dict() if hasattr(rec, "to_dict") else rec
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
