"""Geospatial records. MASTER_PROMPT §8."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

MODEL_VERSION = "cropai-geo-0.4.0-untrained"

HOTSPOT_STATES = ("isolated", "emerging", "established", "declining")


@dataclass
class GeoPoint:
    observation_id: str
    lat: float
    lon: float
    timestamp: str
    crop_id: str = ""
    farm_id: str = ""
    field_id: str = ""
    disease_id: str = ""
    pest_id: str = ""
    severity_bin: str = ""
    confidence: float | None = None
    expert_validated: bool = False
    is_synthetic: bool = False
    h3_index: str = ""
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Hotspot:
    hotspot_id: str
    state: str
    h3_index: str
    lat: float
    lon: float
    n_observations: int
    score: float
    recent_frac: float
    method: str
    observation_ids: list[str] = field(default_factory=list)
    crop_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    is_synthetic: bool = False
    backend: str = "grid_fallback"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HotspotReport:
    as_of: str
    n_observations: int
    n_hotspots: int
    hotspots: list[Hotspot]
    method: str
    h3_backend: str
    model_version: str = MODEL_VERSION
    warnings: list[str] = field(default_factory=list)
    is_synthetic: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data
