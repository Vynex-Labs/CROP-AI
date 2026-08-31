"""Risk request / output schema. MASTER_PROMPT §7 required horizons."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cropai.dataset.schema import ObservationRecord, TrapRecord, WeatherRecord
from cropai.utils.logging import utc_now_iso

MODEL_VERSION = "cropai-risk-0.3.0-untrained"

HORIZONS = (1, 3, 7)


@dataclass
class RiskRequest:
    crop_id: str
    timestamp: str = ""
    lat: float | None = None
    lon: float | None = None
    variety: str = ""
    growth_stage: str = "unknown"
    planting_date: str = ""
    irrigation: str = ""
    weather: list[WeatherRecord] = field(default_factory=list)
    traps: list[TrapRecord] = field(default_factory=list)
    observations: list[ObservationRecord] = field(default_factory=list)
    recent_image_detections: float | None = None
    soil_moisture: float | None = None
    soil_properties: dict[str, float] = field(default_factory=dict)
    h3_index: str = ""
    nearby_positive_count: float | None = None
    hotspot_score: float | None = None
    is_synthetic: bool = False
    season: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RiskOutput:
    timestamp: str
    crop: str
    disease_risk_1d: float
    disease_risk_3d: float
    disease_risk_7d: float
    pest_risk_1d: float
    pest_risk_3d: float
    pest_risk_7d: float
    calibrated: bool = False
    calibration_method: str = "none"
    reduced_confidence: bool = False
    missing_inputs: list[str] = field(default_factory=list)
    used_features: dict[str, float | None] = field(default_factory=dict)
    backend: str = "heuristic_unvalidated"
    model_version: str = MODEL_VERSION
    warnings: list[str] = field(default_factory=list)
    is_synthetic: bool = False
    expert_referral: bool = False
    referral_reason: str = ""
    confidence: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["disease_risk"] = {
            "1": self.disease_risk_1d,
            "3": self.disease_risk_3d,
            "7": self.disease_risk_7d,
        }
        data["pest_risk"] = {
            "1": self.pest_risk_1d,
            "3": self.pest_risk_3d,
            "7": self.pest_risk_7d,
        }
        return data

    @classmethod
    def empty(cls, *, crop: str, reason: str, backend: str = "none") -> "RiskOutput":
        return cls(
            timestamp=utc_now_iso(),
            crop=crop or "unknown",
            disease_risk_1d=0.5,
            disease_risk_3d=0.5,
            disease_risk_7d=0.5,
            pest_risk_1d=0.5,
            pest_risk_3d=0.5,
            pest_risk_7d=0.5,
            reduced_confidence=True,
            missing_inputs=["all"],
            backend=backend,
            warnings=[reason],
            expert_referral=True,
            referral_reason=reason,
            confidence=0.0,
        )
