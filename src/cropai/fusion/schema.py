"""Farm-level fused risk. Uncalibrated unless evaluated."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

MODEL_VERSION = "cropai-fusion-0.4.0-untrained"


@dataclass
class FusionInput:
    crop_id: str
    vision_confidence: float | None = None
    vision_referral: bool = False
    vision_untrained: bool = False
    severity: str = ""
    weather_risk: float | None = None
    trap_risk: float | None = None
    historical_risk: float | None = None
    spatial_risk: float | None = None
    missing_weather: bool = False
    missing_trap: bool = False
    is_synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FusionOutput:
    timestamp: str
    crop: str
    farm_risk: float
    components: dict[str, float | None]
    weights_used: dict[str, float]
    missing: list[str]
    reduced_confidence: bool
    conflicting_signals: bool
    expert_referral: bool
    referral_reason: str
    calibrated: bool
    backend: str
    model_version: str = MODEL_VERSION
    warnings: list[str] = field(default_factory=list)
    is_synthetic: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
