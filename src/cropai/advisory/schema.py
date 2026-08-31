"""Structured advisory. Generative text must not override these fields."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

MODEL_VERSION = "cropai-advisory-0.4.0"


@dataclass
class AdvisoryRequest:
    crop_id: str
    disease_id: str = ""
    pest_id: str = ""
    confidence: float | None = None
    severity: str = ""
    farm_risk: float | None = None
    language: str = "en"
    vision_referral: bool = False
    fusion_referral: bool = False
    is_synthetic: bool = False


@dataclass
class AdvisoryOutput:
    timestamp: str
    crop: str
    disease_id: str
    pest_id: str
    diagnosis_shown: str
    risk_label: str
    confidence: float | None
    severity: str
    categories: dict[str, list[str]]
    chemical_control_allowed: bool
    advisory_allowed: str
    expert_referral: bool
    laboratory_referral: bool
    follow_up: list[str]
    language: str
    labels: dict[str, str]
    ipm_status: str
    ipm_ref: str
    source: str
    warnings: list[str] = field(default_factory=list)
    model_version: str = MODEL_VERSION
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
