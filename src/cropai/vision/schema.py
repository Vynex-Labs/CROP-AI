"""Synchronized vision output. Matches MASTER_PROMPT §6 required fields."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cropai.utils.logging import utc_now_iso


@dataclass
class Detection:
    object_id: str
    class_id: str
    bbox_xywh: list[float]
    confidence: float
    crop_hint: str = ""


@dataclass
class QualityReport:
    ok: bool
    score: float
    width: int = 0
    height: int = 0
    reasons: list[str] = field(default_factory=list)
    action: str = "ok"  # ok | reject | request_better_image


@dataclass
class VisionOutput:
    timestamp: str
    crop: str
    plant_or_leaf_id: str
    object_ids: list[str]
    bounding_boxes: list[list[float]]
    object_classes: list[str]
    disease_class: str
    disease_probabilities: dict[str, float]
    segmentation_masks: list[list[list[float]]]
    affected_area: float | None
    severity_estimate: str
    confidence: float
    model_version: str
    pest_classes: list[str] = field(default_factory=list)
    expert_referral: bool = False
    referral_reason: str = ""
    image_quality: str = "ok"
    reject_reason: str = ""
    severity_method: str = "none"
    backend: str = "dummy"
    warnings: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def rejected(
        cls,
        *,
        reason: str,
        model_version: str,
        quality: str = "reject",
        backend: str = "none",
    ) -> "VisionOutput":
        return cls(
            timestamp=utc_now_iso(),
            crop="unknown",
            plant_or_leaf_id="",
            object_ids=[],
            bounding_boxes=[],
            object_classes=[],
            disease_class="unknown",
            disease_probabilities={},
            segmentation_masks=[],
            affected_area=None,
            severity_estimate="unknown",
            confidence=0.0,
            model_version=model_version,
            expert_referral=True,
            referral_reason=reason,
            image_quality=quality,
            reject_reason=reason,
            backend=backend,
            warnings=[reason],
        )


MODEL_VERSION = "cropai-vision-0.2.0-untrained"
