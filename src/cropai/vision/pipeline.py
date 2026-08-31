"""Image → detect → classify → segment → severity. Offline-capable."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cropai.config.loader import load_crop_config, load_inference_config
from cropai.domain.taxonomy import Taxonomy
from cropai.utils.logging import setup_logging, utc_now_iso
from cropai.vision.backends import load_classifier, load_detector, load_segmenter
from cropai.vision.preprocess import load_image
from cropai.vision.schema import MODEL_VERSION, Detection, VisionOutput
from cropai.vision.severity import affected_area_pct, severity_bin


log = setup_logging()


def _entropy(probs: dict[str, float]) -> float:
    import math

    values = [p for p in probs.values() if p > 0]
    if not values:
        return 0.0
    return -sum(p * math.log(p, 2) for p in values)


def _referral(probs: dict[str, float], thresholds: dict[str, Any], quality_ok: bool) -> tuple[bool, str]:
    if not quality_ok:
        return True, "image_quality"
    if not probs:
        return True, "no_probabilities"
    ranked = sorted(probs.values(), reverse=True)
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else 0.0
    high = float(thresholds.get("diagnosis_high", 0.85))
    review = float(thresholds.get("diagnosis_review", 0.60))
    if top < review:
        return True, "low_confidence"
    if top < high and (top - second) < 0.15:
        return True, "ambiguous_top2"
    if _entropy(probs) > 3.5:
        return True, "high_entropy"
    return False, ""


class PerceptionPipeline:
    def __init__(
        self,
        *,
        detector: Any | None = None,
        classifier: Any | None = None,
        segmenter: Any | None = None,
        detector_weights: Path | None = None,
        classifier_weights: Path | None = None,
        segmenter_weights: Path | None = None,
        taxonomy: Taxonomy | None = None,
        model_version: str = MODEL_VERSION,
        enable_segmentation: bool = True,
    ) -> None:
        self.taxonomy = taxonomy or Taxonomy()
        self.crop_cfg = load_crop_config()
        self.inf_cfg = load_inference_config()
        self.thresholds = dict(self.crop_cfg.get("confidence_thresholds") or {})
        self.detector = detector or load_detector(detector_weights)
        self.classifier = classifier or load_classifier(classifier_weights)
        self.segmenter = segmenter or load_segmenter(segmenter_weights)
        self.model_version = model_version
        self.enable_segmentation = enable_segmentation

    def _backend_name(self) -> str:
        parts = [
            getattr(getattr(self.detector, "info", None), "kind", "dummy"),
            getattr(getattr(self.classifier, "info", None), "kind", "dummy"),
            getattr(getattr(self.segmenter, "info", None), "kind", "dummy"),
        ]
        return "+".join(parts)

    def infer(self, image_path: str | Path, crop_hint: str = "") -> VisionOutput:
        img, quality = load_image(image_path)
        backend = self._backend_name()
        if img is None or quality.action == "reject":
            reason = ",".join(quality.reasons) or "invalid_image"
            log.warning("reject image %s (%s)", image_path, reason)
            return VisionOutput.rejected(
                reason=reason,
                model_version=self.model_version,
                quality="reject",
                backend=backend,
            )
        if quality.action == "request_better_image":
            reason = "low_image_quality:" + ",".join(quality.reasons)
            log.warning("low quality %s (%s)", image_path, reason)
            out = VisionOutput.rejected(
                reason=reason,
                model_version=self.model_version,
                quality="request_better_image",
                backend=backend,
            )
            out.expert_referral = True
            return out

        warnings: list[str] = []
        try:
            detections: list[Detection] = self.detector.predict(img)
        except Exception as exc:
            log.exception("detector failure")
            warnings.append(f"detector_failure:{exc}")
            detections = []

        leaves = [d for d in detections if d.class_id in {"leaf", "plant", "fruit"}]
        pests = [d for d in detections if d.class_id in {"pest", "trap"} or d.class_id not in {"leaf", "plant", "fruit", "lesion"}]
        roi = leaves[0] if leaves else (detections[0] if detections else None)

        try:
            probs = self.classifier.predict(img, crop_hint=crop_hint)
        except Exception as exc:
            log.exception("classifier failure")
            warnings.append(f"classifier_failure:{exc}")
            probs = {}

        masks: list[list[list[float]]] = []
        if self.enable_segmentation:
            try:
                masks = self.segmenter.predict(img, roi)
            except Exception as exc:
                log.exception("segmenter failure")
                warnings.append(f"segmenter_failure:{exc}")
                masks = []

        lesion_boxes = [d.bbox_xywh for d in detections if d.class_id == "lesion"]
        area, method = affected_area_pct(
            masks=masks,
            boxes=lesion_boxes or ([roi.bbox_xywh] if roi else []),
            image_w=quality.width,
            image_h=quality.height,
            roi_box=roi.bbox_xywh if roi else None,
        )
        if method == "box_proxy":
            warnings.append("severity_from_box_proxy_not_mask")
        sev = severity_bin(area, self.taxonomy)

        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        disease = ranked[0][0] if ranked else "unknown"
        conf = ranked[0][1] if ranked else 0.0
        crop_id = crop_hint or (disease.rsplit("_", 1)[0] if disease.endswith("_healthy") else disease.split("_")[0])
        if crop_id not in self.taxonomy.crop_ids():
            crop_id = crop_hint or "unknown"

        refer, why = _referral(probs, self.thresholds, quality.ok)
        if not getattr(getattr(self.classifier, "info", None), "trained", False):
            refer = True
            why = why or "untrained_weights"
            warnings.append("classifier_untrained_do_not_treat_as_diagnosis")
            conf = min(conf, 0.4)

        objects = detections
        return VisionOutput(
            timestamp=utc_now_iso(),
            crop=crop_id,
            plant_or_leaf_id=roi.object_id if roi else "",
            object_ids=[d.object_id for d in objects],
            bounding_boxes=[d.bbox_xywh for d in objects],
            object_classes=[d.class_id for d in objects],
            disease_class=disease if not refer or conf >= float(self.thresholds.get("diagnosis_review", 0.60)) else "unknown",
            disease_probabilities=probs,
            segmentation_masks=masks,
            affected_area=area,
            severity_estimate=sev,
            confidence=float(conf),
            model_version=self.model_version,
            pest_classes=[d.class_id for d in pests],
            expert_referral=refer,
            referral_reason=why,
            image_quality="ok",
            severity_method=method,
            backend=backend,
            warnings=warnings,
            extras={
                "quality_score": quality.score,
                "quality_reasons": quality.reasons,
                "n_detections": len(detections),
                "segmentation_enabled": self.enable_segmentation,
                "top_class": disease,
                "untrained": not getattr(getattr(self.classifier, "info", None), "trained", False),
            },
        )


def run_perception(image_path: str | Path, crop_hint: str = "") -> VisionOutput:
    return PerceptionPipeline().infer(image_path, crop_hint=crop_hint)
