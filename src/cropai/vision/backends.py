"""Perception backends. Dummy is default when PyTorch / Ultralytics / weights are absent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from cropai.domain.taxonomy import Taxonomy
from cropai.utils.logging import setup_logging
from cropai.vision.schema import Detection, MODEL_VERSION


log = setup_logging()


@dataclass
class BackendInfo:
    name: str
    kind: str  # dummy | pytorch | ultralytics
    model_id: str
    weights_path: str = ""
    trained: bool = False
    device: str = "cpu"
    notes: str = ""


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def ultralytics_available() -> bool:
    try:
        import ultralytics  # noqa: F401

        return True
    except Exception:
        return False


class DummyDetector:
    info = BackendInfo(name="dummy_detector", kind="dummy", model_id="none", notes="untrained placeholder")

    def predict(self, image: Image.Image) -> list[Detection]:
        w, h = image.size
        margin = 0.1
        box = [w * margin, h * margin, w * (1 - 2 * margin), h * (1 - 2 * margin)]
        return [
            Detection(object_id="leaf-0", class_id="leaf", bbox_xywh=box, confidence=0.4, crop_hint=""),
        ]


class DummyClassifier:
    def __init__(self, taxonomy: Taxonomy | None = None) -> None:
        self.taxonomy = taxonomy or Taxonomy()
        self.classes = self.taxonomy.classifier_classes()
        self.info = BackendInfo(
            name="dummy_classifier",
            kind="dummy",
            model_id="none",
            notes="untrained placeholder — probabilities are NOT diagnoses",
        )

    def predict(self, image: Image.Image, crop_hint: str = "") -> dict[str, float]:
        classes = self.taxonomy.disease_ids(crop_hint) if crop_hint else self.classes
        if not classes:
            classes = self.classes or ["unknown"]
        n = len(classes)
        # Near-uniform → forces expert referral. Never a confident fake diagnosis.
        mass = 1.0 / n
        return {c: round(mass, 6) for c in classes}


class DummySegmenter:
    info = BackendInfo(name="dummy_segmenter", kind="dummy", model_id="none", notes="untrained placeholder")

    def predict(self, image: Image.Image, roi: Detection | None = None) -> list[list[list[float]]]:
        if roi is None:
            return []
        x, y, w, h = roi.bbox_xywh
        # Small diamond inside ROI so severity plumbing can be tested.
        cx, cy = x + w / 2, y + h / 2
        rw, rh = w * 0.15, h * 0.15
        return [
            [
                [cx - rw, cy],
                [cx, cy - rh],
                [cx + rw, cy],
                [cx, cy + rh],
            ]
        ]


def load_detector(weights: Path | None = None, device: str = "cpu") -> Any:
    if weights and Path(weights).exists() and ultralytics_available():
        from cropai.vision.torch_models import UltralyticsDetector

        return UltralyticsDetector(Path(weights), device=device)
    return DummyDetector()


def load_classifier(weights: Path | None = None, device: str = "cpu") -> Any:
    if weights and Path(weights).exists() and torch_available():
        from cropai.vision.torch_models import TorchClassifier

        return TorchClassifier.from_checkpoint(Path(weights), device=device)
    return DummyClassifier()


def load_segmenter(weights: Path | None = None, device: str = "cpu") -> Any:
    if weights and Path(weights).exists() and ultralytics_available():
        from cropai.vision.torch_models import UltralyticsSegmenter

        return UltralyticsSegmenter(Path(weights), device=device)
    return DummySegmenter()


def describe_stack() -> dict[str, Any]:
    return {
        "torch": torch_available(),
        "ultralytics": ultralytics_available(),
        "model_version": MODEL_VERSION,
        "detector_default": "dummy" if not ultralytics_available() else "ultralytics_if_weights",
        "classifier_default": "dummy" if not torch_available() else "pytorch_if_weights",
        "note": "No trained weights are claimed. Dummy backends emit low-confidence placeholders.",
    }
