"""PyTorch / Ultralytics adapters. Imported only when those packages exist."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from cropai.domain.taxonomy import Taxonomy
from cropai.vision.backends import BackendInfo
from cropai.vision.schema import Detection


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASSIFIER_REGISTRY = {
    "efficientnet_v2_s": "efficientnet_v2_s",
    "convnext_tiny": "convnext_tiny",
    "mobilenet_v3_large": "mobilenet_v3_large",
    "resnet50": "resnet50",
}


def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    if requested in {"cuda", "auto"} and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_classifier(name: str, num_classes: int, pretrained: bool = True):
    import torchvision.models as models

    if name not in CLASSIFIER_REGISTRY:
        raise KeyError(f"unknown classifier {name}. known={sorted(CLASSIFIER_REGISTRY)}")
    ctor = getattr(models, CLASSIFIER_REGISTRY[name])
    weights = "DEFAULT" if pretrained else None
    try:
        model = ctor(weights=weights)
    except TypeError:
        model = ctor(pretrained=pretrained)
    # Replace classification head.
    if hasattr(model, "classifier"):
        clf = model.classifier
        if hasattr(clf, "__getitem__"):
            last = clf[-1]
            in_f = last.in_features
            import torch.nn as nn

            model.classifier[-1] = nn.Linear(in_f, num_classes)
        elif hasattr(clf, "in_features"):
            import torch.nn as nn

            model.classifier = nn.Linear(clf.in_features, num_classes)
    elif hasattr(model, "fc"):
        import torch.nn as nn

        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise RuntimeError(f"cannot replace head for {name}")
    return model


class TorchClassifier:
    def __init__(self, model: Any, classes: list[str], device: str, model_id: str, weights_path: str = "") -> None:
        self.model = model
        self.classes = classes
        self.device = device
        self.info = BackendInfo(
            name="torch_classifier",
            kind="pytorch",
            model_id=model_id,
            weights_path=weights_path,
            trained=bool(weights_path),
            device=device,
        )

    @classmethod
    def from_checkpoint(cls, path: Path, device: str = "auto") -> "TorchClassifier":
        import torch

        device = resolve_device(device)
        blob = torch.load(path, map_location=device)
        classes = list(blob["classes"])
        name = blob.get("arch", "efficientnet_v2_s")
        model = build_classifier(name, len(classes), pretrained=False)
        model.load_state_dict(blob["state_dict"])
        model.to(device)
        model.eval()
        return cls(model, classes, device, name, str(path))

    def predict(self, image: Image.Image, crop_hint: str = "") -> dict[str, float]:
        import torch
        from torchvision.transforms import functional as F

        x = F.to_tensor(image.resize((384, 384)))
        x = F.normalize(x, IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.softmax(logits, dim=1)[0].tolist()
        return {c: float(p) for c, p in zip(self.classes, prob, strict=False)}


class UltralyticsDetector:
    def __init__(self, weights: Path, device: str = "cpu") -> None:
        from ultralytics import YOLO

        self.device = resolve_device(device)
        self.model = YOLO(str(weights))
        self.info = BackendInfo(
            name="ultralytics_detector",
            kind="ultralytics",
            model_id=weights.stem,
            weights_path=str(weights),
            trained=True,
            device=self.device,
        )

    def predict(self, image: Image.Image) -> list[Detection]:
        results = self.model.predict(image, device=self.device, verbose=False)
        out: list[Detection] = []
        if not results:
            return out
        res = results[0]
        names = res.names if hasattr(res, "names") else {}
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            return out
        xywh = boxes.xywh.cpu().tolist() if hasattr(boxes.xywh, "cpu") else []
        cls = boxes.cls.cpu().tolist() if hasattr(boxes.cls, "cpu") else []
        conf = boxes.conf.cpu().tolist() if hasattr(boxes.conf, "cpu") else []
        for i, box in enumerate(xywh):
            cid = names.get(int(cls[i]), str(int(cls[i]))) if cls else "object"
            cf = float(conf[i]) if i < len(conf) else 0.0
            out.append(
                Detection(
                    object_id=f"{cid}-{i}",
                    class_id=str(cid),
                    bbox_xywh=[float(x) for x in box[:4]],
                    confidence=cf,
                )
            )
        return out


class UltralyticsSegmenter:
    def __init__(self, weights: Path, device: str = "cpu") -> None:
        from ultralytics import YOLO

        self.device = resolve_device(device)
        self.model = YOLO(str(weights))
        self.info = BackendInfo(
            name="ultralytics_segmenter",
            kind="ultralytics",
            model_id=weights.stem,
            weights_path=str(weights),
            trained=True,
            device=self.device,
        )

    def predict(self, image: Image.Image, roi: Detection | None = None) -> list[list[list[float]]]:
        results = self.model.predict(image, device=self.device, verbose=False)
        polys: list[list[list[float]]] = []
        if not results:
            return polys
        res = results[0]
        if getattr(res, "masks", None) is None or res.masks is None:
            return polys
        xy = res.masks.xy if hasattr(res.masks, "xy") else []
        for poly in xy:
            polys.append([[float(x), float(y)] for x, y in poly])
        return polys
