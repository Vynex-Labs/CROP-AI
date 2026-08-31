"""PyTorch → ONNX export. TensorRT is Phase 5 and only when CUDA is present."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def export_onnx_classifier(checkpoint: Path, out_path: Path, opset: int = 17) -> dict[str, Any]:
    try:
        import torch
        from cropai.vision.torch_models import TorchClassifier
    except Exception as exc:
        return {"ok": False, "reason": f"torch_unavailable:{exc}"}
    if not checkpoint.exists():
        return {"ok": False, "reason": "checkpoint_missing"}
    clf = TorchClassifier.from_checkpoint(checkpoint)
    clf.model.eval()
    dummy = torch.randn(1, 3, 384, 384, device=clf.device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        clf.model,
        dummy,
        str(out_path),
        input_names=["image"],
        output_names=["logits"],
        opset_version=opset,
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    return {"ok": True, "path": str(out_path), "opset": opset}


def export_ultralytics_onnx(weights: Path, out_path: Path | None = None) -> dict[str, Any]:
    if not weights.exists():
        return {"ok": False, "reason": "weights_missing"}
    try:
        from ultralytics import YOLO
    except Exception as exc:
        return {"ok": False, "reason": f"ultralytics_unavailable:{exc}"}
    model = YOLO(str(weights))
    exported = model.export(format="onnx")
    return {"ok": True, "path": str(exported), "requested": str(out_path) if out_path else None}
