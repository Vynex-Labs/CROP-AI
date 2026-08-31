"""PyTorch → ONNX → TensorRT plan. Does not invent successful exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cropai.config.loader import load_runtime_config, load_training_config
from cropai.runtime.select import available_weights, describe_runtimes
from cropai.utils.paths import models_dir
from cropai.vision.export import export_onnx_classifier, export_ultralytics_onnx


def tensorrt_export(onnx_path: Path, engine_path: Path) -> dict[str, Any]:
    try:
        import tensorrt  # noqa: F401
    except Exception as exc:
        return {"ok": False, "reason": f"tensorrt_unavailable:{exc}"}
    if not onnx_path.exists():
        return {"ok": False, "reason": "onnx_missing"}
    return {
        "ok": False,
        "reason": "tensorrt_builder_not_wired_without_cuda_engine_job",
        "onnx": str(onnx_path),
        "engine": str(engine_path),
    }


def export_plan(*, dry_run: bool = True) -> dict[str, Any]:
    rt = load_runtime_config()
    tcfg = load_training_config()
    avail = describe_runtimes()
    weights = available_weights()
    steps = [
        {"step": "pytorch_checkpoint", "status": "PRESENT" if any(k.endswith(".pt") for k in weights) else "MISSING"},
        {"step": "onnx_export", "status": "NOT RUN" if dry_run else "PENDING"},
        {
            "step": "tensorrt_fp16",
            "status": "NOT AVAILABLE" if not (avail["tensorrt"] and avail["cuda"]) else "PENDING",
        },
        {"step": "ort_or_pytorch_fallback", "status": "CODE READY"},
    ]
    return {
        "dry_run": dry_run,
        "path": "pytorch → onnx → tensorrt(fp16) → optimized inference",
        "fallback": "onnxruntime or pytorch or dummy",
        "precision_target": (rt.get("precision") or {}).get("gpu_target", "fp16"),
        "int8_adopted": False,
        "opset": (rt.get("export") or tcfg.get("export") or {}).get("opset", 17),
        "runtimes": avail,
        "weights": weights,
        "steps": steps,
        "status": "HARNESS_READY_NO_EXPORT_WITHOUT_CHECKPOINTS",
    }


def run_export(*, dry_run: bool = True) -> dict[str, Any]:
    plan = export_plan(dry_run=dry_run)
    if dry_run:
        return plan
    models = models_dir()
    results = []
    clf = models / "classifier.pt"
    if clf.exists():
        results.append(export_onnx_classifier(clf, models / "classifier.onnx"))
    else:
        results.append({"ok": False, "reason": "classifier_checkpoint_missing"})
    det = models / "detector.pt"
    if det.exists():
        results.append(export_ultralytics_onnx(det, models / "detector.onnx"))
    else:
        results.append({"ok": False, "reason": "detector_weights_missing"})
    trt = tensorrt_export(models / "classifier.onnx", models / "classifier.fp16.engine")
    results.append(trt)
    plan["results"] = results
    plan["dry_run"] = False
    return plan
