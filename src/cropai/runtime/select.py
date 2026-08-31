"""Runtime selection: TensorRT → ORT → PyTorch → dummy. Never fake TensorRT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cropai.config.loader import load_inference_config, load_runtime_config
from cropai.utils.hardware import detect_hardware
from cropai.utils.paths import models_dir


def _try_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def describe_runtimes() -> dict[str, Any]:
    hw = detect_hardware()
    return {
        "tensorrt": _try_import("tensorrt"),
        "onnxruntime": _try_import("onnxruntime"),
        "pytorch": bool(hw.get("torch_installed")),
        "cuda": bool(hw.get("cuda_available")),
        "dummy": True,
        "nvidia_smi": bool(hw.get("nvidia_smi")),
    }


def available_weights() -> dict[str, str]:
    root = models_dir()
    found: dict[str, str] = {}
    for name in ("classifier.pt", "detector.pt", "segmenter.pt", "classifier.onnx", "detector.onnx"):
        p = root / name
        if p.exists():
            found[name] = str(p)
    return found


def select_runtime(preferred: list[str] | None = None) -> dict[str, Any]:
    """First matching runtime that can actually run. Dummy always works."""
    inf = load_inference_config()
    rt = load_runtime_config()
    order = list(preferred or (rt.get("runtimes") or inf.get("runtimes") or {}).get("fallback_order") or [])
    if not order:
        order = ["tensorrt", "onnxruntime", "pytorch", "dummy"]
    avail = describe_runtimes()
    weights = available_weights()
    has_onnx = any(k.endswith(".onnx") for k in weights)
    has_pt = any(k.endswith(".pt") for k in weights)
    chosen = "dummy"
    reason = "no_trained_weights"
    for name in order:
        if name == "tensorrt" and avail["tensorrt"] and avail["cuda"] and has_onnx:
            chosen, reason = "tensorrt", "tensorrt_cuda_onnx"
            break
        if name == "onnxruntime" and avail["onnxruntime"] and has_onnx:
            chosen, reason = "onnxruntime", "onnx_weights_present"
            break
        if name == "pytorch" and avail["pytorch"] and has_pt:
            chosen, reason = "pytorch", "pytorch_checkpoint_present"
            break
        if name == "dummy":
            chosen, reason = "dummy", "sandbox_or_untrained"
            break
    return {
        "selected": chosen,
        "reason": reason,
        "order": order,
        "available": avail,
        "weights": weights,
        "precision_policy": ((rt.get("precision") or {}).get("gpu_target") or inf.get("precision") or "fp16"),
        "int8_adopted": bool((rt.get("precision") or {}).get("int8", {}).get("adopted", False)),
    }
