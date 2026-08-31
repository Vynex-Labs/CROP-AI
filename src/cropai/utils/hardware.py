"""Hardware / CUDA detection. Never silently claim a GPU exists."""

from __future__ import annotations

import platform
import shutil
import subprocess
from typing import Any


def _safe_torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_installed": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
        "gpu_count": 0,
        "vram_bytes": None,
        "device": "cpu",
    }
    try:
        import torch  # type: ignore
    except Exception:
        return info
    info["torch_installed"] = True
    info["torch_version"] = getattr(torch, "__version__", None)
    cuda_ok = bool(torch.cuda.is_available())
    info["cuda_available"] = cuda_ok
    info["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    if cuda_ok:
        info["gpu_count"] = int(torch.cuda.device_count())
        info["gpu_name"] = torch.cuda.get_device_name(0)
        try:
            props = torch.cuda.get_device_properties(0)
            info["vram_bytes"] = int(getattr(props, "total_memory", 0))
        except Exception:
            info["vram_bytes"] = None
        info["device"] = "cuda"
    return info


def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def detect_hardware() -> dict[str, Any]:
    torch_info = _safe_torch_info()
    report = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "nvidia_smi": nvidia_smi_available(),
        **torch_info,
        "warning": None,
    }
    if not report["cuda_available"]:
        report["warning"] = (
            "WARNING: CUDA unavailable. Continuing with CPU fallback where practical. "
            "Do not silently treat CPU training as the intended GPU path."
        )
    return report


def format_hardware_report(report: dict[str, Any] | None = None) -> str:
    report = report or detect_hardware()
    vram = report.get("vram_bytes")
    vram_str = f"{vram / (1024 ** 3):.2f} GiB" if isinstance(vram, int) and vram else "n/a"
    lines = [
        f"GPU detected: {'yes' if report.get('cuda_available') else 'no'}",
        f"GPU name: {report.get('gpu_name') or 'n/a'}",
        f"VRAM: {vram_str}",
        f"CUDA available: {report.get('cuda_available')}",
        f"PyTorch CUDA version: {report.get('cuda_version') or 'n/a'}",
        f"CUDA device: {report.get('device')}",
        f"PyTorch installed: {report.get('torch_installed')} ({report.get('torch_version')})",
        f"nvidia-smi: {report.get('nvidia_smi')}",
        f"Python: {report.get('python')}",
        f"OS: {report.get('os')}",
    ]
    if report.get("warning"):
        lines.append(str(report["warning"]))
    return "\n".join(lines)


def run_nvidia_smi() -> str | None:
    if not nvidia_smi_available():
        return None
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None
