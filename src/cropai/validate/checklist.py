"""MASTER_PROMPT §31 success checklist. Evidence-gated — never auto-VERIFIED."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cropai.config.loader import load_yaml
from cropai.runtime.select import available_weights, describe_runtimes
from cropai.utils.hardware import detect_hardware
from cropai.utils.paths import data_dir, models_dir, repo_root


NOT_VERIFIED = "NOT VERIFIED"
IMPLEMENTED = "IMPLEMENTED_NOT_VERIFIED"


def _has_real_images() -> bool:
    raw = data_dir() / "raw"
    if not raw.exists():
        return False
    for p in raw.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"} and p.name != ".gitkeep":
            return True
    return False


def _has_field_images() -> bool:
    field = data_dir() / "raw" / "field_maharashtra"
    if not field.exists():
        return False
    return any(p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in field.rglob("*"))


def _freeze() -> dict[str, Any]:
    path = repo_root() / "configs" / "architecture_freeze.yaml"
    return load_yaml(path)


def checklist() -> list[dict[str, Any]]:
    hw = detect_hardware()
    rt = describe_runtimes()
    weights = available_weights()
    freeze = _freeze()
    items = [
        ("Offline-capable crop disease inference", "src/cropai/vision/pipeline.py", IMPLEMENTED, "dummy/untrained path only"),
        ("Pest detection", "src/cropai/vision/", IMPLEMENTED, "dummy detector; no trained pest weights"),
        ("Crop/plant recognition", "src/cropai/vision/", IMPLEMENTED, "untrained"),
        ("Disease classification", "src/cropai/vision/", IMPLEMENTED, "untrained dummy → unknown + referral"),
        ("Symptom localization", "src/cropai/vision/", IMPLEMENTED, "dummy boxes/masks"),
        ("Severity estimation", "src/cropai/vision/severity.py", IMPLEMENTED, "mask/box proxy; not lab index"),
        ("Weather-based risk forecasting", "src/cropai/risk/", IMPLEMENTED, "heuristic_unvalidated; 0 outbreak labels"),
        ("Pest-trap/sensor integration", "src/cropai/dataset/schema.py TrapRecord", IMPLEMENTED, "schema + synthetic; no CROPSAP dump"),
        ("Farm-level risk estimation", "src/cropai/fusion/", IMPLEMENTED, "uncalibrated weighted mix"),
        ("Geospatial hotspot detection", "src/cropai/geo/", IMPLEMENTED, "unvalidated; grid fallback"),
        ("Expert validation workflow", "src/cropai/advisory/", IMPLEMENTED, "rules tested; no field experts"),
        ("Structured IPM advisory", "knowledge/ipm/", IMPLEMENTED, "placeholder; chemicals blocked"),
        ("Multilingual farmer-facing output", "src/cropai/advisory/i18n.py", IMPLEMENTED, "labels only en/hi/mr"),
        ("Follow-up monitoring", "src/cropai/storage/observations.py", IMPLEMENTED, "JSONL follow_ups; no field series"),
        ("Structured field observations", "ObservationRecord", IMPLEMENTED, "0 real field rows"),
        ("Agriculture-official dashboard", "—", NOT_VERIFIED, "no UI"),
        ("Real-time/near-real-time inference where required", "src/cropai/runtime/", NOT_VERIFIED, "dummy timings omitted as FPS"),
        ("Stable long-duration execution", "src/cropai/runtime/endurance.py", NOT_VERIFIED, "5–60 min NOT RUN"),
        ("Reproducible training", "src/cropai/vision/run_meta.py", IMPLEMENTED, "metadata schema; no trained run"),
        ("CUDA-accelerated training", "train_linux.sh", NOT_VERIFIED, f"cuda_available={hw.get('cuda_available')}"),
        ("Exported deployment models", "models/", NOT_VERIFIED, f"weights={weights or 'none'}"),
        ("ONNX/TensorRT deployment path where supported", "src/cropai/runtime/export_plan.py", IMPLEMENTED, f"tensorrt={rt.get('tensorrt')} onnxruntime={rt.get('onnxruntime')}"),
        ("Final benchmark", "src/cropai/validate/", NOT_VERIFIED, "harness only; freeze=" + str(freeze.get("frozen"))),
        ("Complete documentation", "README.md AGENT.md phase*.md", IMPLEMENTED, "docs present; success not demonstrated"),
    ]
    out = []
    for name, impl, status, note in items:
        # Field images / real weights would still not auto-verify without measured metrics.
        verified = False
        out.append(
            {
                "requirement": name,
                "implementation": impl,
                "status": status if not verified else "VERIFIED",
                "evidence": note,
            }
        )
    return out


def blockers() -> list[str]:
    b = []
    if not _has_real_images():
        b.append("no_real_images_in_data_raw")
    if not _has_field_images():
        b.append("no_maharashtra_field_images")
    if not available_weights():
        b.append("no_exported_or_trained_weights")
    hw = detect_hardware()
    if not hw.get("cuda_available"):
        b.append("cuda_unavailable_in_this_environment")
    freeze = _freeze()
    if not freeze.get("frozen"):
        b.append("architecture_not_frozen")
    b.append("no_measured_vision_metrics")
    b.append("no_measured_forecast_metrics")
    b.append("no_measured_spatial_metrics")
    b.append("no_official_dashboard_ui")
    return b
