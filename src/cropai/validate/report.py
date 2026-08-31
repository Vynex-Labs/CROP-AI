"""Assemble Phase 6 report. Every numeric metric stays NOT MEASURED without labels."""

from __future__ import annotations

from typing import Any

from cropai.config.loader import load_yaml
from cropai.geo.benchmark import benchmark_plan as geo_plan
from cropai.risk.benchmark import benchmark_plan as risk_plan
from cropai.runtime.benchmark import benchmark_plan as runtime_plan
from cropai.runtime.select import select_runtime
from cropai.utils.hardware import detect_hardware, format_hardware_report
from cropai.utils.logging import utc_now_iso
from cropai.utils.paths import repo_root
from cropai.validate.checklist import blockers, checklist
from cropai.vision.benchmark import benchmark_plan as vision_plan

NOT_MEASURED = "NOT MEASURED"


def _nm_block(keys: list[str]) -> dict[str, str]:
    return {k: NOT_MEASURED for k in keys}


def architecture_freeze() -> dict[str, Any]:
    return load_yaml(repo_root() / "configs" / "architecture_freeze.yaml")


def final_report() -> dict[str, Any]:
    freeze = architecture_freeze()
    items = checklist()
    verified = [i for i in items if i["status"] == "VERIFIED"]
    return {
        "timestamp": utc_now_iso(),
        "phase": "6",
        "architecture_frozen": bool(freeze.get("frozen")),
        "freeze": freeze,
        "hardware": format_hardware_report(detect_hardware()),
        "runtime": select_runtime(),
        "pipeline": [
            "Field Image",
            "Detection",
            "Classification",
            "Segmentation",
            "Severity",
            "Weather / Soil / Crop Inputs",
            "Risk Forecast",
            "Spatial Analysis",
            "Risk Fusion",
            "Decision Engine",
            "Advisory / Expert Referral",
            "Structured Observation",
            "Dashboard",
        ],
        "vision_benchmark": {
            "detection": _nm_block(["mAP50", "mAP50-95", "precision", "recall"]),
            "classification": _nm_block(
                ["accuracy", "macro_f1", "precision", "recall", "per_class_recall", "confusion_matrix", "calibration"]
            ),
            "segmentation": _nm_block(["IoU", "Dice", "failure_rate"]),
            "severity": _nm_block(["MAE", "RMSE", "error_by_severity_category"]),
        },
        "forecast_benchmark": _nm_block(
            [
                "roc_auc",
                "pr_auc",
                "precision",
                "recall",
                "macro_f1",
                "brier",
                "calibration",
                "false_negative_rate",
                "forecast_lead_time",
            ]
        ),
        "spatial_benchmark": _nm_block(
            [
                "hotspot_precision",
                "hotspot_recall",
                "false_hotspot_rate",
                "missed_hotspot_rate",
                "spatial_stability",
                "temporal_stability",
            ]
        ),
        "system_benchmark": _nm_block(
            [
                "end_to_end_latency",
                "images_per_sec",
                "gpu_utilization",
                "cpu_utilization",
                "ram",
                "vram",
                "startup_time",
                "model_loading_time",
                "failure_rate",
                "recovery_time",
            ]
        ),
        "gate_harnesses": {
            "vision": vision_plan().get("status"),
            "risk": risk_plan().get("status"),
            "geo": geo_plan().get("status"),
            "runtime": runtime_plan().get("status"),
        },
        "success_checklist": items,
        "n_verified": len(verified),
        "n_checklist": len(items),
        "blockers": blockers(),
        "project_complete": False,
        "reason_incomplete": "MASTER_PROMPT §31 requires demonstrated field capability. None of the checkboxes are VERIFIED.",
        "answers": {
            "selected_models": "none — architecture not frozen",
            "false_negative_priority": "policy recorded; not measured",
            "dashboard": "NOT IMPLEMENTED",
        },
        "status": "HARNESS_READY_FINAL_BENCHMARK_NOT_RUN",
    }
