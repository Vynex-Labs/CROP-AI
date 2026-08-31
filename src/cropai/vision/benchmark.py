"""Phase 2.5 comparison harness. Does not invent winners."""

from __future__ import annotations

import argparse
import json

from cropai.config.loader import load_training_config
from cropai.utils.hardware import detect_hardware, format_hardware_report
from cropai.vision.backends import describe_stack, torch_available, ultralytics_available
from cropai.vision.train_classifier import plan as plan_clf
from cropai.vision.train_detector import plan_detector


CANDIDATES = {
    "classification": {
        "primary": "efficientnet_v2_s",
        "alternatives": ["convnext_tiny", "mobilenet_v3_large", "resnet50"],
        "metrics_required": ["accuracy", "precision", "recall", "macro_f1", "latency", "memory", "model_size"],
    },
    "detection": {
        "primary": "yolo11n",
        "alternatives": ["yolo11s", "yolov8n"],
        "metrics_required": ["precision", "recall", "mAP50", "mAP50-95", "latency", "memory"],
    },
    "segmentation": {
        "primary": "yolo11n-seg",
        "alternatives": ["yolo11s-seg", "skip_segmentation_use_box_area"],
        "metrics_required": ["IoU", "Dice", "latency", "severity_MAE_with_vs_without"],
        "necessity_question": "Does segmentation improve severity enough to justify cost?",
    },
}


def benchmark_plan() -> dict:
    tcfg = load_training_config()
    hw = detect_hardware()
    rows = []
    for task, spec in CANDIDATES.items():
        for name in [spec["primary"], *spec["alternatives"]]:
            rows.append(
                {
                    "task": task,
                    "candidate": name,
                    "selected": "NOT SELECTED — pending measured results",
                    "accuracy": "NOT MEASURED",
                    "precision": "NOT MEASURED",
                    "recall": "NOT MEASURED",
                    "macro_f1": "NOT MEASURED",
                    "latency": "NOT MEASURED",
                    "memory": "NOT MEASURED",
                    "model_size": "NOT MEASURED",
                    "field_robustness": "NOT MEASURED",
                    "reason": "No trained weights / no public images in this clone.",
                }
            )
    return {
        "hardware": format_hardware_report(hw),
        "stack": describe_stack(),
        "torch": torch_available(),
        "ultralytics": ultralytics_available(),
        "training_defaults": {
            "classifier": tcfg["classifier"],
            "detector": tcfg["detector"],
            "segmentation": tcfg["segmentation"],
        },
        "plans": {
            "classifier": plan_clf(),
            "detector": plan_detector(False),
            "segmentation": plan_detector(True),
        },
        "comparison_rows": rows,
        "answers": {
            "is_efficientnetv2s_best": "UNKNOWN — not measured on this dataset/hardware",
            "is_yolo11_best": "UNKNOWN — not measured",
            "is_segmentation_necessary": "UNKNOWN — harness will compare mask vs box-proxy severity after training",
        },
        "status": "HARNESS_READY_METRICS_PENDING_USER_TRAINING",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args(argv)
    plan = benchmark_plan()
    plan["dry_run"] = args.dry_run
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
