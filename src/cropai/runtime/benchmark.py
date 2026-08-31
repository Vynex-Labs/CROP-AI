"""Phase 5.5 harness. Unrun long tests and GPU metrics stay NOT MEASURED."""

from __future__ import annotations

import json

from cropai.runtime.export_plan import export_plan
from cropai.runtime.schema import NOT_MEASURED
from cropai.runtime.select import describe_runtimes, select_runtime
from cropai.utils.hardware import detect_hardware, format_hardware_report


def benchmark_plan() -> dict:
    sel = select_runtime()
    avail = describe_runtimes()
    hw = detect_hardware()
    metrics = [
        "image_loading_latency",
        "preprocessing_latency",
        "yolo_latency",
        "classification_latency",
        "segmentation_latency",
        "risk_model_latency",
        "spatial_processing_latency",
        "fusion_latency",
        "total_inference_latency",
        "fps",
        "cpu_utilization",
        "gpu_utilization",
        "ram",
        "vram",
        "startup_time",
        "model_loading_time",
        "thermal",
        "long_run_stability",
    ]
    rows = []
    for prec in ("fp32", "fp16", "int8"):
        row = {"precision": prec, "selected": "NOT SELECTED — pending measured results on target GPU"}
        for m in metrics:
            row[m] = NOT_MEASURED
        if prec == "fp16":
            row["selected"] = "POLICY TARGET for GPU — not measured here"
        if prec == "int8":
            row["selected"] = "NOT ADOPTED — needs calibration + measured speedup"
        rows.append(row)
    return {
        "hardware": format_hardware_report(hw),
        "runtimes": avail,
        "selected_runtime": sel,
        "export": export_plan(dry_run=True),
        "comparison_rows": rows,
        "long_run": {k: "NOT RUN" for k in ("5min", "15min", "30min", "60min")},
        "answers": {
            "measured_performance": "Dummy-path smoke only. YOLO/TRT/ORT: NOT MEASURED.",
            "bottleneck": "UNKNOWN — no trained GPU pipeline in this clone",
            "optimization_applied": "none (do not optimize without a real profile)",
            "remaining_issues": [
                "no checkpoints",
                "no CUDA",
                "no TensorRT",
                "INT8 not adopted",
                "1-hour endurance not run",
            ],
            "int8_adopted": False,
            "fp16_measured": False,
        },
        "status": "HARNESS_READY_METRICS_PENDING_USER_GPU",
    }


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(benchmark_plan(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
