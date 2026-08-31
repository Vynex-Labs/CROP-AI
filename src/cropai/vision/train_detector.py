"""YOLO detection / segmentation training via Ultralytics when installed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cropai.config.loader import load_training_config
from cropai.utils.hardware import detect_hardware, format_hardware_report
from cropai.utils.logging import setup_logging
from cropai.utils.paths import data_dir, runs_dir
from cropai.vision.backends import ultralytics_available
from cropai.vision.run_meta import collect_run_meta, write_run_meta


log = setup_logging()


def plan_detector(seg: bool = False) -> dict[str, Any]:
    tcfg = load_training_config()
    key = "segmentation" if seg else "detector"
    hw = detect_hardware()
    return {
        "task": "segmentation" if seg else "detection",
        "candidate": tcfg[key]["candidate"],
        "alternatives": tcfg[key]["alternatives"],
        "image_size": tcfg[key]["image_size"],
        "epochs": tcfg[key]["epochs"],
        "ultralytics": ultralytics_available(),
        "cuda": hw.get("cuda_available"),
        "hardware": format_hardware_report(hw),
        "evaluate_necessity": tcfg[key].get("evaluate_necessity", False),
        "status": "NOT_TRAINED",
    }


def _data_yaml(seg: bool) -> Path:
    sub = "segment" if seg else "detect"
    return data_dir() / "synthetic" / "smoke" / "yolo" / sub / "dataset.yaml"


def train_yolo(*, seg: bool = False, smoke: bool = False, dry_run: bool = False) -> dict[str, Any]:
    summary = plan_detector(seg)
    if dry_run:
        return {**summary, "dry_run": True}
    if not ultralytics_available():
        summary["ok"] = False
        summary["reason"] = "ultralytics_not_installed"
        log.warning("Ultralytics not installed. Cannot train YOLO on this machine.")
        return summary
    from ultralytics import YOLO

    tcfg = load_training_config()
    key = "segmentation" if seg else "detector"
    name = str(tcfg[key]["candidate"])
    # yolo11n-seg vs yolo11n
    weights = name if name.endswith(".pt") else f"{name}.pt"
    hw = detect_hardware()
    if not hw.get("cuda_available"):
        log.warning("WARNING: CUDA unavailable. Continuing with CPU fallback where practical.")
    data_yaml = _data_yaml(seg)
    if not data_yaml.exists():
        return {**summary, "ok": False, "reason": f"missing_yolo_yaml:{data_yaml}"}
    epochs = 1 if smoke else int(tcfg[key]["epochs"])
    imgsz = 96 if smoke else int(tcfg[key]["image_size"])
    project = str(runs_dir() / ("segmentation" if seg else "detector"))
    model = YOLO(weights)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        project=project,
        name="smoke" if smoke else "full",
        exist_ok=True,
        pretrained=True,
        device=0 if hw.get("cuda_available") else "cpu",
        verbose=True,
    )
    run_dir = Path(project) / ("smoke" if smoke else "full")
    write_run_meta(
        run_dir,
        collect_run_meta(
            model=weights,
            smoke=smoke,
            task=summary["task"],
            results=str(results),
        ),
    )
    return {**summary, "ok": True, "run_dir": str(run_dir), "smoke": smoke}
