"""Evaluation scripts. Never invent metrics — compute or mark PENDING."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cropai.dataset.schema import read_csv
from cropai.utils.paths import data_dir
from cropai.vision.metrics import classification_report, detection_prf, mae_rmse
from cropai.vision.pipeline import PerceptionPipeline
from cropai.vision.schema import MODEL_VERSION


def evaluate_classifier_split(split_csv: Path, crop_hint: str = "") -> dict:
    if not split_csv.exists():
        return {"status": "PENDING", "reason": f"missing {split_csv}"}
    rows = read_csv(split_csv)
    if not rows:
        return {"status": "PENDING", "reason": "empty_split"}
    pipe = PerceptionPipeline(enable_segmentation=False)
    y_true, y_pred = [], []
    n_refer = 0
    for rec in rows:
        img = data_dir() / rec.relpath
        if not img.exists():
            continue
        out = pipe.infer(img, crop_hint=rec.crop_id)
        if out.expert_referral:
            n_refer += 1
        y_true.append(rec.class_id)
        y_pred.append(out.disease_class)
    if not y_true:
        return {"status": "PENDING", "reason": "no_readable_images"}
    report = classification_report(y_true, y_pred)
    untrained = not getattr(getattr(pipe.classifier, "info", None), "trained", False)
    report["expert_referral_rate"] = n_refer / len(y_true)
    report["untrained_backend"] = untrained
    report["status"] = "SMOKE_DUMMY" if untrained else "COMPUTED"
    report["model_version"] = MODEL_VERSION
    report["note"] = "Dummy/untrained backends are not field performance."
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["classifier", "detector", "segmentation", "severity"], default="classifier")
    p.add_argument("--split", default="val")
    p.add_argument("--version", default=None)
    args = p.parse_args(argv)
    from cropai.config.loader import load_dataset_config

    version = args.version or str(load_dataset_config().get("version"))
    csv_path = data_dir() / "splits" / version / f"{args.split}.csv"
    if args.task == "classifier":
        result = evaluate_classifier_split(csv_path)
    else:
        result = {
            "task": args.task,
            "status": "PENDING_USER_TRAINING",
            "reason": "no trained weights; detection/seg/severity metrics require labeled field or public sets + checkpoints",
            "detection_metrics": ["precision", "recall", "mAP50", "mAP50-95"],
            "segmentation_metrics": ["IoU", "Dice", "pixel_precision", "pixel_recall"],
            "severity_metrics": ["MAE", "RMSE"],
        }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
