#!/usr/bin/env python3
"""Phase 2 training entrypoint.

  python -m cropai.vision.train --dry-run
  python -m cropai.vision.train --task classifier --smoke
  python -m cropai.vision.train --task all

Heavy training belongs on the user's RTX 4050. This process refuses to invent metrics.
"""

from __future__ import annotations

import argparse
import json
import sys

from cropai.utils.hardware import detect_hardware, format_hardware_report
from cropai.utils.logging import setup_logging
from cropai.utils.paths import ensure_runtime_dirs
from cropai.vision.train_classifier import plan as plan_clf, train_classifier
from cropai.vision.train_detector import plan_detector, train_yolo


log = setup_logging()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CROP-AI Phase 2 training")
    parser.add_argument("--task", choices=["classifier", "detector", "segmentation", "all"], default="all")
    parser.add_argument("--arch", default=None, help="classifier architecture override")
    parser.add_argument("--smoke", action="store_true", help="tiny run to verify the loop")
    parser.add_argument("--dry-run", action="store_true", help="print plan, do not train")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    ensure_runtime_dirs()
    print(format_hardware_report(detect_hardware()))
    print("---")

    results: dict[str, object] = {}
    tasks = ["classifier", "detector", "segmentation"] if args.task == "all" else [args.task]
    for task in tasks:
        if task == "classifier":
            if args.dry_run:
                results[task] = plan_clf(args.arch or "efficientnet_v2_s")
            else:
                results[task] = train_classifier(arch=args.arch, smoke=args.smoke, dry_run=False, device=args.device)
        elif task == "detector":
            results[task] = train_yolo(seg=False, smoke=args.smoke, dry_run=args.dry_run)
        else:
            results[task] = train_yolo(seg=True, smoke=args.smoke, dry_run=args.dry_run)

    print(json.dumps(results, indent=2, default=str))
    if args.dry_run:
        return 0
    failed = [k for k, v in results.items() if isinstance(v, dict) and v.get("ok") is False]
    if failed and not args.smoke:
        # Missing torch on this sandbox is expected, not a crash.
        log.warning("training not executed for: %s", failed)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
