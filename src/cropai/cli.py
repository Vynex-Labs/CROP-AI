"""Command-line entry: python -m cropai <command>."""

from __future__ import annotations

import argparse
import json
import sys

from cropai import __version__
from cropai.dataset.prepare import prepare_dataset
from cropai.domain.taxonomy import Taxonomy
from cropai.utils.hardware import detect_hardware, format_hardware_report
from cropai.utils.paths import ensure_runtime_dirs, repo_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cropai", description="CROP-AI SIH26131 toolkit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("version", help="Print package version")
    sub.add_parser("hardware", help="Detect CUDA / GPU")
    sub.add_parser("taxonomy", help="Validate crop_config.yaml")
    p_prep = sub.add_parser("prepare-dataset", help="Build manifests, splits, validation")
    p_prep.add_argument("--synthetic-only", action="store_true")
    p_prep.add_argument("--no-synthetic", action="store_true")
    p_prep.add_argument("--seed", type=int, default=42)
    p_prep.add_argument("--version", default=None)

    p_inf = sub.add_parser("infer", help="Run perception pipeline on one image")
    p_inf.add_argument("image")
    p_inf.add_argument("--crop", default="")
    p_inf.add_argument("--no-seg", action="store_true")

    sub.add_parser("benchmark-vision", help="Phase 2.5 comparison plan (no invented scores)")
    p_eval = sub.add_parser("evaluate-vision", help="Evaluate perception (dummy if untrained)")
    p_eval.add_argument("--task", default="classifier")
    p_eval.add_argument("--split", default="val")

    args = parser.parse_args(argv)
    if args.cmd == "version":
        print(__version__)
        return 0
    if args.cmd == "hardware":
        print(format_hardware_report(detect_hardware()))
        return 0
    if args.cmd == "taxonomy":
        tax = Taxonomy()
        errors = tax.validate_integrity()
        print(
            json.dumps(
                {
                    "n_crops": len(tax.crop_ids()),
                    "n_diseases": len(tax.disease_ids()),
                    "n_pests": len(tax.pest_ids()),
                    "n_healthy": len(tax.healthy_ids()),
                    "operational_crops": tax.operational_crops(),
                    "training_public_crops": tax.training_public_crops(),
                    "missing_local_field": tax.crops_missing_local_field(),
                    "errors": errors,
                },
                indent=2,
            )
        )
        return 1 if errors else 0
    if args.cmd == "prepare-dataset":
        ensure_runtime_dirs()
        summary = prepare_dataset(
            include_synthetic=not args.no_synthetic,
            synthetic_only=args.synthetic_only,
            seed=args.seed,
            dataset_version=args.version,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary.get("validation_ok") else 2
    if args.cmd == "infer":
        from cropai.vision.pipeline import PerceptionPipeline

        out = PerceptionPipeline(enable_segmentation=not args.no_seg).infer(args.image, crop_hint=args.crop)
        print(json.dumps(out.to_dict(), indent=2))
        return 0 if not out.reject_reason else 2
    if args.cmd == "benchmark-vision":
        from cropai.vision.benchmark import benchmark_plan

        print(json.dumps(benchmark_plan(), indent=2))
        return 0
    if args.cmd == "evaluate-vision":
        from cropai.vision.evaluate import main as eval_main

        return eval_main(["--task", args.task, "--split", args.split])
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
