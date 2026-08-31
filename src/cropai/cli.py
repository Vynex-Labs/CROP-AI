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

    p_risk = sub.add_parser("forecast-risk", help="1/3/7-day disease and pest risk (uncalibrated)")
    p_risk.add_argument("--crop", required=True)
    p_risk.add_argument("--weather", default="", help="JSONL of WeatherRecord")
    p_risk.add_argument("--traps", default="", help="JSONL of TrapRecord")
    p_risk.add_argument("--observations", default="", help="JSONL of ObservationRecord")
    p_risk.add_argument("--as-of", default="", dest="as_of")
    p_risk.add_argument("--lat", type=float, default=None)
    p_risk.add_argument("--lon", type=float, default=None)
    p_risk.add_argument("--growth-stage", default="unknown")
    p_risk.add_argument("--planting-date", default="")
    p_risk.add_argument("--humidity", type=float, default=None)
    p_risk.add_argument("--temperature", type=float, default=None)
    p_risk.add_argument("--rainfall-7d", type=float, default=None, dest="rainfall_7d")
    p_risk.add_argument("--trap-7d", type=int, default=None, dest="trap_7d")
    p_risk.add_argument("--backend", default="auto")
    p_risk.add_argument("--synthetic", action="store_true")

    sub.add_parser("benchmark-risk", help="Phase 3.5 comparison plan (no invented scores)")
    sub.add_parser("evaluate-risk", help="Evaluate risk (NOT MEASURED without labels)")

    p_hs = sub.add_parser("hotspots", help="Detect hotspots from observation JSONL")
    p_hs.add_argument("--observations", required=True)
    p_hs.add_argument("--as-of", default="", dest="as_of")

    p_fuse = sub.add_parser("fuse-risk", help="Farm-level fusion (uncalibrated weighted model)")
    p_fuse.add_argument("--crop", required=True)
    p_fuse.add_argument("--vision-conf", type=float, default=None)
    p_fuse.add_argument("--vision-untrained", action="store_true")
    p_fuse.add_argument("--weather-risk", type=float, default=None)
    p_fuse.add_argument("--trap-risk", type=float, default=None)
    p_fuse.add_argument("--historical-risk", type=float, default=None)
    p_fuse.add_argument("--spatial-risk", type=float, default=None)
    p_fuse.add_argument("--synthetic", action="store_true")

    p_adv = sub.add_parser("advise", help="Structured IPM advisory (no invented chemicals)")
    p_adv.add_argument("--crop", required=True)
    p_adv.add_argument("--disease", default="")
    p_adv.add_argument("--pest", default="")
    p_adv.add_argument("--confidence", type=float, default=None)
    p_adv.add_argument("--severity", default="")
    p_adv.add_argument("--farm-risk", type=float, default=None)
    p_adv.add_argument("--lang", default="en")

    sub.add_parser("benchmark-geo", help="Phase 4.5 comparison plan (no invented scores)")

    p_prof = sub.add_parser("profile-pipeline", help="Time dummy/real e2e path (no invented GPU FPS)")
    p_prof.add_argument("image")
    p_prof.add_argument("--crop", default="rice")
    p_prof.add_argument("--repeats", type=int, default=3)

    p_exp = sub.add_parser("export-models", help="PyTorch→ONNX→TensorRT plan")
    p_exp.add_argument("--dry-run", action="store_true", default=True)
    p_exp.add_argument("--execute", action="store_true")

    p_end = sub.add_parser("endurance", help="Short endurance smoke; long durations stay NOT RUN")
    p_end.add_argument("image")
    p_end.add_argument("--seconds", type=float, default=2.0)
    p_end.add_argument("--crop", default="rice")

    sub.add_parser("benchmark-runtime", help="Phase 5.5 comparison plan (no invented scores)")

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
    if args.cmd == "forecast-risk":
        from pathlib import Path

        from cropai.dataset.schema import TrapRecord, WeatherRecord
        from cropai.risk.engine import RiskEngine
        from cropai.risk.tables import request_from_files
        from cropai.utils.logging import utc_now_iso

        ts = args.as_of or utc_now_iso()
        req = request_from_files(
            crop_id=args.crop,
            weather_path=Path(args.weather) if args.weather else None,
            trap_path=Path(args.traps) if args.traps else None,
            obs_path=Path(args.observations) if args.observations else None,
            timestamp=ts,
            lat=args.lat,
            lon=args.lon,
            growth_stage=args.growth_stage,
            planting_date=args.planting_date,
            is_synthetic=args.synthetic,
        )
        if args.humidity is not None or args.temperature is not None or args.rainfall_7d is not None:
            req.weather.append(
                WeatherRecord(
                    station_id="cli",
                    timestamp=ts,
                    lat=args.lat,
                    lon=args.lon,
                    temperature_c=args.temperature,
                    humidity_pct=args.humidity,
                    rainfall_mm=args.rainfall_7d,
                    is_synthetic=args.synthetic,
                    source="cli_scalar",
                )
            )
        if args.trap_7d is not None:
            req.traps.append(
                TrapRecord(
                    trap_id="cli",
                    timestamp=ts,
                    trap_type="unspecified",
                    pest_id="unknown",
                    count=int(args.trap_7d),
                    crop_id=args.crop,
                    is_synthetic=args.synthetic,
                    source="cli_scalar",
                )
            )
        out = RiskEngine(backend_kind=args.backend).forecast(req)
        print(json.dumps(out.to_dict(), indent=2))
        return 0
    if args.cmd == "benchmark-risk":
        from cropai.risk.benchmark import benchmark_plan

        print(json.dumps(benchmark_plan(), indent=2))
        return 0
    if args.cmd == "evaluate-risk":
        from cropai.risk.evaluate import main as eval_risk

        return eval_risk([])
    if args.cmd == "hotspots":
        from pathlib import Path

        from cropai.geo.hotspots import detect_hotspots
        from cropai.risk.tables import load_observations

        recs = load_observations(Path(args.observations))
        report = detect_hotspots(recs, as_of=args.as_of or None)
        print(json.dumps(report.to_dict(), indent=2))
        return 0
    if args.cmd == "fuse-risk":
        from cropai.fusion.engine import FusionEngine
        from cropai.fusion.schema import FusionInput

        out = FusionEngine().fuse(
            FusionInput(
                crop_id=args.crop,
                vision_confidence=args.vision_conf,
                vision_untrained=args.vision_untrained,
                weather_risk=args.weather_risk,
                trap_risk=args.trap_risk,
                historical_risk=args.historical_risk,
                spatial_risk=args.spatial_risk,
                missing_weather=args.weather_risk is None,
                missing_trap=args.trap_risk is None,
                is_synthetic=args.synthetic,
            )
        )
        print(json.dumps(out.to_dict(), indent=2))
        return 0
    if args.cmd == "advise":
        from cropai.advisory.engine import AdvisoryEngine
        from cropai.advisory.schema import AdvisoryRequest

        out = AdvisoryEngine().advise(
            AdvisoryRequest(
                crop_id=args.crop,
                disease_id=args.disease,
                pest_id=args.pest,
                confidence=args.confidence,
                severity=args.severity,
                farm_risk=args.farm_risk,
                language=args.lang,
            )
        )
        print(json.dumps(out.to_dict(), indent=2))
        return 0
    if args.cmd == "benchmark-geo":
        from cropai.geo.benchmark import benchmark_plan

        print(json.dumps(benchmark_plan(), indent=2))
        return 0
    if args.cmd == "profile-pipeline":
        from pathlib import Path

        from cropai.runtime.profile import profile_pipeline

        print(json.dumps(profile_pipeline(Path(args.image), crop=args.crop, repeats=args.repeats).to_dict(), indent=2))
        return 0
    if args.cmd == "export-models":
        from cropai.runtime.export_plan import run_export

        print(json.dumps(run_export(dry_run=not args.execute), indent=2))
        return 0
    if args.cmd == "endurance":
        from pathlib import Path

        from cropai.runtime.endurance import run_endurance

        print(json.dumps(run_endurance(Path(args.image), seconds=args.seconds, crop=args.crop), indent=2))
        return 0
    if args.cmd == "benchmark-runtime":
        from cropai.runtime.benchmark import benchmark_plan

        print(json.dumps(benchmark_plan(), indent=2))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
