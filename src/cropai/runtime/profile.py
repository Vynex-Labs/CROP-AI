"""Profile the deploy session. Dummy numbers are dummy numbers."""

from __future__ import annotations

import json
from pathlib import Path

from cropai.config.loader import load_runtime_config
from cropai.runtime.schema import NOT_MEASURED, MODEL_VERSION, ProfileReport
from cropai.runtime.select import describe_runtimes, select_runtime
from cropai.runtime.session import DeploySession, rss_bytes
from cropai.utils.hardware import detect_hardware
from cropai.utils.logging import utc_now_iso
from cropai.utils.paths import runs_dir


def profile_pipeline(
    image: Path,
    *,
    crop: str = "rice",
    repeats: int = 3,
    warmup: int | None = None,
) -> ProfileReport:
    cfg = load_runtime_config()
    n_warm = int(warmup if warmup is not None else (cfg.get("warmup") or {}).get("images", 2))
    session = DeploySession()
    session.warmup(image, n=n_warm)
    totals: list[float] = []
    last_rss = rss_bytes()
    stages: dict[str, list[float]] = {}
    for _ in range(max(1, repeats)):
        out = session.run_once(image, crop=crop)
        tm = out["timings_ms"]
        totals.append(float(tm.get("total") or 0.0))
        for k, v in tm.items():
            stages.setdefault(k, []).append(float(v))
        last_rss = out.get("rss_bytes") or last_rss
    mean = {k: round(sum(vs) / len(vs), 3) for k, vs in stages.items()}
    mean_total = mean.get("total") or 0.0
    fps = round(1000.0 / mean_total, 3) if mean_total > 0 else None
    avail = describe_runtimes()
    sel = select_runtime()
    hw = detect_hardware()
    warnings = [
        "dummy_path" if sel["selected"] == "dummy" else f"runtime_{sel['selected']}",
        "do_not_report_as_yolo_fps" if sel["selected"] == "dummy" else "measure_on_target_hardware",
    ]
    if not hw.get("cuda_available"):
        warnings.append("cuda_unavailable_gpu_metrics_not_measured")
    report = ProfileReport(
        timestamp=utc_now_iso(),
        backend=str(sel["selected"]),
        precision_policy=str(sel["precision_policy"]),
        n_repeats=repeats,
        timings_ms=mean,
        fps=fps if sel["selected"] != "dummy" else None,
        rss_bytes=last_rss if isinstance(last_rss, int) else None,
        gpu_util=NOT_MEASURED,
        vram_bytes=hw.get("vram_bytes") if hw.get("cuda_available") else NOT_MEASURED,
        warmup_done=session.warmup_done,
        int8_adopted=bool(sel.get("int8_adopted")),
        tensorrt=bool(avail["tensorrt"]),
        onnxruntime=bool(avail["onnxruntime"]),
        pytorch=bool(avail["pytorch"]),
        dummy=sel["selected"] == "dummy",
        warnings=warnings,
        extras={
            "model_version": MODEL_VERSION,
            "hardware": {
                "cuda": hw.get("cuda_available"),
                "gpu": hw.get("gpu_name"),
            },
            "fps_note": "FPS omitted on dummy backend so it cannot be quoted as model throughput.",
            "yolo_latency": NOT_MEASURED,
            "classification_latency": NOT_MEASURED,
            "segmentation_latency": NOT_MEASURED,
        },
    )
    out_dir = runs_dir() / "deployment"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "last_profile.json").write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return report
