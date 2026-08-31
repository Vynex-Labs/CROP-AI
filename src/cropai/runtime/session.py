"""Timed end-to-end session. Dummy backends are labeled; not YOLO FPS."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from cropai.advisory.engine import AdvisoryEngine
from cropai.advisory.schema import AdvisoryRequest
from cropai.fusion.engine import FusionEngine
from cropai.fusion.schema import FusionInput
from cropai.risk.engine import RiskEngine
from cropai.risk.schema import RiskRequest
from cropai.runtime.cache import get_cached
from cropai.runtime.select import select_runtime
from cropai.utils.logging import setup_logging
from cropai.vision.pipeline import PerceptionPipeline
from cropai.vision.preprocess import load_image

log = setup_logging()


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def rss_bytes() -> int | None:
    try:
        import resource
        import sys

        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return rss * 1024 if sys.platform != "darwin" else rss
    except Exception:
        return None


class DeploySession:
    def __init__(self) -> None:
        self.runtime = select_runtime()
        self.pipeline: PerceptionPipeline = get_cached("perception", PerceptionPipeline)
        self.risk = get_cached("risk", RiskEngine)
        self.fusion = get_cached("fusion", FusionEngine)
        self.advisory = get_cached("advisory", AdvisoryEngine)
        self.warmup_done = False

    def warmup(self, image: Path | str, n: int = 2) -> None:
        for _ in range(max(0, n)):
            try:
                self.pipeline.infer(image)
            except Exception:
                log.exception("warmup failure")
        self.warmup_done = True

    def run_once(
        self,
        image: Path | str,
        *,
        crop: str = "rice",
        language: str = "en",
    ) -> dict[str, Any]:
        timings: dict[str, float] = {}
        warnings: list[str] = ["backend_is_" + str(self.runtime.get("selected"))]
        if self.runtime.get("selected") == "dummy":
            warnings.append("dummy_timings_are_not_yolo_or_efficientnet_fps")

        t = time.perf_counter()
        _img, quality = load_image(image)
        timings["image_load"] = _ms(t)

        t = time.perf_counter()
        try:
            vision = self.pipeline.infer(image, crop_hint=crop)
        except Exception as exc:
            log.exception("vision failure")
            warnings.append(f"vision_failure:{exc}")
            from cropai.vision.schema import MODEL_VERSION, VisionOutput

            vision = VisionOutput.rejected(reason=f"model_failure:{exc}", model_version=MODEL_VERSION)
        timings["vision_total"] = _ms(t)

        t = time.perf_counter()
        risk = self.risk.forecast(RiskRequest(crop_id=crop or vision.crop, is_synthetic=True))
        timings["risk"] = _ms(t)

        t = time.perf_counter()
        fusion = self.fusion.fuse(
            FusionInput(
                crop_id=crop or vision.crop,
                vision_confidence=vision.confidence,
                vision_referral=vision.expert_referral,
                vision_untrained=bool(vision.extras.get("untrained")),
                severity=vision.severity_estimate,
                weather_risk=None,
                trap_risk=None,
                missing_weather=True,
                missing_trap=True,
                is_synthetic=True,
            )
        )
        timings["fusion"] = _ms(t)

        t = time.perf_counter()
        advisory = self.advisory.advise(
            AdvisoryRequest(
                crop_id=crop or vision.crop,
                disease_id="" if vision.disease_class == "unknown" else vision.disease_class,
                confidence=vision.confidence,
                severity=vision.severity_estimate,
                farm_risk=fusion.farm_risk,
                language=language,
                vision_referral=vision.expert_referral,
                fusion_referral=fusion.expert_referral,
                is_synthetic=True,
            )
        )
        timings["advisory"] = _ms(t)
        timings["total"] = round(
            timings["image_load"] + timings["vision_total"] + timings["risk"] + timings["fusion"] + timings["advisory"],
            3,
        )

        return {
            "vision": vision.to_dict(),
            "risk": risk.to_dict(),
            "fusion": fusion.to_dict(),
            "advisory": advisory.to_dict(),
            "timings_ms": timings,
            "image_quality": quality.action,
            "runtime": self.runtime,
            "warnings": warnings + list(vision.warnings) + list(risk.warnings),
            "rss_bytes": rss_bytes(),
            "notes": {
                "detect_ms": "bundled_in_vision_total",
                "classify_ms": "bundled_in_vision_total",
                "segment_ms": "bundled_in_vision_total",
                "yolo_latency": "NOT MEASURED — dummy path",
                "classification_latency": "NOT MEASURED — dummy path",
                "segmentation_latency": "NOT MEASURED — dummy path",
            },
        }
