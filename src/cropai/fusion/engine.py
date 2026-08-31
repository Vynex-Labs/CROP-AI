"""Calibrated-deterministic weighted fusion. Not a GNN. Not fitted."""

from __future__ import annotations

from cropai.config.loader import load_geo_config
from cropai.fusion.schema import MODEL_VERSION, FusionInput, FusionOutput
from cropai.utils.logging import utc_now_iso


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class FusionEngine:
    def __init__(self) -> None:
        self.cfg = load_geo_config()
        self.fcfg = dict(self.cfg.get("fusion") or {})
        self.weights = {k: float(v) for k, v in dict(self.fcfg.get("weights") or {}).items()}
        self.conflict_spread = float(self.fcfg.get("conflict_spread", 0.50))

    def fuse(self, inp: FusionInput) -> FusionOutput:
        warnings = ["fusion_weighted_deterministic_unvalidated", "scores_are_uncalibrated"]
        if inp.is_synthetic:
            warnings.append("inputs_are_synthetic")
        missing: list[str] = []
        components: dict[str, float | None] = {
            "vision": None,
            "weather": None,
            "historical": None,
            "trap": None,
            "spatial": None,
        }

        if inp.vision_untrained or inp.vision_referral or inp.vision_confidence is None:
            missing.append("vision")
            if inp.vision_untrained:
                warnings.append("vision_untrained_excluded")
            if inp.vision_referral and inp.vision_confidence is not None:
                warnings.append("vision_referral_excluded")
        else:
            components["vision"] = _clip01(float(inp.vision_confidence))

        if inp.missing_weather or inp.weather_risk is None:
            missing.append("weather")
        else:
            components["weather"] = _clip01(float(inp.weather_risk))

        if inp.historical_risk is None:
            missing.append("historical")
        else:
            components["historical"] = _clip01(float(inp.historical_risk))

        if inp.missing_trap or inp.trap_risk is None:
            missing.append("trap")
        else:
            components["trap"] = _clip01(float(inp.trap_risk))

        if inp.spatial_risk is None:
            missing.append("spatial")
        else:
            components["spatial"] = _clip01(float(inp.spatial_risk))

        present = {k: v for k, v in components.items() if v is not None and self.weights.get(k, 0) > 0}
        used: dict[str, float] = {}
        if present:
            raw_sum = sum(self.weights.get(k, 0.0) for k in present)
            if raw_sum <= 0:
                farm = 0.5
            else:
                used = {k: self.weights[k] / raw_sum for k in present}
                farm = sum(present[k] * used[k] for k in present)
        else:
            farm = 0.5
            warnings.append("no_channels_climatology_0.5")

        vals = list(present.values())
        spread = (max(vals) - min(vals)) if len(vals) >= 2 else 0.0
        conflict = spread >= self.conflict_spread
        if conflict:
            warnings.append("conflicting_signals")

        reduced = bool(missing)
        refer = False
        reason = ""
        if len(present) == 0:
            refer = True
            reason = "insufficient_inputs"
        elif conflict:
            refer = True
            reason = "conflicting_signals"
        elif "vision" in missing and "weather" in missing and "trap" in missing:
            refer = True
            reason = "insufficient_inputs"

        return FusionOutput(
            timestamp=utc_now_iso(),
            crop=inp.crop_id,
            farm_risk=round(_clip01(farm), 6),
            components=components,
            weights_used=used,
            missing=missing,
            reduced_confidence=reduced,
            conflicting_signals=conflict,
            expert_referral=refer,
            referral_reason=reason,
            calibrated=False,
            backend=str(self.fcfg.get("candidate", "weighted_deterministic")),
            model_version=MODEL_VERSION,
            warnings=warnings,
            is_synthetic=inp.is_synthetic,
            extras={"gnn": False, "calibrated": False, "severity": inp.severity},
        )


def fuse_risk(inp: FusionInput) -> FusionOutput:
    return FusionEngine().fuse(inp)
