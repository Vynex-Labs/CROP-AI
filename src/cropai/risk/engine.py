"""Farm-level near-term risk. Offline. Continues with missing inputs."""

from __future__ import annotations

from typing import Any

from cropai.config.loader import load_crop_config, load_risk_config
from cropai.domain.taxonomy import Taxonomy
from cropai.risk.backends import RiskBackend, load_backend
from cropai.risk.calibrate import IdentityCalibrator, load_calibrator
from cropai.risk.features import build_features
from cropai.risk.missing import classify_missing, confidence_after_missing
from cropai.risk.schema import MODEL_VERSION, RiskOutput, RiskRequest
from cropai.utils.logging import setup_logging, utc_now_iso

log = setup_logging()


def _crop_index(taxonomy: Taxonomy, crop_id: str) -> float | None:
    ids = list(taxonomy.crop_ids())
    if crop_id in ids:
        return float(ids.index(crop_id))
    return None


def _bin_label(score: float, bins: list[dict[str, Any]]) -> str:
    for b in bins:
        if float(b["min"]) <= score < float(b["max"]):
            return str(b["id"])
    return "unknown"


class RiskEngine:
    def __init__(
        self,
        *,
        backend: RiskBackend | None = None,
        taxonomy: Taxonomy | None = None,
        model_version: str = MODEL_VERSION,
        backend_kind: str = "auto",
    ) -> None:
        self.taxonomy = taxonomy or Taxonomy()
        self.crop_cfg = load_crop_config()
        self.risk_cfg = load_risk_config()
        self.heuristic_cfg = dict(self.risk_cfg.get("heuristic") or {})
        self.backend = backend or load_backend(
            kind=backend_kind, heuristic_cfg=self.heuristic_cfg
        )
        self.calibrator: IdentityCalibrator = load_calibrator(
            str((self.risk_cfg.get("calibration") or {}).get("method") or "none")
        )
        self.model_version = str(self.risk_cfg.get("model_version") or model_version)
        self.bins = list((self.risk_cfg.get("score_bins") or []))

    def forecast(self, request: RiskRequest) -> RiskOutput:
        warnings: list[str] = []
        if request.is_synthetic:
            warnings.append("inputs_are_synthetic_not_field_weather")
        crop = request.crop_id or "unknown"
        if crop not in self.taxonomy.crop_ids():
            warnings.append(f"unknown_crop:{crop}")
        stages = list(self.taxonomy.crop_stages)
        feats = build_features(
            request,
            crop_stages=stages,
            crop_index=_crop_index(self.taxonomy, crop),
        )
        missing = classify_missing(feats)
        penalties = dict((self.heuristic_cfg.get("missing") or {}))
        conf, reduced = confidence_after_missing(missing, penalties=penalties)
        try:
            raw = self.backend.predict(feats)
        except Exception as exc:
            log.exception("risk backend failure")
            warnings.append(f"backend_failure:{exc}")
            raw = {
                "disease_1": 0.5,
                "disease_3": 0.5,
                "disease_7": 0.5,
                "pest_1": 0.5,
                "pest_3": 0.5,
                "pest_7": 0.5,
            }
            reduced = True
            conf = min(conf, 0.2)
        scores = self.calibrator.apply(raw)
        kind = getattr(getattr(self.backend, "info", None), "kind", "unknown")
        trained = bool(getattr(getattr(self.backend, "info", None), "trained", False))
        if kind == "heuristic_unvalidated":
            warnings.append("heuristic_unvalidated_not_outbreak_skill")
        if not trained and kind != "heuristic_unvalidated":
            warnings.append("untrained_backend")
        warnings.append("scores_are_uncalibrated")
        if "weather" in missing:
            warnings.append("missing_weather_reduced_confidence")
        if "trap" in missing:
            warnings.append("missing_trap_reduced_confidence")
        if "soil" in missing:
            warnings.append("missing_soil_recorded")

        insufficient = "weather" in missing and "trap" in missing and "history" in missing
        refer = bool(insufficient and self.heuristic_cfg.get("insufficient_inputs_referral", True))
        reason = "insufficient_inputs" if refer else ""
        if kind == "dummy":
            refer = True
            reason = reason or "untrained_dummy"

        extras = {
            "disease_bin_1d": _bin_label(scores.get("disease_1", 0.5), self.bins),
            "pest_bin_1d": _bin_label(scores.get("pest_1", 0.5), self.bins),
            "n_weather_rows": len(request.weather),
            "n_trap_rows": len(request.traps),
            "n_observations": len(request.observations),
            "calibrated": False,
            "trained": trained,
        }
        return RiskOutput(
            timestamp=utc_now_iso(),
            crop=crop,
            disease_risk_1d=float(scores.get("disease_1", 0.5)),
            disease_risk_3d=float(scores.get("disease_3", 0.5)),
            disease_risk_7d=float(scores.get("disease_7", 0.5)),
            pest_risk_1d=float(scores.get("pest_1", 0.5)),
            pest_risk_3d=float(scores.get("pest_3", 0.5)),
            pest_risk_7d=float(scores.get("pest_7", 0.5)),
            calibrated=False,
            calibration_method=self.calibrator.method,
            reduced_confidence=reduced,
            missing_inputs=missing,
            used_features=feats,
            backend=kind,
            model_version=self.model_version,
            warnings=warnings,
            is_synthetic=bool(request.is_synthetic),
            expert_referral=refer,
            referral_reason=reason,
            confidence=conf,
            extras=extras,
        )


def forecast_risk(request: RiskRequest, backend_kind: str = "auto") -> RiskOutput:
    return RiskEngine(backend_kind=backend_kind).forecast(request)
