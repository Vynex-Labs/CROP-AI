"""Structured decision engine. No generative override of safety rules."""

from __future__ import annotations

from cropai.advisory.i18n import labels_for
from cropai.advisory.knowledge import find_entry, load_engine_rules
from cropai.advisory.rules import apply_rules
from cropai.advisory.schema import MODEL_VERSION, AdvisoryOutput, AdvisoryRequest
from cropai.domain.taxonomy import Taxonomy
from cropai.utils.logging import utc_now_iso


def _risk_label(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score < 0.33:
        return "low"
    if score < 0.66:
        return "moderate"
    return "high"


def _list(entry: dict, key: str) -> list[str]:
    val = entry.get(key) or []
    if isinstance(val, list):
        return [str(x) for x in val if str(x).strip()]
    return []


class AdvisoryEngine:
    def __init__(self, taxonomy: Taxonomy | None = None) -> None:
        self.taxonomy = taxonomy or Taxonomy()
        self.rules_doc = load_engine_rules()
        self.rules = list(self.rules_doc.get("rules") or [])

    def advise(self, req: AdvisoryRequest) -> AdvisoryOutput:
        warnings: list[str] = []
        if req.is_synthetic:
            warnings.append("synthetic_inputs")
        pack, entry = find_entry(req.crop_id, req.disease_id, req.pest_id)
        ipm_status = str((pack or {}).get("status") or "missing")
        source = str((pack or {}).get("source") or "")
        chemical_policy = str((entry or {}).get("chemical_policy") or "unavailable_escalate")
        class_known = bool(req.disease_id and req.disease_id in self.taxonomy.disease_ids()) or bool(
            req.pest_id and req.pest_id in self.taxonomy.pest_ids()
        )
        if not pack:
            warnings.append("ipm_pack_missing")
            ipm_status = "missing"
        if not entry:
            warnings.append("ipm_entry_missing")
        if ipm_status != "verified":
            warnings.append("ipm_not_verified_chemicals_blocked")

        ctx = {
            "confidence": req.confidence,
            "class_known": class_known,
            "ipm_status": ipm_status,
            "severity": req.severity,
            "chemical_policy": chemical_policy,
            "expert_referral": bool(req.vision_referral or req.fusion_referral),
        }
        decision = apply_rules(self.rules, ctx)

        categories = {
            "monitoring": _list(entry or {}, "monitoring"),
            "cultural_control": _list(entry or {}, "cultural_control"),
            "mechanical_control": _list(entry or {}, "mechanical_control"),
            "biological_control": _list(entry or {}, "biological_control"),
            "chemical_control": [],
            "safe_use_guidance": _list(entry or {}, "safe_use_guidance"),
            "follow_up_monitoring": _list(entry or {}, "follow_up_monitoring"),
            "expert_referral": [],
        }
        # Never copy chemical_control from YAML unless verified AND allowed.
        if decision["chemical_control_allowed"] and ipm_status == "verified":
            categories["chemical_control"] = _list(entry or {}, "chemical_control")
        else:
            decision["chemical_control_allowed"] = False
            if _list(entry or {}, "chemical_control"):
                warnings.append("chemical_rows_present_but_blocked_by_policy")

        if decision["advisory_allowed"] is False or decision["advisory_allowed"] == "monitoring_only":
            # Drop non-monitoring action bodies except cultural already in placeholder (they're non-chemical).
            # Cultural/mechanical from cited stubs may remain; chemicals stay empty.
            pass

        show_name = req.disease_id or req.pest_id or "unknown"
        if req.confidence is not None and float(req.confidence) < 0.60:
            show_name = "unknown"
            warnings.append("low_confidence_diagnosis_suppressed")
            decision["expert_referral"] = True

        if decision["expert_referral"]:
            categories["expert_referral"] = [
                "Consult local extension / KVKs. Do not treat this output as a confirmed diagnosis."
            ]
        if not decision["chemical_control_allowed"]:
            categories["safe_use_guidance"] = list(
                dict.fromkeys(
                    categories["safe_use_guidance"]
                    + ["Do not apply any pesticide based on this system output."]
                )
            )

        labels = labels_for(req.language)
        warnings.extend(decision.get("messages") or [])

        return AdvisoryOutput(
            timestamp=utc_now_iso(),
            crop=req.crop_id,
            disease_id=req.disease_id,
            pest_id=req.pest_id,
            diagnosis_shown=show_name,
            risk_label=_risk_label(req.farm_risk),
            confidence=req.confidence,
            severity=req.severity,
            categories=categories,
            chemical_control_allowed=bool(decision["chemical_control_allowed"]),
            advisory_allowed=str(decision["advisory_allowed"]),
            expert_referral=bool(decision["expert_referral"]),
            laboratory_referral=bool(decision["laboratory_referral"]),
            follow_up=categories["follow_up_monitoring"],
            language=req.language if req.language in {"en", "hi", "mr"} else "en",
            labels=labels,
            ipm_status=ipm_status,
            ipm_ref=str((entry or {}).get("disease_id") or (entry or {}).get("pest_id") or ""),
            source=source,
            warnings=warnings,
            model_version=MODEL_VERSION,
            extras={
                "matched_rules": decision.get("matched_rules"),
                "class_known": class_known,
                "chemical_policy": chemical_policy,
            },
        )


def advise(req: AdvisoryRequest) -> AdvisoryOutput:
    return AdvisoryEngine().advise(req)
