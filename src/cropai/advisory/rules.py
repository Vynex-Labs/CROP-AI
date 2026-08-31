"""Deterministic eligibility. First-to-forbid chemicals wins; referrals OR."""

from __future__ import annotations

from typing import Any


def _get(ctx: dict[str, Any], key: str, default=None):
    return ctx.get(key, default)


def _match_clause(clause: dict[str, Any], ctx: dict[str, Any]) -> bool:
    if "any" in clause:
        return any(_match_clause(c, ctx) if isinstance(c, dict) else False for c in clause["any"])
    if "diagnosis_confidence_gte" in clause:
        conf = ctx.get("confidence")
        if conf is None or float(conf) < float(clause["diagnosis_confidence_gte"]):
            return False
    if "diagnosis_confidence_lt" in clause:
        conf = ctx.get("confidence")
        if conf is None or not (float(conf) < float(clause["diagnosis_confidence_lt"])):
            return False
    if "class_known" in clause:
        known = bool(ctx.get("class_known"))
        if bool(clause["class_known"]) != known:
            return False
    if "ipm_status_in" in clause:
        if str(ctx.get("ipm_status") or "") not in list(clause["ipm_status_in"]):
            return False
    if "severity_in" in clause:
        if str(ctx.get("severity") or "") not in list(clause["severity_in"]):
            return False
    if "chemical_policy" in clause:
        if str(ctx.get("chemical_policy") or "") != str(clause["chemical_policy"]):
            return False
    return True


def apply_rules(rules: list[dict[str, Any]], ctx: dict[str, Any]) -> dict[str, Any]:
    expert = bool(ctx.get("expert_referral", False))
    lab = False
    chemical = True
    advisory_allowed: str | bool = True
    messages: list[str] = []
    matched: list[str] = []
    for rule in rules:
        clause = dict(rule.get("if") or {})
        if not _match_clause(clause, ctx):
            continue
        matched.append(str(rule.get("id") or ""))
        then = dict(rule.get("then") or {})
        if then.get("expert_referral"):
            expert = True
        if then.get("laboratory_referral"):
            lab = True
        if "chemical_control_allowed" in then and not then.get("chemical_control_allowed"):
            chemical = False
        if "advisory_allowed" in then:
            val = then["advisory_allowed"]
            if val is False:
                advisory_allowed = False
            elif val == "monitoring_only" and advisory_allowed is not False:
                advisory_allowed = "monitoring_only"
        if then.get("message"):
            messages.append(str(then["message"]))
    if ctx.get("ipm_status") in {"placeholder", "missing", None, ""}:
        chemical = False
        if advisory_allowed is True:
            advisory_allowed = "monitoring_only"
        expert = True
    if ctx.get("chemical_policy") in {"unavailable_escalate", "", None}:
        chemical = False
    return {
        "expert_referral": expert,
        "laboratory_referral": lab,
        "chemical_control_allowed": chemical,
        "advisory_allowed": advisory_allowed,
        "messages": messages,
        "matched_rules": [m for m in matched if m],
    }
