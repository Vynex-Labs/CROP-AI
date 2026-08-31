from cropai.advisory.engine import AdvisoryEngine
from cropai.advisory.schema import AdvisoryRequest


def test_placeholder_ipm_blocks_chemicals_and_refers():
    out = AdvisoryEngine().advise(
        AdvisoryRequest(crop_id="rice", disease_id="rice_blast", confidence=0.92, severity="early")
    )
    assert out.chemical_control_allowed is False
    assert out.categories["chemical_control"] == []
    assert out.expert_referral is True
    assert out.ipm_status == "placeholder"
    assert out.advisory_allowed in {"monitoring_only", "False", "false"}
    assert out.categories["monitoring"]


def test_low_confidence_suppresses_diagnosis():
    out = AdvisoryEngine().advise(
        AdvisoryRequest(crop_id="rice", disease_id="rice_blast", confidence=0.39, severity="early")
    )
    assert out.diagnosis_shown == "unknown"
    assert out.expert_referral is True
    assert out.chemical_control_allowed is False


def test_severe_triggers_laboratory_referral():
    out = AdvisoryEngine().advise(
        AdvisoryRequest(crop_id="rice", disease_id="rice_blast", confidence=0.9, severity="severe")
    )
    assert out.laboratory_referral is True
    assert out.expert_referral is True


def test_missing_ipm_pack_escalates():
    out = AdvisoryEngine().advise(
        AdvisoryRequest(crop_id="pearl_millet", disease_id="pearl_millet_blast", confidence=0.9)
    )
    assert out.ipm_status == "missing"
    assert out.chemical_control_allowed is False
    assert out.expert_referral is True


def test_multilingual_labels_do_not_change_actions():
    en = AdvisoryEngine().advise(
        AdvisoryRequest(crop_id="cotton", pest_id="pink_bollworm", confidence=0.7, language="en")
    )
    hi = AdvisoryEngine().advise(
        AdvisoryRequest(crop_id="cotton", pest_id="pink_bollworm", confidence=0.7, language="hi")
    )
    assert en.categories["chemical_control"] == hi.categories["chemical_control"] == []
    assert hi.labels["disease"] != en.labels["disease"]
    assert hi.language == "hi"


def test_never_emits_dosage():
    out = AdvisoryEngine().advise(
        AdvisoryRequest(crop_id="rice", disease_id="rice_blast", confidence=0.99, severity="moderate")
    )
    blob = " ".join(sum(out.categories.values(), []))
    for banned in ["ml/L", "g/ha", "kg/ha", "ppm", "dose"]:
        assert banned.lower() not in blob.lower()
