from cropai.storage.observations import ObservationStore
from cropai.dataset.schema import ObservationRecord
from cropai.utils.hardware import detect_hardware, format_hardware_report


def test_hardware_report_does_not_invent_gpu():
    report = detect_hardware()
    text = format_hardware_report(report)
    assert "CUDA available" in text
    if not report["cuda_available"]:
        assert report["warning"]
        assert "CUDA unavailable" in report["warning"]


def test_observation_store_roundtrip(tmp_path):
    store = ObservationStore(tmp_path / "obs.jsonl")
    rec = ObservationRecord(
        observation_id="o1",
        timestamp="2026-08-31T00:00:00+00:00",
        crop_id="cotton",
        farm_id="f1",
        field_id="p1",
        disease_id="cotton_bacterial_blight",
        severity_bin="early",
        affected_area_pct=8.0,
        confidence=0.4,
        is_synthetic=True,
        follow_up_of="",
    )
    rec2 = ObservationRecord(
        observation_id="o2",
        timestamp="2026-09-04T00:00:00+00:00",
        crop_id="cotton",
        farm_id="f1",
        field_id="p1",
        disease_id="cotton_bacterial_blight",
        severity_bin="moderate",
        affected_area_pct=14.0,
        confidence=0.5,
        is_synthetic=True,
        follow_up_of="o1",
        notes="progression",
    )
    store.append(rec)
    store.append(rec2)
    fu = store.follow_ups("o1")
    assert len(fu) == 1
    assert fu[0].affected_area_pct == 14.0
