from datetime import date, timedelta

from cropai.dataset.schema import ObservationRecord
from cropai.fusion.engine import FusionEngine
from cropai.fusion.schema import FusionInput
from cropai.geo.decay import decay_weight
from cropai.geo.distance import haversine_km
from cropai.geo.hotspots import detect_hotspots


AS_OF = date(2026, 8, 31)


def _obs(oid: str, lat: float, lon: float, *, days_ago: int = 0, conf=0.9, sev="moderate", validated=True):
    d = AS_OF - timedelta(days=days_ago)
    return ObservationRecord(
        observation_id=oid,
        timestamp=f"{d.isoformat()}T12:00:00+00:00",
        crop_id="rice",
        farm_id="f",
        field_id="fld",
        lat=lat,
        lon=lon,
        disease_id="rice_blast",
        severity_bin=sev,
        confidence=conf,
        source="test",
        is_synthetic=True,
        expert_validated=validated,
    )


def test_haversine_and_decay():
    assert haversine_km(19.0, 74.0, 19.0, 74.0) == 0.0
    assert decay_weight(0, 7) == 1.0
    assert abs(decay_weight(7, 7) - 0.5) < 1e-9
    assert decay_weight(14, 7) < decay_weight(1, 7)


def test_single_case_isolated():
    report = detect_hotspots([_obs("a", 19.0, 74.0)], as_of=AS_OF)
    assert report.n_observations == 1
    assert report.hotspots[0].state == "isolated"
    assert report.hotspots[0].is_synthetic is True
    assert "unvalidated" in ",".join(report.warnings)


def test_nearby_cluster_not_isolated():
    # ~1 km offsets around 19N 74E
    recs = [
        _obs("n1", 19.000, 74.000),
        _obs("n2", 19.004, 74.000),
        _obs("n3", 19.000, 74.004),
        _obs("n4", 19.003, 74.003),
    ]
    report = detect_hotspots(recs, as_of=AS_OF)
    clustered = [h for h in report.hotspots if h.state != "isolated"]
    assert clustered, report.hotspots
    assert clustered[0].n_observations >= 3


def test_distant_cases_not_one_cluster():
    recs = [
        _obs("pune", 18.52, 73.85),
        _obs("nagpur", 21.15, 79.09),
    ]
    report = detect_hotspots(recs, as_of=AS_OF)
    assert all(h.state == "isolated" for h in report.hotspots)
    assert report.n_hotspots >= 2


def test_old_versus_recent_declining():
    recs = [
        _obs("o1", 19.0, 74.0, days_ago=20, sev="severe"),
        _obs("o2", 19.004, 74.0, days_ago=18, sev="severe"),
        _obs("o3", 19.0, 74.004, days_ago=16, sev="severe"),
        _obs("o4", 19.003, 74.003, days_ago=15, sev="moderate"),
    ]
    report = detect_hotspots(recs, as_of=AS_OF)
    clustered = [h for h in report.hotspots if h.n_observations >= 3]
    assert clustered
    assert clustered[0].state == "declining"


def test_recent_cluster_emerging_or_established():
    recs = [_obs(f"r{i}", 19.0 + i * 0.002, 74.0, days_ago=1) for i in range(4)]
    report = detect_hotspots(recs, as_of=AS_OF)
    clustered = [h for h in report.hotspots if h.n_observations >= 3]
    assert clustered
    assert clustered[0].state in {"emerging", "established"}


def test_low_confidence_downweighted_vs_validated():
    low = detect_hotspots([_obs("l", 19.0, 74.0, conf=0.2, validated=False)], as_of=AS_OF)
    high = detect_hotspots([_obs("h", 19.0, 74.0, conf=0.95, validated=True)], as_of=AS_OF)
    assert low.hotspots[0].score <= high.hotspots[0].score


def test_fusion_missing_weather_and_trap():
    out = FusionEngine().fuse(
        FusionInput(
            crop_id="rice",
            weather_risk=None,
            missing_weather=True,
            trap_risk=None,
            missing_trap=True,
            spatial_risk=0.4,
            historical_risk=0.3,
            vision_untrained=True,
            is_synthetic=True,
        )
    )
    assert "weather" in out.missing
    assert "trap" in out.missing
    assert out.reduced_confidence is True
    assert out.calibrated is False
    assert 0 <= out.farm_risk <= 1


def test_fusion_conflict_referral():
    out = FusionEngine().fuse(
        FusionInput(
            crop_id="rice",
            vision_confidence=0.95,
            weather_risk=0.1,
            trap_risk=0.1,
            historical_risk=0.1,
            spatial_risk=0.1,
        )
    )
    assert out.conflicting_signals is True
    assert out.expert_referral is True
    assert out.referral_reason == "conflicting_signals"


def test_benchmark_plan_does_not_invent_scores():
    from cropai.geo.benchmark import benchmark_plan

    plan = benchmark_plan()
    assert plan["answers"]["selected_spatial_model"] == "NOT SELECTED"
    assert plan["answers"]["is_h3_kde_dbscan_sufficient"].startswith("UNKNOWN")
    for row in plan["comparison_rows"]:
        assert row["hotspot_precision"] == "NOT MEASURED"
