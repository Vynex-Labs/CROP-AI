from datetime import date, timedelta

from cropai.dataset.schema import ObservationRecord, TrapRecord, WeatherRecord
from cropai.risk.engine import RiskEngine
from cropai.risk.metrics import report, roc_auc
from cropai.risk.schema import RiskRequest
from cropai.risk.tables import generate_synthetic_weather_series, request_from_files


def _wx(day: date, *, temp=26.0, rh=90.0, rain=20.0, wind=2.0, soil=None) -> WeatherRecord:
    return WeatherRecord(
        station_id="t",
        timestamp=f"{day.isoformat()}T12:00:00+00:00",
        temperature_c=temp,
        humidity_pct=rh,
        rainfall_mm=rain,
        wind_ms=wind,
        soil_moisture=soil,
        is_synthetic=True,
        source="test",
        missing_fields="" if soil is not None else "soil_moisture",
    )


def _trap(day: date, count: int) -> TrapRecord:
    return TrapRecord(
        trap_id="t",
        timestamp=f"{day.isoformat()}T12:00:00+00:00",
        trap_type="pheromone",
        pest_id="brown_planthopper",
        count=count,
        crop_id="rice",
        is_synthetic=True,
        source="test",
    )


def test_wet_humid_disease_risk_exceeds_dry():
    as_of = date(2026, 8, 31)
    days = [as_of - timedelta(days=i) for i in range(7)]
    wet = [_wx(d, rh=92, rain=15) for d in days]
    dry = [_wx(d, rh=40, rain=0.0, temp=26) for d in days]
    eng = RiskEngine()
    high = eng.forecast(RiskRequest(crop_id="rice", timestamp=as_of.isoformat(), weather=wet, is_synthetic=True))
    low = eng.forecast(RiskRequest(crop_id="rice", timestamp=as_of.isoformat(), weather=dry, is_synthetic=True))
    assert high.disease_risk_1d > low.disease_risk_1d
    assert high.calibrated is False
    assert "heuristic_unvalidated" in high.backend
    assert any("uncalibrated" in w for w in high.warnings)


def test_high_trap_pest_risk_exceeds_zero():
    as_of = date(2026, 8, 31)
    days = [as_of - timedelta(days=i) for i in range(7)]
    weather = [_wx(d, rh=60, rain=0.0) for d in days]
    high_t = [_trap(d, 18) for d in days]
    zero_t = [_trap(d, 0) for d in days]
    eng = RiskEngine()
    high = eng.forecast(
        RiskRequest(crop_id="rice", timestamp=as_of.isoformat(), weather=weather, traps=high_t, is_synthetic=True)
    )
    low = eng.forecast(
        RiskRequest(crop_id="rice", timestamp=as_of.isoformat(), weather=weather, traps=zero_t, is_synthetic=True)
    )
    assert high.pest_risk_1d > low.pest_risk_1d


def test_missing_weather_continues_reduced_confidence():
    as_of = date(2026, 8, 31)
    traps = [_trap(as_of, 4)]
    out = RiskEngine().forecast(
        RiskRequest(crop_id="rice", timestamp=as_of.isoformat(), traps=traps, is_synthetic=True)
    )
    assert "weather" in out.missing_inputs
    assert out.reduced_confidence is True
    assert 0.0 <= out.disease_risk_1d <= 1.0
    assert out.pest_risk_1d >= 0.0


def test_missing_trap_continues_reduced_confidence():
    as_of = date(2026, 8, 31)
    weather = [_wx(as_of, rh=70, rain=2)]
    out = RiskEngine().forecast(
        RiskRequest(crop_id="rice", timestamp=as_of.isoformat(), weather=weather, is_synthetic=True)
    )
    assert "trap" in out.missing_inputs
    assert out.reduced_confidence is True
    assert out.disease_risk_7d <= out.disease_risk_1d + 1e-9


def test_insufficient_inputs_referral():
    out = RiskEngine().forecast(RiskRequest(crop_id="rice", is_synthetic=True))
    assert out.expert_referral is True
    assert out.referral_reason == "insufficient_inputs"
    assert out.reduced_confidence is True


def test_output_has_required_horizons():
    as_of = date(2026, 8, 31)
    out = RiskEngine().forecast(
        RiskRequest(crop_id="cotton", timestamp=as_of.isoformat(), weather=[_wx(as_of)], is_synthetic=True)
    )
    d = out.to_dict()
    for key in [
        "disease_risk_1d",
        "disease_risk_3d",
        "disease_risk_7d",
        "pest_risk_1d",
        "pest_risk_3d",
        "pest_risk_7d",
        "timestamp",
        "crop",
        "model_version",
        "calibrated",
    ]:
        assert key in d
    assert d["disease_risk"]["1"] == d["disease_risk_1d"]
    assert d["calibrated"] is False


def test_history_days_since_outbreak():
    as_of = date(2026, 8, 31)
    obs = ObservationRecord(
        observation_id="o1",
        timestamp=f"{(as_of - timedelta(days=2)).isoformat()}T00:00:00+00:00",
        crop_id="rice",
        disease_id="rice_blast",
        is_synthetic=True,
        source="test",
    )
    out = RiskEngine().forecast(
        RiskRequest(
            crop_id="rice",
            timestamp=as_of.isoformat(),
            weather=[_wx(as_of, rh=80, rain=5)],
            observations=[obs],
            is_synthetic=True,
        )
    )
    assert out.used_features.get("days_since_previous_outbreak") == 2.0
    assert out.used_features.get("recent_positive_detections") == 1.0


def test_synthetic_series_roundtrip(tmp_path):
    paths = generate_synthetic_weather_series(tmp_path, days=10, as_of=date(2026, 8, 31))
    req = request_from_files(
        crop_id="rice",
        weather_path=paths["weather"],
        trap_path=paths["traps"],
        obs_path=paths["observations"],
        timestamp="2026-08-31",
        is_synthetic=True,
    )
    out = RiskEngine().forecast(req)
    assert out.is_synthetic is True
    assert out.used_features.get("rainfall_7d") is not None
    assert (tmp_path / "SYNTHETIC.txt").exists()


def test_metrics_separable_and_empty():
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    assert roc_auc(y, s) == 1.0
    empty = report([], [], source="none")
    assert empty["roc_auc"] == "NOT MEASURED"
    tagged = report(y, s, source="synthetic")
    assert tagged["n"] == 4
    assert "synthetic" in tagged["source"]


def test_train_dry_run():
    from cropai.risk.train import main

    assert main(["--dry-run"]) == 0


def test_benchmark_plan_does_not_invent_scores():
    from cropai.risk.benchmark import benchmark_plan

    plan = benchmark_plan()
    assert plan["answers"]["is_lightgbm_best"].startswith("UNKNOWN")
    assert plan["answers"]["selected_risk_model"] == "NOT SELECTED"
    assert "NO" in plan["answers"]["does_data_support_ml_forecasting"]
    for row in plan["comparison_rows"]:
        assert row["roc_auc"] == "NOT MEASURED"
        assert "NOT SELECTED" in row["selected"]


def test_evaluate_without_labels_is_not_measured():
    from cropai.risk.evaluate import evaluate_available

    ev = evaluate_available()
    assert ev["roc_auc"] == "NOT MEASURED"
    assert ev["n_real_outbreak_labels"] == 0
