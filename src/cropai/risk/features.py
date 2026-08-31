"""Temporal feature engineering. Values come from caller records, never invented weather."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from cropai.dataset.schema import ObservationRecord, TrapRecord, WeatherRecord, parse_iso
from cropai.risk.schema import RiskRequest


def _as_date(value: str | None) -> date | None:
    if not value:
        return None
    parsed = parse_iso(value)
    if parsed is not None:
        return parsed.date()
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def resolve_as_of(request: RiskRequest) -> date:
    return _as_date(request.timestamp) or date.today()


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _sum(values: list[float]) -> float | None:
    return float(sum(values)) if values else None


def _in_window(ts: str, as_of: date, days: int) -> bool:
    d = _as_date(ts)
    if d is None:
        return False
    start = as_of - timedelta(days=days - 1)
    return start <= d <= as_of


def _weather_by_day(rows: list[WeatherRecord], as_of: date, days: int) -> dict[date, list[WeatherRecord]]:
    grouped: dict[date, list[WeatherRecord]] = {}
    for row in rows:
        d = _as_date(row.timestamp)
        if d is None or not _in_window(row.timestamp, as_of, days):
            continue
        grouped.setdefault(d, []).append(row)
    return grouped


def _daily_mean(rows: list[WeatherRecord], attr: str) -> float | None:
    vals = [float(getattr(r, attr)) for r in rows if getattr(r, attr) is not None]
    return _mean(vals)


def _daily_sum(rows: list[WeatherRecord], attr: str) -> float | None:
    vals = [float(getattr(r, attr)) for r in rows if getattr(r, attr) is not None]
    return _sum(vals)


def _growth_stage_index(stage: str, stages: list[str]) -> float | None:
    if not stage or stage == "unknown":
        return None
    if stage in stages:
        return float(stages.index(stage))
    return None


def build_features(
    request: RiskRequest,
    *,
    crop_stages: list[str] | None = None,
    crop_index: float | None = None,
) -> dict[str, float | None]:
    as_of = resolve_as_of(request)
    w7 = _weather_by_day(request.weather, as_of, 7)
    w3 = {d: v for d, v in w7.items() if d >= as_of - timedelta(days=2)}
    w1 = {d: v for d, v in w7.items() if d == as_of}

    def rain(days_map: dict[date, list[WeatherRecord]]) -> float | None:
        parts = [_daily_sum(rows, "rainfall_mm") for rows in days_map.values()]
        parts_f = [p for p in parts if p is not None]
        return _sum(parts_f)

    def hum(days_map: dict[date, list[WeatherRecord]]) -> float | None:
        parts = [_daily_mean(rows, "humidity_pct") for rows in days_map.values()]
        return _mean([p for p in parts if p is not None])

    def temp(days_map: dict[date, list[WeatherRecord]]) -> float | None:
        parts = [_daily_mean(rows, "temperature_c") for rows in days_map.values()]
        return _mean([p for p in parts if p is not None])

    temps_today = []
    if as_of in w1:
        temps_today = [float(r.temperature_c) for r in w1[as_of] if r.temperature_c is not None]
    temp_range = (max(temps_today) - min(temps_today)) if len(temps_today) >= 2 else None

    rain_days = 0
    rain_known_days = 0
    for rows in w7.values():
        r = _daily_sum(rows, "rainfall_mm")
        if r is None:
            continue
        rain_known_days += 1
        if r > 0.1:
            rain_days += 1
    rain_freq = (rain_days / rain_known_days) if rain_known_days else None

    wind_vals = [
        float(r.wind_ms)
        for rows in w7.values()
        for r in rows
        if r.wind_ms is not None
    ]
    soil_vals = [
        float(r.soil_moisture)
        for rows in w7.values()
        for r in rows
        if r.soil_moisture is not None
    ]
    if request.soil_moisture is not None:
        soil_vals.append(float(request.soil_moisture))

    traps_1 = [t.count for t in request.traps if _in_window(t.timestamp, as_of, 1)]
    traps_3 = [t.count for t in request.traps if _in_window(t.timestamp, as_of, 3)]
    traps_7 = [t.count for t in request.traps if _in_window(t.timestamp, as_of, 7)]
    traps_prev = [
        t.count
        for t in request.traps
        if _in_window(t.timestamp, as_of - timedelta(days=3), 3)
        and not _in_window(t.timestamp, as_of, 3)
    ]
    trap_1 = _sum([float(c) for c in traps_1])
    trap_3 = _sum([float(c) for c in traps_3])
    trap_7 = _sum([float(c) for c in traps_7])
    trend = None
    if trap_3 is not None and traps_prev:
        trend = trap_3 - sum(traps_prev)
    elif trap_7 is not None and trap_1 is not None:
        trend = trap_1 - (trap_7 / 7.0)

    healthy_suffix = "_healthy"
    pos = [
        o
        for o in request.observations
        if _in_window(o.timestamp, as_of, 7)
        and o.disease_id
        and not str(o.disease_id).endswith(healthy_suffix)
    ]
    pest_obs = [
        o
        for o in request.observations
        if _in_window(o.timestamp, as_of, 7) and o.pest_id
    ]
    outbreak_dates = sorted(
        d
        for o in request.observations
        if o.disease_id and not str(o.disease_id).endswith(healthy_suffix)
        for d in [_as_date(o.timestamp)]
        if d is not None and d <= as_of
    )
    days_since_outbreak = (as_of - outbreak_dates[-1]).days if outbreak_dates else None

    planted = _as_date(request.planting_date)
    days_planted = (as_of - planted).days if planted else None

    month = float(as_of.month)
    stages = crop_stages or []

    recent_det = request.recent_image_detections
    if recent_det is None and pos:
        recent_det = float(len(pos))

    features: dict[str, float | None] = {
        "temperature_c": temp(w1) if w1 else temp(w7),
        "humidity_pct": hum(w1) if w1 else hum(w7),
        "rainfall_mm": rain(w1),
        "wind_ms": _mean(wind_vals),
        "soil_moisture": _mean(soil_vals),
        "rainfall_1d": rain(w1),
        "rainfall_3d": rain(w3),
        "rainfall_7d": rain(w7),
        "humidity_rolling_3d": hum(w3),
        "humidity_rolling_7d": hum(w7),
        "temperature_range_1d": temp_range,
        "high_humidity_hours": None,  # needs hourly data; do not invent
        "rainfall_frequency_7d": rain_freq,
        "crop_index": crop_index,
        "growth_stage_index": _growth_stage_index(request.growth_stage, stages),
        "days_since_planting": float(days_planted) if days_planted is not None else None,
        "days_since_previous_outbreak": float(days_since_outbreak) if days_since_outbreak is not None else None,
        "recent_positive_detections": float(recent_det) if recent_det is not None else None,
        "recent_pest_count_trend": trend,
        "trap_count_1d": trap_1,
        "trap_count_3d": trap_3,
        "trap_count_7d": trap_7,
        "nearby_positive_count": request.nearby_positive_count,
        "hotspot_score": request.hotspot_score,
        "month": month,
        "lat": request.lat,
        "lon": request.lon,
        "recent_pest_observations": float(len(pest_obs)) if pest_obs else None,
    }
    return features


def feature_vector(features: dict[str, float | None], names: list[str]) -> list[float]:
    """Impute missing numerics as 0 for ML backends; callers must also pass missing flags."""
    return [0.0 if features.get(n) is None else float(features[n]) for n in names]


def numeric_feature_names() -> list[str]:
    return [
        "temperature_c",
        "humidity_pct",
        "rainfall_mm",
        "wind_ms",
        "soil_moisture",
        "rainfall_1d",
        "rainfall_3d",
        "rainfall_7d",
        "humidity_rolling_3d",
        "humidity_rolling_7d",
        "temperature_range_1d",
        "high_humidity_hours",
        "rainfall_frequency_7d",
        "crop_index",
        "growth_stage_index",
        "days_since_planting",
        "days_since_previous_outbreak",
        "recent_positive_detections",
        "recent_pest_count_trend",
        "trap_count_1d",
        "trap_count_3d",
        "trap_count_7d",
        "nearby_positive_count",
        "hotspot_score",
        "month",
        "missing_weather",
        "missing_trap",
        "missing_soil",
        "missing_history",
    ]
