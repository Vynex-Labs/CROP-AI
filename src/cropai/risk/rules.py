"""Unvalidated statistical heuristic. Not fitted. Not a probability."""

from __future__ import annotations

from typing import Any


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _temp_suit(temp_c: float, opt: float, span: float) -> float:
    if span <= 0:
        return 0.0
    return _clip01(1.0 - abs(temp_c - opt) / span)


def _weighted_mean(parts: list[tuple[float, float]]) -> float | None:
    usable = [(p, w) for p, w in parts if w > 0]
    if not usable:
        return None
    num = sum(p * w for p, w in usable)
    den = sum(w for _, w in usable)
    return num / den if den else None


def disease_score(features: dict[str, float | None], cfg: dict[str, Any], horizon: int) -> float:
    dcfg = dict(cfg.get("disease") or {})
    h = features.get("humidity_rolling_7d")
    if h is None:
        h = features.get("humidity_pct")
    rain = features.get("rainfall_7d")
    if rain is None:
        rain = features.get("rainfall_3d")
    if rain is None:
        rain = features.get("rainfall_1d")
    sat = float(dcfg.get("rainfall_saturate_mm", 50.0))
    parts: list[tuple[float, float]] = []
    if h is not None:
        parts.append((_clip01(float(h) / 100.0), float(dcfg.get("humidity_weight", 0.35))))
    if rain is not None:
        parts.append((_clip01(float(rain) / sat), float(dcfg.get("rainfall_7d_weight", 0.25))))
    t = features.get("temperature_c")
    if t is not None:
        parts.append(
            (
                _temp_suit(float(t), float(dcfg.get("temp_opt_c", 25.0)), float(dcfg.get("temp_range_c", 12.0))),
                float(dcfg.get("temperature_suitability_weight", 0.15)),
            )
        )
    det = features.get("recent_positive_detections")
    if det is not None:
        parts.append((_clip01(float(det) / 5.0), float(dcfg.get("recent_detections_weight", 0.15))))
    dso = features.get("days_since_previous_outbreak")
    if dso is not None:
        parts.append((_clip01(1.0 - min(float(dso), 30.0) / 30.0), float(dcfg.get("days_since_outbreak_weight", 0.10))))
    raw = _weighted_mean(parts)
    if raw is None:
        raw = 0.5
    scale = float((cfg.get("horizon_scale") or {}).get(horizon, (cfg.get("horizon_scale") or {}).get(str(horizon), 1.0)))
    return round(_clip01(raw * scale), 6)


def pest_score(features: dict[str, float | None], cfg: dict[str, Any], horizon: int) -> float:
    pcfg = dict(cfg.get("pest") or {})
    sat = float(pcfg.get("trap_saturate_count", 20.0))
    parts: list[tuple[float, float]] = []
    t7 = features.get("trap_count_7d")
    if t7 is None:
        t7 = features.get("trap_count_3d")
    if t7 is None:
        t7 = features.get("trap_count_1d")
    if t7 is not None:
        parts.append((_clip01(float(t7) / sat), float(pcfg.get("trap_7d_weight", 0.45))))
    trend = features.get("recent_pest_count_trend")
    if trend is not None:
        parts.append((_clip01(0.5 + float(trend) / (2.0 * sat)), float(pcfg.get("trap_trend_weight", 0.20))))
    t = features.get("temperature_c")
    if t is not None:
        parts.append((_temp_suit(float(t), 28.0, 14.0), float(pcfg.get("temperature_weight", 0.15))))
    pest_obs = features.get("recent_pest_observations")
    if pest_obs is not None:
        parts.append((_clip01(float(pest_obs) / 5.0), float(pcfg.get("recent_pest_obs_weight", 0.20))))
    raw = _weighted_mean(parts)
    if raw is None:
        raw = 0.5
    scale = float((cfg.get("horizon_scale") or {}).get(horizon, (cfg.get("horizon_scale") or {}).get(str(horizon), 1.0)))
    return round(_clip01(raw * scale), 6)


def score_all(features: dict[str, float | None], heuristic_cfg: dict[str, Any]) -> dict[str, float]:
    scales = heuristic_cfg.get("horizon_scale") or {1: 1.0, 3: 0.9, 7: 0.8}
    horizons = []
    for key in scales:
        try:
            horizons.append(int(key))
        except (TypeError, ValueError):
            continue
    if not horizons:
        horizons = [1, 3, 7]
    out: dict[str, float] = {}
    for h in sorted(set(horizons)):
        out[f"disease_{h}"] = disease_score(features, heuristic_cfg, h)
        out[f"pest_{h}"] = pest_score(features, heuristic_cfg, h)
    return out
