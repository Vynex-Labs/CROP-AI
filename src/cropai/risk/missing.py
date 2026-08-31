"""Missing-data policy. MASTER_PROMPT §19: continue, mark reduced confidence."""

from __future__ import annotations

from typing import Any


WEATHER_KEYS = {
    "temperature_c",
    "humidity_pct",
    "rainfall_mm",
    "wind_ms",
    "rainfall_1d",
    "rainfall_3d",
    "rainfall_7d",
    "humidity_rolling_3d",
    "humidity_rolling_7d",
    "temperature_range_1d",
    "rainfall_frequency_7d",
}
TRAP_KEYS = {"trap_count_1d", "trap_count_3d", "trap_count_7d", "recent_pest_count_trend"}
SOIL_KEYS = {"soil_moisture"}
HISTORY_KEYS = {"days_since_previous_outbreak", "recent_positive_detections"}


def classify_missing(features: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if all(features.get(k) is None for k in WEATHER_KEYS):
        missing.append("weather")
    else:
        for k in WEATHER_KEYS:
            if features.get(k) is None:
                missing.append(k)
    if all(features.get(k) is None for k in TRAP_KEYS):
        missing.append("trap")
    else:
        for k in TRAP_KEYS:
            if features.get(k) is None:
                missing.append(k)
    if all(features.get(k) is None for k in SOIL_KEYS):
        missing.append("soil")
    if all(features.get(k) is None for k in HISTORY_KEYS):
        missing.append("history")
    if features.get("high_humidity_hours") is None:
        missing.append("high_humidity_hours")
    return missing


def _normalize_penalties(penalties: dict[str, float] | None) -> dict[str, float]:
    raw = dict(penalties or {})
    return {
        "weather": float(raw.get("weather", raw.get("weather_confidence_penalty", 0.35))),
        "trap": float(raw.get("trap", raw.get("trap_confidence_penalty", 0.25))),
        "soil": float(raw.get("soil", raw.get("soil_confidence_penalty", 0.05))),
        "history": float(raw.get("history", raw.get("history_confidence_penalty", 0.10))),
    }


def confidence_after_missing(
    missing: list[str],
    *,
    penalties: dict[str, float] | None = None,
    base: float = 0.7,
) -> tuple[float, bool]:
    """Return (confidence, reduced_flag). Never claims calibration."""
    penalties = _normalize_penalties(penalties)
    conf = base
    groups = {m for m in missing if m in penalties}
    for g in groups:
        conf -= float(penalties[g])
    # Per-field weather holes are milder than a full weather outage.
    if "weather" not in groups:
        n_partial = sum(1 for m in missing if m in WEATHER_KEYS)
        conf -= min(0.15, 0.02 * n_partial)
    conf = max(0.05, min(0.7, conf))
    reduced = bool(groups) or conf < base
    return round(conf, 4), reduced
