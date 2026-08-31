"""Temporal decay so old observations do not dominate current risk."""

from __future__ import annotations

from datetime import date, datetime

from cropai.dataset.schema import parse_iso


def as_date(value: str | None) -> date | None:
    if not value:
        return None
    parsed = parse_iso(value)
    if parsed is not None:
        return parsed.date()
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def age_days(timestamp: str, as_of: date) -> float | None:
    d = as_date(timestamp)
    if d is None:
        return None
    return float((as_of - d).days)


def decay_weight(age: float | None, half_life_days: float) -> float:
    if age is None:
        return 0.0
    if age < 0:
        return 0.0
    if half_life_days <= 0:
        return 1.0 if age == 0 else 0.0
    return 0.5 ** (age / float(half_life_days))
