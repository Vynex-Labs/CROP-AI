"""Simple Gaussian KDE on the sphere (haversine). Density is not a probability."""

from __future__ import annotations

import math

from cropai.geo.distance import haversine_km
from cropai.geo.schema import GeoPoint


def kde_score(lat: float, lon: float, points: list[GeoPoint], bandwidth_km: float) -> float:
    if bandwidth_km <= 0 or not points:
        return 0.0
    total = 0.0
    for p in points:
        d = haversine_km(lat, lon, p.lat, p.lon)
        total += float(p.weight) * math.exp(-0.5 * (d / bandwidth_km) ** 2)
    return total
