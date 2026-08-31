"""Pure-Python DBSCAN on lat/lon (haversine). No sklearn required."""

from __future__ import annotations

from cropai.geo.distance import haversine_km
from cropai.geo.schema import GeoPoint


def dbscan_labels(points: list[GeoPoint], *, eps_km: float, min_samples: int) -> list[int]:
    """Return cluster ids; -1 is noise / isolated."""
    n = len(points)
    labels = [-1] * n
    visited = [False] * n
    cluster_id = 0

    def neighbors(i: int) -> list[int]:
        pi = points[i]
        return [
            j
            for j, pj in enumerate(points)
            if haversine_km(pi.lat, pi.lon, pj.lat, pj.lon) <= eps_km
        ]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nb = neighbors(i)
        if len(nb) < min_samples:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        seeds = [j for j in nb if j != i]
        k = 0
        while k < len(seeds):
            j = seeds[k]
            if not visited[j]:
                visited[j] = True
                nbj = neighbors(j)
                if len(nbj) >= min_samples:
                    for q in nbj:
                        if q not in seeds:
                            seeds.append(q)
            if labels[j] < 0:
                labels[j] = cluster_id
            k += 1
        cluster_id += 1
    return labels
