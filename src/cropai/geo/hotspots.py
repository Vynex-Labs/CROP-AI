"""Hotspot detection: H3 aggregation + time decay + DBSCAN + KDE."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any, Iterable

from cropai.config.loader import load_crop_config, load_geo_config
from cropai.dataset.schema import ObservationRecord
from cropai.geo.dbscan import dbscan_labels
from cropai.geo.decay import age_days, as_date, decay_weight
from cropai.geo.h3index import cell_index, describe_h3
from cropai.geo.kde import kde_score
from cropai.geo.schema import MODEL_VERSION, GeoPoint, Hotspot, HotspotReport
from cropai.utils.logging import utc_now_iso


def _severity_weight(bin_id: str, table: dict[str, float]) -> float:
    if bin_id in table:
        return float(table[bin_id])
    return float(table.get("unknown", 0.3))


def observations_to_points(
    records: Iterable[ObservationRecord],
    *,
    as_of: date,
    cfg: dict[str, Any],
    resolution: int,
) -> tuple[list[GeoPoint], str]:
    half = float((cfg.get("decay") or {}).get("half_life_days", 7))
    sev_table = dict(cfg.get("severity_weight") or {})
    low_c = float(cfg.get("low_confidence_below", 0.60))
    unval = float(cfg.get("unvalidated_observation_factor", 0.50))
    points: list[GeoPoint] = []
    backend = "grid_fallback"
    for rec in records:
        if rec.lat is None or rec.lon is None:
            continue
        idx, backend = cell_index(float(rec.lat), float(rec.lon), resolution)
        age = age_days(rec.timestamp, as_of)
        w = decay_weight(age, half)
        w *= _severity_weight(rec.severity_bin or "", sev_table)
        conf = rec.confidence
        if conf is None:
            w *= unval
        else:
            w *= max(0.05, min(1.0, float(conf)))
            if float(conf) < low_c:
                w *= unval
        if not rec.expert_validated:
            w *= unval
        points.append(
            GeoPoint(
                observation_id=rec.observation_id,
                lat=float(rec.lat),
                lon=float(rec.lon),
                timestamp=rec.timestamp,
                crop_id=rec.crop_id,
                farm_id=rec.farm_id,
                field_id=rec.field_id,
                disease_id=rec.disease_id,
                pest_id=rec.pest_id,
                severity_bin=rec.severity_bin,
                confidence=rec.confidence,
                expert_validated=rec.expert_validated,
                is_synthetic=rec.is_synthetic,
                h3_index=idx,
                weight=w,
            )
        )
    return points, backend


def _classify(n: int, recent_frac: float, cfg: dict[str, Any]) -> str:
    c = dict(cfg.get("classification") or {})
    if n <= int(c.get("isolated_max_points", 1)):
        return "isolated"
    emerging = float(c.get("emerging_recent_frac", 0.70))
    declining = float(c.get("declining_recent_frac", 0.30))
    min_est = int(c.get("established_min_points", 3))
    if recent_frac >= emerging:
        return "emerging"
    if recent_frac <= declining:
        return "declining"
    if n >= min_est:
        return "established"
    return "emerging"


def detect_hotspots(
    records: list[ObservationRecord],
    *,
    as_of: str | date | None = None,
    resolution: int | None = None,
) -> HotspotReport:
    cfg = load_geo_config()
    crop_cfg = load_crop_config()
    geo_cfg = dict(crop_cfg.get("geospatial") or {})
    res = int(resolution or cfg.get("h3", {}).get("resolution_cluster") or geo_cfg.get("h3_resolution_cluster") or 8)
    as_of_d = as_date(as_of) if isinstance(as_of, str) else as_of
    as_of_d = as_of_d or date.today()
    recent_window = int((cfg.get("classification") or {}).get("recent_window_days", 7))
    dbscan_cfg = dict(cfg.get("dbscan") or {})
    kde_bw = float((cfg.get("kde") or {}).get("bandwidth_km", 8.0))

    points, backend = observations_to_points(records, as_of=as_of_d, cfg=cfg, resolution=res)
    warnings = []
    if backend == "grid_fallback":
        warnings.append("h3_package_missing_using_grid_fallback")
    if any(p.is_synthetic for p in points):
        warnings.append("inputs_are_synthetic_not_field_outbreaks")
    warnings.append("hotspot_scores_unvalidated")

    labels = dbscan_labels(
        points,
        eps_km=float(dbscan_cfg.get("eps_km", 5.0)),
        min_samples=int(dbscan_cfg.get("min_samples", 3)),
    )

    groups: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[lab].append(i)

    hotspots: list[Hotspot] = []
    hid = 0
    for lab, idxs in sorted(groups.items(), key=lambda kv: kv[0]):
        cluster_pts = [points[i] for i in idxs]
        n = len(cluster_pts)
        recent = [
            p
            for p in cluster_pts
            if (age_days(p.timestamp, as_of_d) is not None and age_days(p.timestamp, as_of_d) <= recent_window)
        ]
        recent_frac = (len(recent) / n) if n else 0.0
        lat = sum(p.lat for p in cluster_pts) / n
        lon = sum(p.lon for p in cluster_pts) / n
        idx, _ = cell_index(lat, lon, res)
        score = kde_score(lat, lon, points, kde_bw)
        # Isolated DBSCAN noise still becomes an isolated observation hotspot.
        state = "isolated" if lab < 0 and n == 1 else _classify(n, recent_frac, cfg)
        if lab < 0 and n > 1:
            # Noise bucket of unrelated points: emit each as isolated.
            for p in cluster_pts:
                hid += 1
                p_score = kde_score(p.lat, p.lon, points, kde_bw)
                hotspots.append(
                    Hotspot(
                        hotspot_id=f"hs_{hid:03d}",
                        state="isolated",
                        h3_index=p.h3_index,
                        lat=p.lat,
                        lon=p.lon,
                        n_observations=1,
                        score=round(p_score, 6),
                        recent_frac=1.0 if (age_days(p.timestamp, as_of_d) or 99) <= recent_window else 0.0,
                        method="dbscan+kde+decay",
                        observation_ids=[p.observation_id],
                        crop_ids=[p.crop_id] if p.crop_id else [],
                        warnings=list(warnings),
                        is_synthetic=p.is_synthetic,
                        backend=backend,
                    )
                )
            continue
        hid += 1
        hotspots.append(
            Hotspot(
                hotspot_id=f"hs_{hid:03d}",
                state=state,
                h3_index=idx,
                lat=round(lat, 6),
                lon=round(lon, 6),
                n_observations=n,
                score=round(score, 6),
                recent_frac=round(recent_frac, 4),
                method="dbscan+kde+decay",
                observation_ids=[p.observation_id for p in cluster_pts],
                crop_ids=sorted({p.crop_id for p in cluster_pts if p.crop_id}),
                warnings=list(warnings),
                is_synthetic=any(p.is_synthetic for p in cluster_pts),
                backend=backend,
            )
        )

    return HotspotReport(
        as_of=as_of_d.isoformat(),
        n_observations=len(points),
        n_hotspots=len(hotspots),
        hotspots=hotspots,
        method="h3_or_grid + dbscan + kde + time_decay",
        h3_backend=backend,
        model_version=MODEL_VERSION,
        warnings=warnings,
        is_synthetic=any(p.is_synthetic for p in points),
        extras={
            "h3": describe_h3(),
            "resolution": res,
            "n_dbscan_clusters": len([k for k in groups if k >= 0]),
            "n_noise": len(groups.get(-1, [])),
            "calibrated": False,
        },
    )


def nearby_positive_count(points: list[GeoPoint], lat: float, lon: float, radius_km: float = 5.0) -> int:
    from cropai.geo.distance import haversine_km

    return sum(1 for p in points if haversine_km(lat, lon, p.lat, p.lon) <= radius_km)


def cell_hotspot_score(points: list[GeoPoint], h3_index: str) -> float:
    return float(sum(p.weight for p in points if p.h3_index == h3_index))
