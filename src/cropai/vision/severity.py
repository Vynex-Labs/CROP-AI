"""Severity from lesion area. Segmentation is a measurement, not a diagnosis."""

from __future__ import annotations

from cropai.domain.taxonomy import Taxonomy


def polygon_area(points: list[list[float]] | list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += float(x1) * float(y2) - float(x2) * float(y1)
    return abs(area) / 2.0


def box_area(xywh: list[float]) -> float:
    if len(xywh) < 4:
        return 0.0
    return max(0.0, float(xywh[2])) * max(0.0, float(xywh[3]))


def affected_area_pct(
    *,
    masks: list[list[list[float]]],
    boxes: list[list[float]],
    image_w: int,
    image_h: int,
    roi_box: list[float] | None = None,
) -> tuple[float | None, str]:
    """Return (percent, method). Prefer mask; fall back to lesion-box / ROI."""
    denom = float(max(image_w, 1) * max(image_h, 1))
    if roi_box and box_area(roi_box) > 0:
        denom = box_area(roi_box)
    mask_area = sum(polygon_area(poly) for poly in masks)
    if mask_area > 0 and denom > 0:
        return round(100.0 * min(1.0, mask_area / denom), 3), "mask"
    lesion_boxes = [b for b in boxes if len(b) >= 4]
    if lesion_boxes and denom > 0:
        # Weak proxy — never treat as laboratory severity.
        return round(100.0 * min(1.0, sum(box_area(b) for b in lesion_boxes) / denom), 3), "box_proxy"
    return None, "none"


def severity_bin(affected_pct: float | None, taxonomy: Taxonomy | None = None) -> str:
    if affected_pct is None:
        return "unknown"
    tax = taxonomy or Taxonomy()
    return tax.severity_from_area(affected_pct).id
