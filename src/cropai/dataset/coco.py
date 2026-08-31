"""Minimal COCO detection writer / reader."""

from __future__ import annotations

import json
from pathlib import Path

from cropai.dataset.schema import DetectionAnnotation, ImageRecord
from cropai.utils.logging import utc_now_iso


def write_coco_detection(
    path: Path,
    records: list[ImageRecord],
    annotations: list[DetectionAnnotation],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for ann in annotations:
        for box in ann.boxes:
            if box.class_id not in names:
                names.append(box.class_id)
    categories = [{"id": i + 1, "name": n, "supercategory": "agri"} for i, n in enumerate(names)]
    cat_index = {n: i + 1 for i, n in enumerate(names)}
    images = []
    coco_anns = []
    ann_id = 1
    by_id = {a.sample_id: a for a in annotations}
    for i, rec in enumerate(records, start=1):
        images.append(
            {
                "id": i,
                "file_name": rec.relpath,
                "width": rec.width or 0,
                "height": rec.height or 0,
                "is_synthetic": rec.is_synthetic,
            }
        )
        ann = by_id.get(rec.sample_id)
        if not ann:
            continue
        for box in ann.boxes:
            coco_anns.append(
                {
                    "id": ann_id,
                    "image_id": i,
                    "category_id": cat_index.get(box.class_id, 1),
                    "bbox": [round(box.x, 3), round(box.y, 3), round(box.w, 3), round(box.h, 3)],
                    "area": round(max(box.w, 0) * max(box.h, 0), 3),
                    "iscrowd": 0,
                    "is_synthetic": True if rec.is_synthetic else False,
                }
            )
            ann_id += 1
        for poly in ann.polygons:
            flat = [c for pt in poly.points for c in pt]
            coco_anns.append(
                {
                    "id": ann_id,
                    "image_id": i,
                    "category_id": cat_index.get(poly.class_id, 1),
                    "segmentation": [flat],
                    "bbox": _poly_bbox(poly.points),
                    "area": _poly_area(poly.points),
                    "iscrowd": 0,
                    "is_synthetic": True if rec.is_synthetic else False,
                }
            )
            ann_id += 1
    payload = {
        "info": {
            "description": "CROP-AI annotations",
            "version": records[0].dataset_version if records else "unknown",
            "date_created": utc_now_iso(),
        },
        "licenses": [{"id": 1, "name": records[0].license if records else "unknown"}],
        "images": images,
        "annotations": coco_anns,
        "categories": categories,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _poly_bbox(points: list[tuple[float, float]]) -> list[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, y0 = min(xs), min(ys)
    return [x0, y0, max(xs) - x0, max(ys) - y0]


def _poly_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def load_coco(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
