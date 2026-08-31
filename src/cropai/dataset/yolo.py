"""YOLO detection / segmentation writers."""

from __future__ import annotations

from pathlib import Path

from cropai.dataset.schema import DetectionAnnotation, ImageRecord


def _class_index(names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(names)}


def _collect_names(annotations: list[DetectionAnnotation], polygons: bool) -> list[str]:
    names: list[str] = []
    for ann in annotations:
        items = ann.polygons if polygons else ann.boxes
        for item in items:
            if item.class_id not in names:
                names.append(item.class_id)
    return names


def write_yolo_detection(
    out_dir: Path,
    records: list[ImageRecord],
    annotations: list[DetectionAnnotation],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = _collect_names(annotations, polygons=False) or ["leaf"]
    index = _class_index(names)
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(exist_ok=True)
    by_id = {a.sample_id: a for a in annotations}
    for rec in records:
        ann = by_id.get(rec.sample_id)
        lines: list[str] = []
        if ann:
            w = max(ann.width, 1)
            h = max(ann.height, 1)
            for box in ann.boxes:
                cx = (box.x + box.w / 2.0) / w
                cy = (box.y + box.h / 2.0) / h
                nw = box.w / w
                nh = box.h / h
                cls = index.get(box.class_id, 0)
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (labels_dir / f"{rec.sample_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (out_dir / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    yaml = [
        f"path: {out_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for i, name in enumerate(names):
        yaml.append(f"  {i}: {name}")
    (out_dir / "dataset.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    return out_dir / "dataset.yaml"


def write_yolo_segmentation(
    out_dir: Path,
    records: list[ImageRecord],
    annotations: list[DetectionAnnotation],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = _collect_names(annotations, polygons=True) or ["lesion"]
    index = _class_index(names)
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(exist_ok=True)
    by_id = {a.sample_id: a for a in annotations}
    for rec in records:
        ann = by_id.get(rec.sample_id)
        lines: list[str] = []
        if ann:
            w = max(ann.width, 1)
            h = max(ann.height, 1)
            for poly in ann.polygons:
                cls = index.get(poly.class_id, 0)
                coords: list[str] = []
                for x, y in poly.points:
                    coords.append(f"{x / w:.6f}")
                    coords.append(f"{y / h:.6f}")
                if coords:
                    lines.append(f"{cls} " + " ".join(coords))
        (labels_dir / f"{rec.sample_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (out_dir / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    return out_dir / "classes.txt"


def parse_yolo_label(path: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    """Return list of (cls, x, y, w, h) in pixels (xywh)."""
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, nw, nh = map(float, parts[1:5])
        bw = nw * width
        bh = nh * height
        x = cx * width - bw / 2
        y = cy * height - bh / 2
        boxes.append((cls, x, y, bw, bh))
    return boxes
