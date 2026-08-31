from pathlib import Path

from cropai.dataset.coco import load_coco, write_coco_detection
from cropai.dataset.synthetic import generate_smoke_dataset
from cropai.dataset.yolo import parse_yolo_label, write_yolo_detection


def test_yolo_boxes_are_normalized_and_parseable(tmp_path: Path):
    records = generate_smoke_dataset(tmp_path, n_per_class=1, seed=9)
    # generator already wrote yolo labels; parse one with boxes
    labels = list((tmp_path / "yolo" / "detect" / "labels").glob("*.txt"))
    assert labels
    nonempty = [p for p in labels if p.read_text(encoding="utf-8").strip()]
    assert nonempty
    boxes = parse_yolo_label(nonempty[0], 96, 96)
    assert boxes
    for cls, x, y, w, h in boxes:
        assert w > 0 and h > 0
        assert x + w >= 0


def test_coco_contains_synthetic_flag(tmp_path: Path):
    records = generate_smoke_dataset(tmp_path, n_per_class=1, seed=8)
    coco = load_coco(tmp_path / "coco" / "instances.json")
    assert coco["images"]
    assert coco["categories"]
    assert any(im.get("is_synthetic") for im in coco["images"])


def test_invalid_yolo_bbox_detected(tmp_path: Path):
    from cropai.dataset.schema import ImageRecord
    from cropai.dataset.validate import DatasetValidator

    rec = ImageRecord(
        sample_id="bad_box",
        relpath=str(tmp_path / "x.png"),
        crop_id="rice",
        class_id="rice_blast",
        is_synthetic=True,
        width=96,
        height=96,
        annotation_path=str(tmp_path / "bad.txt"),
        growth_stage="vegetative",
        symptom_region="leaf",
    )
    from PIL import Image

    Image.new("RGB", (96, 96), (0, 80, 0)).save(tmp_path / "x.png")
    (tmp_path / "bad.txt").write_text("0 2.5 2.5 0.1 0.1\n", encoding="utf-8")
    report = DatasetValidator(data_root=tmp_path).validate([rec])
    assert not report.ok
    assert report.stats["bbox_problems"] >= 1
