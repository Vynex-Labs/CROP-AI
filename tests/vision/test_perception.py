from pathlib import Path

from PIL import Image

from cropai.vision.metrics import accuracy, classification_report, detection_prf, iou_box, mae_rmse
from cropai.vision.pipeline import PerceptionPipeline
from cropai.vision.preprocess import assess_quality, load_image
from cropai.vision.schema import VisionOutput
from cropai.vision.severity import affected_area_pct, polygon_area, severity_bin


def test_invalid_image_rejected(tmp_path: Path):
    missing = tmp_path / "nope.jpg"
    out = PerceptionPipeline().infer(missing)
    assert out.reject_reason
    assert out.expert_referral is True
    assert out.disease_class == "unknown"
    assert out.confidence == 0.0


def test_corrupt_image_rejected(tmp_path: Path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not-an-image")
    img, q = load_image(bad)
    assert img is None
    assert q.action == "reject"


def test_pipeline_dummy_on_synthetic(tmp_path: Path):
    from cropai.dataset.synthetic import generate_smoke_dataset

    recs = generate_smoke_dataset(tmp_path, n_per_class=1, seed=1)
    img = tmp_path / "images" / f"{recs[0].sample_id}.png"
    out = PerceptionPipeline().infer(img, crop_hint=recs[0].crop_id)
    assert out.reject_reason == ""
    assert out.expert_referral is True  # untrained dummy
    assert "untrained" in out.extras
    assert out.object_ids
    assert out.bounding_boxes
    assert out.model_version
    assert out.timestamp
    d = out.to_dict()
    for key in [
        "timestamp",
        "crop",
        "plant_or_leaf_id",
        "object_ids",
        "bounding_boxes",
        "object_classes",
        "disease_class",
        "disease_probabilities",
        "segmentation_masks",
        "affected_area",
        "severity_estimate",
        "confidence",
        "model_version",
    ]:
        assert key in d


def test_low_confidence_forces_referral():
    out = VisionOutput.rejected(reason="low_confidence", model_version="test")
    assert out.expert_referral is True


def test_quality_ok_on_normal_image(tmp_path: Path):
    p = tmp_path / "ok.png"
    Image.new("RGB", (128, 128), (20, 140, 40)).save(p)
    img, q = load_image(p)
    assert img is not None
    assert q.width == 128


def test_tiny_image_rejected(tmp_path: Path):
    p = tmp_path / "tiny.png"
    Image.new("RGB", (8, 8), (10, 10, 10)).save(p)
    _, q = load_image(p)
    assert q.ok is False


def test_metrics_toy():
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    r = classification_report(y_true, y_pred)
    assert 0 < r["accuracy"] < 1
    assert "a" in r["per_class"]
    assert accuracy(y_true, y_true) == 1.0
    assert iou_box([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    pr = detection_prf([("leaf", [0, 0, 10, 10])], [("leaf", [1, 1, 10, 10])])
    assert pr["tp"] == 1
    sev = mae_rmse([10.0, 20.0], [12.0, 18.0])
    assert sev["mae"] == 2.0


def test_severity_from_mask():
    mask = [[[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]]
    assert polygon_area(mask[0]) == 100.0
    pct, method = affected_area_pct(masks=mask, boxes=[], image_w=100, image_h=100)
    assert method == "mask"
    assert pct == 1.0
    assert severity_bin(0.0) == "healthy"
    assert severity_bin(40) == "severe"


def test_train_dry_run():
    from cropai.vision.train import main

    assert main(["--dry-run", "--task", "classifier"]) == 0


def test_benchmark_plan_does_not_invent_scores():
    from cropai.vision.benchmark import benchmark_plan

    plan = benchmark_plan()
    assert plan["answers"]["is_efficientnetv2s_best"].startswith("UNKNOWN")
    for row in plan["comparison_rows"]:
        assert row["accuracy"] == "NOT MEASURED"
        assert "NOT SELECTED" in row["selected"]
