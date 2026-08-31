from pathlib import Path

from cropai.dataset.leakage import detect_group_leakage, detect_hash_leakage
from cropai.dataset.schema import read_csv
from cropai.dataset.split import split_records
from cropai.dataset.synthetic import generate_smoke_dataset
from cropai.dataset.validate import DatasetValidator
from cropai.domain.taxonomy import Taxonomy


def test_synthetic_images_are_flagged(tmp_path: Path):
    records = generate_smoke_dataset(tmp_path, n_per_class=2, seed=1)
    assert records
    assert all(r.is_synthetic for r in records)
    assert all(not r.is_field for r in records)
    assert (tmp_path / "SYNTHETIC.txt").exists()
    assert (tmp_path / "yolo" / "detect" / "dataset.yaml").exists()
    assert (tmp_path / "coco" / "instances.json").exists()
    assert (tmp_path / "observations.jsonl").exists()
    assert (tmp_path / "traps.jsonl").exists()
    assert (tmp_path / "weather.jsonl").exists()
    for rec in records:
        assert (tmp_path / "images" / f"{rec.sample_id}.png").exists()


def test_split_does_not_leak_groups(tmp_path: Path):
    records = generate_smoke_dataset(tmp_path, n_per_class=3, seed=7)
    split = split_records(records, seed=7)
    leaks = detect_group_leakage(split)
    assert leaks == []
    splits = {r.split for r in split}
    assert "train" in splits
    # Tiny sets may not always have all three, but grouped assignment should still be valid.
    assert detect_hash_leakage(split) == []


def test_validator_accepts_synthetic_when_paths_resolvable(tmp_path: Path, monkeypatch):
    records = generate_smoke_dataset(tmp_path, n_per_class=2, seed=3)
    for rec in records:
        rec.relpath = str(tmp_path / "images" / f"{rec.sample_id}.png")
    split = split_records(records, seed=3)
    report = DatasetValidator(taxonomy=Taxonomy(), data_root=tmp_path).validate(split)
    assert report.ok, report.errors[:10]
    assert report.stats["n_field_real"] == 0
    assert any("No real field images" in w for w in report.warnings)


def test_validator_detects_missing_file(tmp_path: Path):
    records = generate_smoke_dataset(tmp_path, n_per_class=1, seed=4)
    records[0].relpath = str(tmp_path / "does_not_exist.png")
    report = DatasetValidator(data_root=tmp_path).validate(records)
    assert not report.ok
    assert any("missing file" in e for e in report.errors)


def test_csv_roundtrip(tmp_path: Path):
    from cropai.dataset.schema import write_csv

    records = generate_smoke_dataset(tmp_path, n_per_class=1, seed=5)
    csv_path = tmp_path / "m.csv"
    write_csv(csv_path, records)
    loaded = read_csv(csv_path)
    assert len(loaded) == len(records)
    assert loaded[0].is_synthetic is True
    assert loaded[0].sample_id == records[0].sample_id
