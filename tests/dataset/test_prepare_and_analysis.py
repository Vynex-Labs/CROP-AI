from cropai.dataset.analysis import analyze_manifest
from cropai.dataset.prepare import prepare_dataset
from cropai.dataset.schema import read_csv
from cropai.utils.paths import data_dir


def test_prepare_synthetic_only_end_to_end():
    summary = prepare_dataset(synthetic_only=True, seed=11, dataset_version="v0.1.0-test")
    assert summary["n_records"] > 0
    assert summary["n_synthetic"] == summary["n_records"]
    assert summary["n_field_real"] == 0
    assert summary["validation_ok"] is True
    splits_dir = data_dir() / "splits" / "v0.1.0-test"
    train = read_csv(splits_dir / "train.csv")
    assert all(r.is_synthetic for r in train)
    assert (data_dir() / "synthetic" / "smoke" / "SYNTHETIC.txt").exists()


def test_analysis_does_not_claim_field_data():
    summary = prepare_dataset(synthetic_only=True, seed=12, dataset_version="v0.1.0-test2")
    from cropai.dataset.schema import read_csv
    from cropai.utils.paths import data_dir as dd

    records = read_csv(dd() / "splits" / "v0.1.0-test2" / "train.csv")
    records += read_csv(dd() / "splits" / "v0.1.0-test2" / "val.csv")
    records += read_csv(dd() / "splits" / "v0.1.0-test2" / "test.csv")
    analysis = analyze_manifest(records)
    assert analysis["n_field_real"] == 0
    assert analysis["gate"]["additional_field_images_required"] is True
    assert analysis["gate"]["public_plus_field_sufficient_for_maharashtra_deployment"] is False
    assert analysis["gate"]["public_plus_field_sufficient_for_transfer_learning_start"] is False
    assert analysis["gate"]["fabricated_field_data"] is False
    assert summary["n_field_real"] == 0
