from cropai.config.loader import (
    load_crop_config,
    load_dataset_config,
    load_geo_config,
    load_inference_config,
    load_risk_config,
    load_runtime_config,
    load_training_config,
)


def test_core_yaml_loads():
    crop = load_crop_config()
    data = load_dataset_config()
    train = load_training_config()
    inf = load_inference_config()
    assert crop["problem"]["id"] == "SIH26131"
    assert "plantvillage" in data["sources"]
    assert data["sources"]["field_maharashtra"]["status"] == "protocol_only_no_images"
    assert data["sources"]["cropsap"]["id"] == "cropsap_maharashtra"
    assert data["sources"]["cropsap"]["status"] == "not_available"
    assert train["device"] == "auto"
    assert inf["offline_first"] is True
    assert inf["failure_policy"]["network"] == "offline_queue_sync"
    risk = load_risk_config()
    assert risk["candidate"] == "lightgbm"
    assert risk["runtime_default"] == "heuristic_unvalidated"
    assert risk["calibration"]["method"] == "none"
    geo = load_geo_config()
    assert geo["fusion"]["candidate"] == "weighted_deterministic"
    assert geo["gnn"]["allowed"] is False
    runtime = load_runtime_config()
    assert runtime["precision"]["int8"]["adopted"] is False
    assert runtime["precision"]["gpu_target"] == "fp16"
    from cropai.validate.report import architecture_freeze

    freeze = architecture_freeze()
    assert freeze["frozen"] is False
