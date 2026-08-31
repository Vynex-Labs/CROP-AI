from cropai.config.loader import (
    load_crop_config,
    load_dataset_config,
    load_inference_config,
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
