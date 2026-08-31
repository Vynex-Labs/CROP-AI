from cropai.validate.checklist import checklist
from cropai.validate.main import main
from cropai.validate.report import final_report


def test_no_checklist_item_is_verified():
    items = checklist()
    assert items
    assert all(i["status"] != "VERIFIED" for i in items)


def test_final_report_does_not_invent_metrics():
    r = final_report()
    assert r["architecture_frozen"] is False
    assert r["project_complete"] is False
    assert r["n_verified"] == 0
    assert r["vision_benchmark"]["detection"]["mAP50"] == "NOT MEASURED"
    assert r["forecast_benchmark"]["roc_auc"] == "NOT MEASURED"
    assert r["spatial_benchmark"]["hotspot_precision"] == "NOT MEASURED"
    assert r["system_benchmark"]["images_per_sec"] == "NOT MEASURED"
    assert "architecture_not_frozen" in r["blockers"]
    assert r["answers"]["dashboard"] == "NOT IMPLEMENTED"


def test_cli_nonzero_when_incomplete():
    assert main([]) == 3
