from pathlib import Path

from PIL import Image

from cropai.runtime.benchmark import benchmark_plan
from cropai.runtime.endurance import run_endurance
from cropai.runtime.export_plan import export_plan, tensorrt_export
from cropai.runtime.profile import profile_pipeline
from cropai.runtime.queue import BoundedInferenceQueue
from cropai.runtime.select import select_runtime
from cropai.runtime.session import DeploySession


def _png(tmp_path: Path) -> Path:
    p = tmp_path / "leaf.png"
    Image.new("RGB", (96, 96), (20, 140, 40)).save(p)
    return p


def test_select_runtime_is_dummy_without_weights():
    sel = select_runtime()
    assert sel["selected"] == "dummy"
    assert sel["int8_adopted"] is False
    assert "dummy" in sel["order"]


def test_queue_rejects_when_full():
    q = BoundedInferenceQueue(maxsize=2, policy_when_full="reject")
    assert q.submit("a")
    assert q.submit("b")
    assert q.submit("c") is False
    assert q.rejected == 1
    assert len(q) == 2
    out = q.drain(lambda x: x.upper())
    assert out == ["A", "B"]
    assert len(q) == 0


def test_export_plan_dry_run_does_not_invent_success():
    plan = export_plan(dry_run=True)
    assert plan["int8_adopted"] is False
    assert plan["dry_run"] is True
    assert any(s["step"] == "onnx_export" and s["status"] == "NOT RUN" for s in plan["steps"])
    trt = tensorrt_export(Path("missing.onnx"), Path("x.engine"))
    assert trt["ok"] is False


def test_benchmark_plan_honest():
    plan = benchmark_plan()
    assert plan["answers"]["int8_adopted"] is False
    assert plan["long_run"]["60min"] == "NOT RUN"
    for row in plan["comparison_rows"]:
        assert row["yolo_latency"] == "NOT MEASURED"
        assert row["fps"] == "NOT MEASURED"


def test_profile_dummy_does_not_quote_fps(tmp_path: Path):
    img = _png(tmp_path)
    report = profile_pipeline(img, crop="rice", repeats=2, warmup=1)
    assert report.backend == "dummy"
    assert report.int8_adopted is False
    assert report.gpu_util == "NOT MEASURED"
    d = report.to_dict()
    assert d["fps"] == "NOT MEASURED"
    assert report.timings_ms["total"] > 0
    assert "dummy_timings_are_not_yolo_or_efficientnet_fps" in " ".join(
        DeploySession().run_once(img, crop="rice")["warnings"]
    )


def test_endurance_smoke_marks_hour_not_run(tmp_path: Path):
    img = _png(tmp_path)
    out = run_endurance(img, seconds=0.6, crop="rice")
    assert out["official_durations"]["60min"] == "NOT RUN"
    assert out["official_durations"]["5min"] == "NOT RUN"
    assert out["queue_unbounded"] is False
    assert out["n_ok"] + out["n_fail"] >= 1
    assert out["gpu_util"] == "NOT MEASURED"
