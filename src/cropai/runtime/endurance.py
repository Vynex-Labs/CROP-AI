"""Long-run harness. Unrun durations stay NOT RUN — never claimed."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from cropai.config.loader import load_runtime_config
from cropai.runtime.queue import BoundedInferenceQueue
from cropai.runtime.schema import NOT_MEASURED
from cropai.runtime.session import DeploySession, rss_bytes
from cropai.utils.logging import utc_now_iso


OFFICIAL = (5, 15, 30, 60)


def run_endurance(
    image: Path,
    *,
    seconds: float = 2.0,
    crop: str = "rice",
) -> dict[str, Any]:
    cfg = load_runtime_config()
    qcfg = dict(cfg.get("queue") or {})
    session = DeploySession()
    session.warmup(image, n=int((cfg.get("warmup") or {}).get("images", 1)))
    queue = BoundedInferenceQueue(maxsize=int(qcfg.get("maxsize", 8)), policy_when_full=str(qcfg.get("policy_when_full", "reject")))
    rss0 = rss_bytes()
    n_ok = n_fail = 0
    t_end = time.perf_counter() + max(0.2, float(seconds))
    totals: list[float] = []
    while time.perf_counter() < t_end:
        accepted = queue.submit(image)
        if not accepted:
            continue
        results = queue.drain(lambda p: session.run_once(p, crop=crop))
        for row in results:
            if row.get("vision", {}).get("reject_reason"):
                n_fail += 1
            else:
                n_ok += 1
            totals.append(float(row.get("timings_ms", {}).get("total") or 0.0))
    rss1 = rss_bytes()
    ran_min = float(seconds) / 60.0
    official_status = {f"{m}min": ("NOT RUN" if ran_min + 1e-6 < m else "RAN_IN_THIS_PROCESS") for m in OFFICIAL}
    return {
        "timestamp": utc_now_iso(),
        "requested_seconds": seconds,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "queue_rejected": queue.rejected,
        "queue_max_depth": queue.max_depth,
        "queue_unbounded": False,
        "rss_start_bytes": rss0,
        "rss_end_bytes": rss1,
        "rss_growth_bytes": (rss1 - rss0) if isinstance(rss0, int) and isinstance(rss1, int) else NOT_MEASURED,
        "mean_total_ms": round(sum(totals) / len(totals), 3) if totals else None,
        "official_durations": official_status,
        "gpu_util": NOT_MEASURED,
        "vram_growth": NOT_MEASURED,
        "thermal": NOT_MEASURED,
        "warnings": [
            "smoke_endurance_not_a_1_hour_test" if seconds < 300 else "check_official_durations_keys",
            "dummy_backend" if session.runtime.get("selected") == "dummy" else session.runtime.get("selected"),
        ],
        "status": "SMOKE" if seconds < 300 else "USER_DURATION",
    }
