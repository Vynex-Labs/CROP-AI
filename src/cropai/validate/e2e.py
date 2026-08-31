"""End-to-end smoke. Explicitly not the Phase 6 field benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cropai.runtime.session import DeploySession


def smoke_e2e(image: Path, crop: str = "rice") -> dict[str, Any]:
    session = DeploySession()
    row = session.run_once(image, crop=crop)
    return {
        "kind": "synthetic_or_untrained_smoke",
        "warning": "This is not a Phase 6 field benchmark. Do not quote as mAP/FPS/AUC.",
        "runtime": row.get("runtime"),
        "timings_ms": row.get("timings_ms"),
        "vision_referral": row.get("vision", {}).get("expert_referral"),
        "advisory_chemicals": row.get("advisory", {}).get("chemical_control_allowed"),
        "dashboard": "NOT IMPLEMENTED",
    }
