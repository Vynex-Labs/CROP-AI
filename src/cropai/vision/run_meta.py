"""Reproducibility sidecar for every training run."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

from cropai.utils.hardware import detect_hardware
from cropai.utils.logging import utc_now_iso
from cropai.utils.paths import repo_root


def git_commit() -> str:
    head = repo_root() / ".git" / "HEAD"
    if not head.exists():
        return ""
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref:"):
        ref = repo_root() / ".git" / text.split(" ", 1)[1].strip()
        return ref.read_text(encoding="utf-8").strip() if ref.exists() else text
    return text


def collect_run_meta(**extra: Any) -> dict[str, Any]:
    hw = detect_hardware()
    meta = {
        "timestamp": utc_now_iso(),
        "git_commit": git_commit(),
        "hardware": hw,
        "os": platform.platform(),
        "python": platform.python_version(),
        **extra,
    }
    return meta


def write_run_meta(run_dir: Path, meta: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "meta.json"
    path.write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    return path
