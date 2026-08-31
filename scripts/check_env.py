#!/usr/bin/env python3
"""Environment validation shared by train_linux.sh and train_windows.bat."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cropai.utils.hardware import detect_hardware, format_hardware_report  # noqa: E402
from cropai.utils.paths import ensure_runtime_dirs, repo_root  # noqa: E402


REQUIRED_CONFIGS = [
    "configs/crop_config.yaml",
    "configs/dataset.yaml",
    "configs/training.yaml",
    "configs/inference.yaml",
    "configs/risk.yaml",
    "MASTER_PROMPT.md",
]

PHASE1_PY = [
    "yaml",
    "PIL",
]


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def main() -> int:
    os.chdir(ROOT)
    print("=== CROP-AI environment check ===")
    print(f"repo: {repo_root()}")
    print(f"python: {sys.version.split()[0]}  executable={sys.executable}")

    missing_cfg = [p for p in REQUIRED_CONFIGS if not (ROOT / p).exists()]
    if missing_cfg:
        for p in missing_cfg:
            _fail(f"missing {p}")
        return 1
    _ok("required config files present")

    missing_py = []
    for mod in PHASE1_PY:
        try:
            __import__(mod)
        except Exception:
            missing_py.append(mod)
    if missing_py:
        _fail(f"missing Python modules {missing_py}. Install: pip install -r requirements-dev.txt")
        return 1
    _ok("phase-1 Python modules import")

    hw = detect_hardware()
    print("--- hardware ---")
    print(format_hardware_report(hw))
    if not hw.get("cuda_available"):
        _warn("CUDA unavailable. Training scripts will refuse silent GPU-less heavy training.")
    else:
        _ok(f"CUDA GPU {hw.get('gpu_name')}")

    dirs = ensure_runtime_dirs()
    _ok("runtime directories ready: " + ", ".join(sorted(dirs)))

    data_raw = dirs["raw"]
    public = False
    if data_raw.exists():
        public = any(
            p.is_file() and p.name not in {".gitkeep", ".gitignore"}
            for p in data_raw.rglob("*")
        )
    if not public:
        _warn("data/raw/ is empty. Dataset prep will use SYNTHETIC smoke data only.")
    else:
        _ok("data/raw/ contains files")

    train_mod = ROOT / "src" / "cropai" / "vision" / "train.py"
    risk_mod = ROOT / "src" / "cropai" / "risk" / "train.py"
    print(
        json.dumps(
            {
                "cuda_available": hw.get("cuda_available"),
                "torch_installed": hw.get("torch_installed"),
                "phase2_train_script": train_mod.exists(),
                "phase3_risk_script": risk_mod.exists(),
                "data_raw_populated": public,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
