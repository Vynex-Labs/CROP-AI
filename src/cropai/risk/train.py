#!/usr/bin/env python3
"""Phase 3 risk training entrypoint.

  python -m cropai.risk.train --dry-run
  python -m cropai.risk.train --smoke

Does not fabricate outbreak skill. Refuses ML fit when real labels are absent
unless --smoke (synthetic-only loop check).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cropai.config.loader import load_risk_config, load_training_config
from cropai.risk.backends import describe_stack, lightgbm_available, sklearn_available, xgboost_available
from cropai.risk.tables import count_real_outbreak_labels, generate_synthetic_weather_series
from cropai.utils.hardware import detect_hardware, format_hardware_report
from cropai.utils.logging import setup_logging
from cropai.utils.paths import data_dir, ensure_runtime_dirs, runs_dir

log = setup_logging()


def plan() -> dict:
    tcfg = load_training_config()
    rcfg = load_risk_config()
    obs_paths = list((data_dir() / "raw").glob("**/*obs*.jsonl")) if (data_dir() / "raw").exists() else []
    n_real = count_real_outbreak_labels(obs_paths)
    min_real = int((rcfg.get("training") or {}).get("min_real_labels", 200))
    return {
        "task": "risk_forecast",
        "candidate": (tcfg.get("risk") or {}).get("candidate", "lightgbm"),
        "alternatives": (tcfg.get("risk") or {}).get("alternatives", ["xgboost", "logistic_baseline"]),
        "runtime_default": rcfg.get("runtime_default"),
        "horizons_days": (tcfg.get("risk") or {}).get("horizons_days", [1, 3, 7]),
        "n_real_outbreak_labels": n_real,
        "min_real_labels": min_real,
        "lightgbm": lightgbm_available(),
        "xgboost": xgboost_available(),
        "sklearn": sklearn_available(),
        "stack": describe_stack(),
        "hardware": format_hardware_report(detect_hardware()),
        "status": "NOT_TRAINED",
        "reason": (
            "No real outbreak labels in this clone. MASTER_PROMPT §7: do not fabricate. "
            "Runtime fallback is heuristic_unvalidated (uncalibrated, unvalidated)."
            if n_real < min_real
            else "Labels present but training not executed in this process."
        ),
        "calibration": "none",
    }


def _smoke_fit() -> dict:
    """Optional loop check on SYNTHETIC random labels. Not agricultural skill."""
    if not lightgbm_available():
        return {
            "ok": False,
            "status": "SMOKE_SKIPPED",
            "reason": "lightgbm not installed",
            "source": "synthetic_random_labels",
            "metrics": "NOT MEASURED — synthetic smoke, package missing",
        }
    import random

    try:
        import lightgbm as lgb
    except Exception as exc:
        return {"ok": False, "status": "SMOKE_SKIPPED", "reason": str(exc)}

    rng = random.Random(0)
    x = [[rng.random() for _ in range(8)] for _ in range(40)]
    y = [rng.randint(0, 1) for _ in range(40)]
    ds = lgb.Dataset(x, label=y)
    booster = lgb.train(
        {"objective": "binary", "verbosity": -1, "num_leaves": 8},
        ds,
        num_boost_round=5,
    )
    out = runs_dir() / "risk" / "smoke_synthetic.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out))
    return {
        "ok": True,
        "status": "SMOKE_SYNTHETIC_ONLY",
        "checkpoint": str(out),
        "source": "synthetic_random_labels",
        "metrics": "NOT MEASURED as outbreak skill",
        "warning": "Do not report this smoke fit as forecast accuracy.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CROP-AI Phase 3 risk training")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--backend", default="lightgbm")
    args = parser.parse_args(argv)

    ensure_runtime_dirs()
    print(format_hardware_report(detect_hardware()))
    print("---")
    result = plan()
    result["backend_requested"] = args.backend
    if args.dry_run:
        result["dry_run"] = True
        print(json.dumps(result, indent=2))
        return 0
    if args.smoke:
        generate_synthetic_weather_series()
        result["smoke"] = _smoke_fit()
        print(json.dumps(result, indent=2))
        return 0
    if result["n_real_outbreak_labels"] < result["min_real_labels"]:
        log.warning("refusing ML risk training: %s", result["reason"])
        print(json.dumps(result, indent=2))
        return 0
    result["status"] = "NOT_EXECUTED"
    result["reason"] = "Real-label training belongs on a data-authorized machine; not run here."
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
