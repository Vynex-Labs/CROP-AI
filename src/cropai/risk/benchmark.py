"""Phase 3.5 comparison harness. Does not invent winners."""

from __future__ import annotations

import argparse
import json

from cropai.config.loader import load_risk_config, load_training_config
from cropai.risk.backends import describe_stack
from cropai.risk.tables import count_real_outbreak_labels
from cropai.risk.train import plan
from cropai.utils.paths import data_dir


CANDIDATES = ["lightgbm", "xgboost", "logistic_baseline", "heuristic_unvalidated"]

METRICS = [
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "macro_f1",
    "brier",
    "calibration",
    "fnr",
    "lead_time",
    "latency",
    "missing_data_robustness",
    "explainability",
]


def benchmark_plan() -> dict:
    tcfg = load_training_config()
    rcfg = load_risk_config()
    raw = data_dir() / "raw"
    n_real = count_real_outbreak_labels(list(raw.glob("**/*obs*.jsonl")) if raw.exists() else [])
    rows = []
    for name in CANDIDATES:
        row = {
            "task": "risk_forecast",
            "candidate": name,
            "selected": "NOT SELECTED — pending measured results on real outbreak labels",
        }
        for m in METRICS:
            row[m] = "NOT MEASURED"
        row["reason"] = "Zero real outbreak labels in this clone; synthetic smoke is not skill."
        rows.append(row)
    supports_ml = n_real >= int((rcfg.get("training") or {}).get("min_real_labels", 200))
    return {
        "stack": describe_stack(),
        "training_defaults": tcfg.get("risk"),
        "runtime_default": rcfg.get("runtime_default"),
        "n_real_outbreak_labels": n_real,
        "comparison_rows": rows,
        "train_plan": plan(),
        "answers": {
            "is_lightgbm_best": "UNKNOWN — not measured",
            "is_xgboost_best": "UNKNOWN — not measured",
            "does_data_support_ml_forecasting": (
                "YES — label count meets configured minimum (training still not run here)"
                if supports_ml
                else "NO — 0 real outbreak labels. MASTER_PROMPT §7: do not fabricate; use statistical/heuristic fallback."
            ),
            "selected_risk_model": "NOT SELECTED",
            "runtime_fallback": "heuristic_unvalidated (uncalibrated, unvalidated)",
        },
        "status": "HARNESS_READY_METRICS_PENDING_REAL_LABELS",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args(argv)
    plan_out = benchmark_plan()
    plan_out["dry_run"] = args.dry_run
    print(json.dumps(plan_out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
