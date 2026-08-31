"""Phase 4.5 comparison harness. Does not invent hotspot accuracy."""

from __future__ import annotations

import json

from cropai.config.loader import load_geo_config
from cropai.geo.h3index import describe_h3


def benchmark_plan() -> dict:
    cfg = load_geo_config()
    metrics = [
        "hotspot_precision",
        "hotspot_recall",
        "false_hotspot_rate",
        "missed_hotspot_rate",
        "spatial_stability",
        "temporal_stability",
        "advisory_consistency",
        "expert_referral_correctness",
    ]
    rows = []
    for name in ["h3+kde+dbscan", "h3_aggregation_only", "gnn"]:
        row = {
            "task": "hotspot_detection",
            "candidate": name,
            "selected": "NOT SELECTED — pending labeled hotspot evaluation",
        }
        for m in metrics:
            row[m] = "NOT MEASURED"
        if name == "gnn":
            row["selected"] = "REJECTED unless measured benefit — MASTER_PROMPT §8"
            row["reason"] = "GNN not in scope without evidence."
        else:
            row["reason"] = "No labeled Maharashtra hotspots in this clone."
        rows.append(row)
    return {
        "h3": describe_h3(),
        "config": {
            "decay_half_life_days": (cfg.get("decay") or {}).get("half_life_days"),
            "dbscan": cfg.get("dbscan"),
            "fusion": (cfg.get("fusion") or {}).get("candidate"),
        },
        "comparison_rows": rows,
        "answers": {
            "is_h3_kde_dbscan_sufficient": "UNKNOWN — qualitative tests pass; no labeled hotspot set",
            "selected_spatial_model": "NOT SELECTED",
            "runtime_methods": "grid_or_h3 + dbscan + kde + time_decay (unvalidated scores)",
            "gnn": "not introduced",
            "fusion_model": "weighted_deterministic (uncalibrated, unfitted)",
        },
        "status": "HARNESS_READY_METRICS_PENDING_LABELED_HOTSPOTS",
    }


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(benchmark_plan(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
