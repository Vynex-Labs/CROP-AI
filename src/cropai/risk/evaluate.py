"""Evaluate risk scores. Refuses to invent ROC-AUC when labels are missing."""

from __future__ import annotations

import argparse
import json

from cropai.risk.metrics import report
from cropai.risk.tables import count_real_outbreak_labels
from cropai.utils.paths import data_dir


def evaluate_available() -> dict:
    raw = data_dir() / "raw"
    paths = list(raw.glob("**/*obs*.jsonl")) if raw.exists() else []
    n_real = count_real_outbreak_labels(paths)
    empty = report([], [], source="none")
    empty["n_real_outbreak_labels"] = n_real
    empty["status"] = "NOT MEASURED"
    empty["reason"] = (
        "No real outbreak labels with paired forecasts in this clone."
        if n_real == 0
        else "Labels exist on disk but no held-out scored set was provided."
    )
    return empty


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="")
    parser.add_argument("--scores", default="")
    args = parser.parse_args(argv)
    if not args.labels or not args.scores:
        print(json.dumps(evaluate_available(), indent=2))
        return 0
    # Explicit paired files only — still tag source from filename.
    y_true: list[int] = []
    y_score: list[float] = []
    from pathlib import Path

    for line in Path(args.labels).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            y_true.append(int(float(line.split(",")[-1])))
    for line in Path(args.scores).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            y_score.append(float(line.split(",")[-1]))
    source = "synthetic" if "synthetic" in args.labels else "user_provided"
    print(json.dumps(report(y_true, y_score, source=source), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
